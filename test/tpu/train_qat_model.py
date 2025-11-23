import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.export
from torchao.quantization.pt2e.quantize_pt2e import (
  prepare_qat_pt2e, 
  convert_pt2e
)
from torchao.quantization.pt2e import move_exported_model_to_eval, move_exported_model_to_train

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
  get_symmetric_quantization_config,
  XNNPACKQuantizer,
)
from test_tpu import reset_dut
import cocotb
from cocotb.clock import Clock
import numpy as np

BATCH_SIZE = 32

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def compute_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def get_quantized_model():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    model = FCNet()

    from torch.export import Dim
    import random
    example_inputs = (torch.randn(random.randint(2, 8), 1, 28, 28),)
    batch = Dim("batch", min=1, max=2048)
    dynamic_shapes = ({0: batch},)
    exported_program = torch.export.export(
        model,
        example_inputs,
        dynamic_shapes=dynamic_shapes
    )

    # --- THE FULL TORCHAO PT2E QUANTIZATION FLOW ---

    print("Preparing exported graph for QAT using XNNPACKQuantizer...")
    
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())

    prepared_model = prepare_qat_pt2e(exported_program.module(), quantizer) 
    # --------------------------------------

    # 2.4. QAT Training Loop
    print("Training begins")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(prepared_model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(3):
        prepared_model = move_exported_model_to_train(prepared_model)

        for images, labels in train_loader:
            optimizer.zero_grad()
            out = prepared_model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        # ---- Accuracy after epoch ----
        prepared_model = move_exported_model_to_eval(prepared_model)
        acc = compute_accuracy(prepared_model, train_loader)
        print(f"Epoch {epoch+1} training accuracy: {acc:.4f}")

    print("Training over")

    # --- FULL TORCHAO PT2E Convert Step ---
    print("Converting prepared model to quantized model...")
    quantized_model = convert_pt2e(prepared_model)
    # --------------------------------------
    
    # Return the converted model (now a quantized GraphModule)
    return quantized_model

# 3. Hardware Test Function 
@cocotb.test()
async def tpu_torch_test(dut):
    # build model
    model = get_quantized_model()
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # compile it with backend
    from torch_backend import make_backend
    backend = make_backend(dut) 
    
    # torch.compile compiles the resulting GraphModule/quantized module
    compiled_model = torch.compile(model, backend=backend)

    # Load a few samples
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    for i, (image, label) in enumerate(test_loader):
        if i >= 5:
            break
        # RUN INFERENCE IN SEPARATE THREAD
        import concurrent.futures
        from cocotb.triggers import Timer

        def run_inference():
            with torch.no_grad():
                return compiled_model(image)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_inference)

            # POLL SIMULATOR WHILE WAITING
            while not future.done():
                await Timer(10, units='ns')  # Keep cocotb alive

            dut_out = future.result()

            print("Predicted Label:", torch.argmax(dut_out, dim=1).item())
            print("Actual label:", label.item())

    print("TEST PASSED")