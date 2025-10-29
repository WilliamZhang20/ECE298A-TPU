import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
import numpy as np
# import HW simulation CocoTB
import cocotb
from cocotb.clock import Clock

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

def get_quantized_model():
    torch.backends.quantized.engine = 'fbgemm'

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    model = FCNet()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = prepare_qat(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("Training begins")
    model.train()
    for epoch in range(5):                     # short training – enough for demo
        for images, labels in train_loader:
            images = images.view(-1, 784)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    print("Training over")
    return convert(model)

@cocotb.test()
async def tpu_matmul_test(dut):
    # build model
    model = get_quantized_model()

    # compile it with backend
    from torch_backend import make_backend
    backend = make_backend(dut)                 
    compiled_model = torch.compile(model, backend=backend)

    # Load a few samples
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=5, shuffle=False)
    images, labels = next(iter(test_loader))
    images = images.view(-1, 784)  # (5, 784)

    # Run model on DUT
    with torch.no_grad():
        dut_out = compiled_model(images)       # goes through the systolic array

    # Run the good CPU model
    cpu_out = model(images)

    # Compare
    diff = (dut_out - cpu_out).abs()
    max_err = diff.max().item()
    assert max_err < 2.0, f"Max error {max_err} too large!"

    print(f"Test passed – max error = {max_err:.3f}")