from test_tpu import matmul
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import numpy as np

"""
The coolest part: setting up QAT in PyTorch
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
import numpy as np

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.dequant_act = DeQuantStub()
        self.activation_quant = QuantStub()
        self.fc2 = nn.Linear(128, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dequant_act(x)
        x = self.activation_quant(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

def prepare_model(model, qconfig='fbgemm'):
    model.qconfig = torch.quantization.get_default_qat_qconfig(qconfig)
    model = prepare_qat(model, inplace=False)
    return model

def train_model(model, train_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 784)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

# Set quantization backend
torch.backends.quantized.engine = 'fbgemm'

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Train and quantize model
model = FCNet()
model = prepare_model(model, qconfig='fbgemm')
train_model(model, train_loader)

# Convert to quantized model
model.eval()
model = convert(model)

# Extract scales and weights
weights = {}
weights['fc1'] = model.fc1.weight().int_repr().numpy().astype(np.int8)
weights['fc2'] = model.fc2.weight().int_repr().numpy().astype(np.int8)

# Get scales from quantization parameters
fc1_activation_scale = model.activation_quant.scale.item()
scales = {
    'fc1_activation_scale': fc1_activation_scale,
    'fc2_output_scale': 1.0  # No quantization on output
}

model_data = {
    'weights': weights,
    'scales': scales
}
torch.save(model_data, 'tpu/qat_model.pt')

# Save test data
test_images, test_labels = next(iter(test_loader))
test_images = test_images[:3].view(-1, 784)
scale = 255 / 1.0
zero_point = -128
test_images_int8 = torch.clamp(torch.round(test_images * scale + zero_point), -128, 127).to(torch.int8).numpy()
test_labels = test_labels[:3].numpy()
np.savez('tpu/mnist_test_data.npz', images=test_images_int8, labels=test_labels)

# Test accuracy
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 784)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Save test images and labels for cocotb (first 3 images for demo)
test_images, test_labels = next(iter(test_loader))
test_images = test_images[:3].view(-1, 784)
# Quantize images to int8: [0, 1] -> [-128, 127]
scale = 255 / 1.0
zero_point = -128
test_images_int8 = torch.clamp(torch.round(test_images * scale + zero_point), -128, 127).to(torch.int8).numpy()
test_labels = test_labels[:3].numpy()
np.savez('tpu/mnist_test_data.npz', images=test_images_int8, labels=test_labels)

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

##########################################################################################
###### Ran Neural Network Training in global space so time is not counted in DUT Sim #####

@cocotb.test
async def test_neural_network_inference(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Load PyTorch QAT weights
    try:
        model_data = torch.load('tpu/qat_model.pt', weights_only=False)
        weights = model_data['weights']
        scales = model_data['scales']
    except Exception as e:
        dut._log.error(f"Failed to load weights: {str(e)}")
        raise
    W1 = weights['fc1']  # 784x128, int8
    W2 = weights['fc2']  # 128x10, int8
    fc1_activation_scale = scales['fc1_activation_scale']

    # Load MNIST test data
    try:
        data = np.load('tpu/mnist_test_data.npz')
        test_images = data['images']  # 3x784, int8
        test_labels = data['labels']  # 3, int32
    except Exception as e:
        dut._log.error(f"Failed to load test data: {str(e)}")
        raise

    num_test_images = test_images.shape[0]
    correct = 0

    for img_idx in range(num_test_images):
        dut._log.info(f"Processing test image {img_idx + 1}/{num_test_images}")
        X = test_images[img_idx:img_idx+1, :]  # 1x784, int8

        Z1_raw = await matmul(dut, X, W1, transpose=True, relu=True)  # 1x128

        Z1_quantized = quantize_activation(Z1_raw, fc1_activation_scale, zero_point=0)

        # Layer 2: Z2 = Z1_quantized @ W2
        Z2_raw = await matmul(dut, Z1_quantized, W2, transpose=True, relu=False)

        # Dequantize final output for prediction
        Z2_scaled = Z2_raw.astype(np.float32) * fc1_activation_scale
        predicted_digit = np.argmax(Z2_scaled)
        true_label = test_labels[img_idx]
        if predicted_digit == true_label:
            correct += 1
        dut._log.info(f"Image {img_idx + 1}: Predicted digit = {predicted_digit}, True label = {true_label}")

    accuracy = 100 * correct / num_test_images
    dut._log.info(f"MNIST inference test passed! Accuracy: {accuracy:.2f}% (tested {num_test_images} images)")
