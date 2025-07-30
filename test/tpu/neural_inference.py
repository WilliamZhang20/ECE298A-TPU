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

def calculate_activation_scale(model, data_loader, layer_name):
    """Calculate the scale for quantizing activations after a specific layer"""
    model.eval()
    max_vals = []
    
    # Hook to capture intermediate activations
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hook
    if layer_name == 'fc1':
        hook = model.fc1.register_forward_hook(hook_fn('fc1'))
    
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= 100:  # Sample first 100 batches for scale calculation
                break
            images = images.view(-1, 784)
            _ = model(images)
            
            if 'fc1' in activations:
                # Apply ReLU manually since we want post-ReLU values
                fc1_output = torch.nn.functional.relu(activations['fc1'])
                max_vals.append(torch.max(torch.abs(fc1_output)).item())
    
    hook.remove()
    
    # Use 99th percentile instead of max to avoid outliers
    max_val = np.percentile(max_vals, 99)
    scale = max_val / 127.0  # Map to [-128, 127] range
    return scale

def quantize_activation(x, scale, zero_point):
    """Quantize floating point values to 8-bit integers"""
    return np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)

def dequantize_activation(x_q, scale, zero_point):
    """Convert 8-bit integers back to floating point"""
    return (x_q.astype(np.float32) - zero_point) * scale

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        # Add quantization stub for intermediate activations
        self.activation_quant = QuantStub()  
        self.fc2 = nn.Linear(128, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu(x)
        # Quantize intermediate activations during training!
        x = self.activation_quant(x)  
        x = self.fc2(x)
        x = self.dequant(x)
        return x

class FakeQuantize(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        
    def forward(self, x):
        # Simulate our hardware quantization during training
        x_quantized = torch.clamp(torch.round(x / self.scale), -128, 127)
        # But keep in float for gradient flow
        return x_quantized * self.scale

class FCNetCustomQuant(nn.Module):
    def __init__(self, activation_scale=1.0):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fake_quant = FakeQuantize(activation_scale)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fake_quant(x)  # Simulate inter-layer quantization
        x = self.fc2(x)
        return x
    
def prepare_model(model, qconfig='qnnpack'):
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

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Train and quantize model
model = FCNet()
model = prepare_model(model)
train_model(model, train_loader)

fc1_activation_scale = calculate_activation_scale(model, train_loader, 'fc1')
# Convert to quantized model
model.eval()
model = convert(model)

# Save int8 weights
weights = {}
weights['fc1'] = model.fc1.weight().int_repr().numpy().astype(np.int8)  # 784x128
weights['fc2'] = model.fc2.weight().int_repr().numpy().astype(np.int8)  # 128x10

scales = {
    'fc1_activation_scale': fc1_activation_scale,
    # For output layer, we typically don't quantize since it's logits
    'fc2_output_scale': 1.0  # Keep as float for final prediction
}

# Save everything together
model_data = {
    'weights': weights,
    'scales': scales
}
torch.save(model_data, 'tpu/qat_model.pt')

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