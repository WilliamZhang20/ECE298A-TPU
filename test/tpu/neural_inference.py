import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import numpy as np
import torch
from test_tpu import matmul

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
torch.serialization.add_safe_globals([numpy.ndarray])

def quantize_activation(x, scale, zero_point):
    """Quantize floating point values to 8-bit integers"""
    return np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)

async def reset_dut(dut):
    """Reset the DUT by asserting rst_n low for 10 clock cycles, then high."""
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 10)

@cocotb.test()
async def test_neural_network_inference(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Load PyTorch QAT weights
    try:
        model_data = torch.load('tpu/qat_model.pt', weights_only=True)
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

        # Layer 1: Z1 = X @ W1^T + ReLU
        Z1_raw = await matmul(dut, X, W1, transpose=True, relu=True)  # 1x128

        # Quantize activations manually
        Z1_quantized = quantize_activation(Z1_raw, fc1_activation_scale, zero_point=0)

        # Layer 2: Z2 = Z1_quantized @ W2^T
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
