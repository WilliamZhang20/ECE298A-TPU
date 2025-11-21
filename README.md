![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# Tiny Tapeout Verilog Processing Unit

- [Read the documentation for project](docs/info.md)

## Overview

This project implements a small-scale, hardware-efficient Tensor Processing Unit (TPU) that performs 2×2 matrix multiplications using a systolic array of Multiply-Accumulate (MAC) units. It is designed in Verilog and deployable via the Tiny Tapeout ASIC flow.

Hardware design files are in the `./src` folder, while the ML inference setup is inside `./test/tpu`.

## Hardware Features

- **Systolic Array:** A 2×2 grid of MAC units propagates data left-to-right and top-to-bottom, emulating a systolic matrix multiplication engine.
- **Signed 8-bit Inputs, 16-bit Outputs:** Handles signed integers (-128 to 127) and accumulates products in 16-bit precision.
- **Streaming Input/Output:** Supports pipelined loading and output to achieve >99.8M operations/sec.
- **Control FSM:** Automates input loading, matrix multiplication timing, and result collection.
- **Optional Features:** On-chip fused matrix transpose (`Bᵀ`) and ReLU activation.

## Machine Learning Ecosystem

- **Accurate Low-Precision Training**: since the chip only runs on 8-bit arithmetic but training needs higher preciison I run Quantization-Aware Training, simulating the quantization noise from 32-bit float to 8-bit integer. This allows minimal accuracy loss during quantization.
  - The process is made easier with TorchAO inserting quantization stubs automatically, and ExecuTorch handling fast quantization and low-precision training.
- **Simplified Model Deployment**: rather than run inference by calling the chip's `matmul` kernel manually while training in a separate `torch.nn` module, a PyTorch compiler backend unifies the process in `torch.nn`. One can train the model using the module, call `torch.compile` with a custom backend, and run inference with `model(input)`. 

---

## System Architecture

Notice in the diagram that data flows from the input through the blue, yellow, red, purple, and green arrows to go from two input matrices to an output matrix.

![Block Diagram](docs/ECE298A-TPU.png)

**Subsystems:**
- **Processing Element (PE):** Executes MAC operations and propagates intermediate values.
- **Systolic Array (MMU):** A mesh of 4 PEs wired in systolic fashion.
- **Unified Memory:** 8-byte register bank storing both input and weight matrices.
- **Control Unit:** Finite-state machine (FSM) handles sequencing and pipelined computation.
- **MMU Feeder:** Schedules data flow between memory, computation, and output.

## Pin Configuration

### Dedicated Input Pins ui_in[7:0]

Is the primary port for inputting matrices used as operands in the product.

### Dedicated Output Pins uo_out[7:0]

8-bit half-sections of signed 16-bit elements of the 2x2 output matrix.

### Bidirectional GPIO Pins

- Index 0 - enabling the loading of elements at dedicated inputs into memory for computation
- Index 1 - enabling fused transpose of second operand in the product.
- Index 2 - enabling the application of Rectified Linear Unit (ReLU) activation at the output

### Control Signals
- Reset rst_n: Active low reset
- Clock clk: System timing (50 MHz)

## Operation
1. Initial Load: loading two 2x2 matrices (8 cycles)
2. Continuous streaming: while taking output of the matrices input in 1, the chip allows overlapped input of the next matrices

## How to test

See the test directory README for detailed instructions on running pre-silicon tests.

### Available test targets

```bash
cd test

# Run individual module tests
make test-top     # complete system
make test-memory  # memory unit
make test-systolic-array  # systolic array integrity
make test-control-unit  # control unit
make test-mmu-feeder  # matrix unit feeder
make test-nn # test neural network inference through the chip!!!

# Run all tests
make
```
