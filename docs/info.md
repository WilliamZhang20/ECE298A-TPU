<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This project is a small-scale matrix multiplier inspired by the Tensor Processing Unit (TPU), an AI inference ASIC made by Google.

It multiplies 2x2 matrices, each of which contain signed, 1-byte (8-bit) elements. It does so in a systolic array circuit, where flow of data is facilitated through the connections between a grid of 4 Multiply-Add-Accumulate (MAC) Processing Elements (PEs).

To store inputs prior to computation, it contains 2 matrices in memory registers, which occupy a total of 8 bytes.

To orchestrate the flow of data between inputs, memory, and outputs, a control unit coordinates state transitions, loads, and stores automatically.

Finally, to schedule the inputs and outputs to and from the systolic array, a feeder module closer to the matrix multiplier works with the control unit.

### The Processing Element

|Signal Name        | Direction     | Blurb             |
|-------------------|---------------|-------------------|
|clk                | input         | The clock!        |
|rst                | input         | Reset             |
|clear              | input         | Clear PE          |
|a_in               | input         | First input       |
|a_out              | output        | Pass-on of input  |
|b_in               | input         | Weight value      |
|b_out              | output        | Pass-on of weight |
|c_out              | output        | Accumulation      |

Let's start from the most atomic element of the matrix multiplier unit (MMU): its processing element (PE). In this 2x2 multiplier, the result is a 4-element square matrix, so there are 4 PEs. The value stored within each PE contributes an element to the output. 

Since each output element of a matrix multiplication is a sum of products, the PE's primary operation is a multiply add accumulate.

It will taken in input terms `a_in` and `b_in`, multiply them, and then add them to an accumulator value `c_out`. Due to the larger values induced by multiplication, the accumulator holds more bits.

Since adjacent PEs corresponding to adjacent elements of the output matrix need the same input and weight values, these input terms are sent to `a_out` and `b_out` respectively, which are connected to other PEs by the systolic array.

Once the multiplication is done, the control unit will want to clear the PEs so that they can reset accumulation for the next matrix product, which is facilitated via the `clear` signal. 

On the other hand, it is non-ideal to reset the entire chip, as it wastes time (an entire clock cycle) and is overkill as it is unecessary to reset other elements.

### The Systolic Array

The systolic array is a network, or grid, of PEs.

Block Diagram...

### The Memory
<!--Specify input/output signals, internal functionality, etc.
--->

### The Control Unit
<!--Specify input/output signals, internal functionality, etc.
--->

### The Matrix Unit Feeder
<!--Specify input/output signals, internal functionality, etc.
--->

## How to test

Notation: the matrix element A_xy denotes a value in the xth row and yth column of the matrix A.

The module will assume an order of input of A matrix values and B matrix values, and outputs. That is, it is expected that inputs come in order of A00, A01, A10, A11, B00, B01, B10, B11, and the outputs will come in the order of C00, C01, C10, C11. This keeps it simple.

### Setup

1. Power Supply: Connect the chip to a stable power supply as per the voltage specifications.
2. Clock Signal: Provide a stable clock signal to the clk pin.
3. Reset: Ensure the rst_n pin is properly connected to allow resetting the chip.

### A Matrix Multiplication Round

1. Initial Reset
    - Perform a reset by pulling the `rst_n` pin low to 0, and waiting for a single clock signal before pulling it back high to 1. This sets initial state values.
2. Initial Matrix Load
    - Load 8 matrix elements into the chip, one per cycle. For example, if your matrices are [[1, 2], [3, 4]], [[5, 6], [7, 8]], you would load in the row-major, first-matrix-first order of 1, 2, 3, 4, 5, 6, 7, 8. This occurs by setting the 8 `ui_in` pins to the 8-bit value of the set matrix element, and waiting one clock cycle before the next can be loaded.
3. Collect Output
    - Thanks to the aggressive pipelining implemented in the chip, once the matrices are already loaded, you can start collecting output! For the above example, the output would be in the order of [19, 22, 43, 50], starting from the cycle right after you finish your last load.
4. Repeat
    - Load 8 more input values, collect 4 outputs, rinse & repeat!

## External hardware

An external microcontroller will send signals over the chip interface, including the clock signal, which will alow it to coordinate I/O on clock edges.
