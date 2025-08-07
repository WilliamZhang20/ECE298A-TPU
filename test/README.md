# TPU Test Suite

This directory encompasses the test suite for our 2x2 TPU.

## Structure

The test suite contains both per-module and system integration tests within their own directories:

* `control_unit/`: Control unit tests
* `memory/`: Unified memory tests
* `mmu_feeder/`: Matrix Unit Feeder tests
* `systolic_array/`: Systolic Array tests
* `tpu/`: Top-level system integration tests

## Setting up

### Local environment

If you are running these tests locally, you will need:

 * Python
 * Icarus Verilog

Both can be installed with your package manager of choice.

From the `test` directory, create a Python virtual environment:

```python
python -m venv <ENV_NAME>

# Linux/macOS
source <ENV_NAME>/bin/activate

# Windows
<ENV_NAME>\Scripts\activate
```

Regardless of whether you require a virtual environment or are running the tests on a pre-configured server, there are some necessary packages that must be installed:

```python
pip install -r requirements.txt
```

## How to run

The `Makefile` in the test directory has been set up with targets for each module and the top-level:

```c
// Per-module tests
make test-top // System integration
make test-memory
make test-systolic-array
make test-control-unit
make test-mmu-feeder

// Run all tests 
make all

// Cleanup build artifacts
make clean
```

To run gate-level simulations, specify the `GATES` flag for `test-top`:

```c
make test-top GATES=yes
```

## Test Information

### Control Unit Tests

**File:** `test_control_unit.py`

**Test Cases**:

* **Reset**: verifies all control signals (mem_addr, mmu_en, mmu_cycle) are properly zeroed when reset is asserted
* **Idle State**: ensures control unit remains in idle state with all outputs at zero when load_en is not asserted for multiple cycles
* **Matrix Loading**: tests 8-cycle matrix loading phase with proper memory address incrementing (0-7) and MMU enable assertion when sufficient elements are loaded
* **MMU Compute Phase**: validates MMU enable signals remain high and mmu_cycle counter increments correctly (2-7) during computation and writeback
* **Full Cycle**: tests complete operation sequence from idle → matrix loading → MMU computation → back to compute state with proper signal transitions
* **Load Enable Handling**: verifies load_en signal toggling is properly ignored during MMU compute phase while mmu_cycle continues incrementing normally
* **Multiple Operations**: stress tests control unit with 100 consecutive operation cycles to ensure state machine reliability and proper signal timing

### Unified Memory Tests

**File:** `test_memory.py`

**Test Cases**:

* **Reset**: check if all values are zeroed when `rst` is asserted.
* **Sequential Read/Write**: write to all `sram` addresses and check the outputs at each address.
* **Load Enable Writes**: verify that writes only occur with `load_en` high.
* **Overwrite Addresses**: write two values to an address; second value should overwrite the first one.
* **Write During Reset**: ensure the validity of a reset if `rst` and `load_en` are asserted in the same cycle.
* **Randomized Burst**: randomized addresses and data are generated within a valid range, written, and read for verification.

### Matrix Unit Feeder Tests

**File:** `test_mmu_feeder.py`

**Test Cases**:

* **Regular, Edge Case, and Transpose**: Test mmu_feeder with regular matrix inputs (minimal functionality), edge case matrix input (check expected behavior with extreme inputs), and regular matrix inputs with tranpose enabled (check transpose functionality).
* **Random Test Vectors**: Generates 10 random Input/Weight matrix pairs and processes them sequentially, checking output against expected values.

**Note**: Expected output control signal values from mmu_feeder (i.e clear and done) are asserted at each mmu_cycle, covering all mmu_feeder functionality.

### Systolic Array Tests

**File:** `test_systolic_array.py`

**Test Cases**:

* **Basic 2x2 Matrix Multiplication**: validates core systolic array functionality with predefined 2x2 matrices, testing proper data flow through processing elements and correct computation of matrix multiplication (A=[[1,2],[3,4]], B=[[5,6],[7,8]], expected C=[[19,22],[43,50]])

### Top-level Tests

**File:** `test_tpu.py`

**Test Cases**:

* **ReLU and Transpose Operations**: tests ReLU activation function and matrix transpose functionality with pipelined loading and reading, validating correct handling of negative values and matrix orientation changes
* **Large Matrix Operations**: stress tests with randomly generated matrices up to 20x20, measuring operations per second and testing all combinations of transpose/ReLU flags to ensure scalability and performance
* **Project Integration Test**: comprehensive end-to-end test simulating full TPU operation with sequential matrix operations
  - Tests signed arithmetic edge cases with values ranging from -128 to 127
  - Validates pipeline clearing between consecutive matrix operations
  - Verifies proper output sequencing across multiple matrix computations
  - Tests extreme value combinations (-128 × 127, -128 × -128)
  - Ensures correct handling of result overflow and saturation

* **Verification Correctness**: output results were validated using commerically and popularly proven tools like NumPy, allowing us to verify that the logic of the circuit is correct. Similarly, when block multiplication was performed in bigger matrices, in which special algorithms were applied to break down the matrices, the NumPy validation was applied outside the hardware's required matrix breakdown, which validates the entire process at both ends.

## Viewing waveforms

Each module's test directory has a `wave` directory, which contains `.vcd` waveforms generated after each test. These can be viewed with either `surfer` or `GTKWave`:

```sh
# GTKWave
gtkwave tb.vcd

# Surfer
surfer tb.vcd
```

## Post-Silicon Tests

A future Python script that interfaces with the chip via the Caravel SoC can feed data and read outputs.
