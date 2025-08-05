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

### Systolic Array Tests

**File:** `test_systolic_array.py`

### Top-level Tests

**File:** `test_tpu.py`

## Viewing waveforms

Each module's test directory has a `wave` directory, which contains `.vcd` waveforms generated after each test. These can be viewed with either `surfer` or `GTKWave`:

```sh
# GTKWave
gtkwave tb.vcd

# Surfer
surfer tb.vcd
```
