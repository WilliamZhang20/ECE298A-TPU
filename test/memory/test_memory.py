import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
import random

async def reset_dut(dut, ncycles: int = 2):
    """Apply reset and idle default values."""
    dut._log.info("Applying reset")
    dut.rst.value = 1
    dut.load_en.value = 0
    dut.addr.value = 0
    dut.in_data.value = 0
    await ClockCycles(dut.clk, ncycles)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

async def write_addr(dut, addr: int, data: int):
    """Single-cycle write helper (write occurs on the rising edge)."""
    dut.load_en.value = 1
    dut.addr.value = addr & 0x7
    dut.in_data.value = data & 0xFF
    await ClockCycles(dut.clk, 1)
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 1)

def snapshot_outputs(dut):
    weights = [int(getattr(dut, f"weight{i}").value) for i in range(4)]
    inputs = [int(getattr(dut, f"input{i}").value) for i in range(4)]
    return weights, inputs

@cocotb.test()
async def test_memory_reset(dut):
    """After reset, all outputs should be 0."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    weights, inputs = snapshot_outputs(dut)
    for i, w in enumerate(weights):
        assert w == 0, f"After reset: weight{i}={w}, expected 0"
    for i, x in enumerate(inputs):
        assert x == 0, f"After reset: input{i}={x}, expected 0"

@cocotb.test()
async def test_sequential_write_and_read(dut):
    """Write all 8 addresses (0..7) with distinct values and check mapped outputs."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    test_data = list(range(8, 16))
    for addr, val in enumerate(test_data):
        await write_addr(dut, addr, val)

    weights, inputs = snapshot_outputs(dut)
    exp_weights = test_data[0:4]
    exp_inputs = test_data[4:8]

    for i in range(4):
        assert weights[i] == exp_weights[i], f"weight{i}={weights[i]}, expected {exp_weights[i]}"
        assert inputs[i] == exp_inputs[i], f"input{i}={inputs[i]}, expected {exp_inputs[i]}"

@cocotb.test()
async def test_load_enable_gating(dut):
    """Writes should only occur when load_en == 1."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    dut.load_en.value = 0
    dut.addr.value = 2
    dut.in_data.value = 0xDE
    await ClockCycles(dut.clk, 1)

    weights, inputs = snapshot_outputs(dut)

    assert weights == [0, 0, 0, 0], f"Unexpected weights after disabled write: {weights}"
    assert inputs == [0, 0, 0, 0], f"Unexpected inputs after disabled write: {inputs}"

    await write_addr(dut, 2, 0xBE)
    weights, _ = snapshot_outputs(dut)
    assert weights[2] == 0xBE, f"weight2={weights[2]}, expected 0xBE"

@cocotb.test()
async def test_overwrite_same_address(dut):
    """Second write to same address should overwrite previous value."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    await write_addr(dut, 5, 0x11)
    await write_addr(dut, 5, 0xA5)

    _, inputs = snapshot_outputs(dut)
    assert inputs[1] == 0xA5, f"input1={inputs[1]}, expected overwrite to 0xA5"

@cocotb.test()
async def test_write_during_reset_ignored(dut):
    """If reset is asserted on a write edge, the write should not persist after reset (assuming sync reset)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.load_en.value = 1
    dut.addr.value = 0
    dut.in_data.value = 0x77
    await ClockCycles(dut.clk, 1)

    dut.rst.value = 0
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 1)

    weights, inputs = snapshot_outputs(dut)
    assert weights == [0, 0, 0, 0], f"weights after write-during-reset: {weights}"
    assert inputs == [0, 0, 0, 0], f"inputs after write-during-reset: {inputs}"

@cocotb.test()
async def test_randomized_burst(dut):
    """Randomized addresses and data; verify final state matches last write per address."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    random.seed(1234)
    last = {addr: 0 for addr in range(8)}

    for _ in range(64):
        addr = random.randrange(0, 8)
        val = random.randrange(0, 256)
        await write_addr(dut, addr, val)
        last[addr] = val

    weights, inputs = snapshot_outputs(dut)
    exp_w = [last[a] for a in range(0, 4)]
    exp_i = [last[a] for a in range(4, 8)]
    assert weights == exp_w, f"randomized: weights={weights}, expected {exp_w}"
    assert inputs == exp_i, f"randomized: inputs={inputs}, expected {exp_i}"