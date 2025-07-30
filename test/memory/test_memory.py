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
async def test_edge_addresses(dut):
    """Write to lowest and highest valid addresses (0 and 7)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    await write_addr(dut, 0, 0xAA)
    await write_addr(dut, 7, 0x55)

    weights, inputs = snapshot_outputs(dut)
    assert weights[0] == 0xAA, f"weight0={weights[0]}, expected 0xAA"
    assert inputs[3] == 0x55, f"input3={inputs[3]}, expected 0x55"

    for i in range(1, 4):
        assert weights[i] == 0, f"weight{i}={weights[i]}, expected 0 after only addr0 write"
    for i in range(0, 3):
        assert inputs[i] == 0, f"input{i}={inputs[i]}, expected 0 before write"

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
async def test_back_to_back_writes(dut):
    """Back-to-back writes on consecutive cycles should all take effect."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for i, val in enumerate([1, 2, 3, 4]):
        await write_addr(dut, i, val)

    for i, val in enumerate([9, 8, 7, 6]):
        await write_addr(dut, 4 + i, val)

    weights, inputs = snapshot_outputs(dut)
    assert weights == [1, 2, 3, 4], f"weights={weights}, expected [1,2,3,4]"
    assert inputs == [9, 8, 7, 6], f"inputs={inputs}, expected [9,8,7,6]"

@cocotb.test()
async def test_min_max_values(dut):
    """Write 0x00 and 0xFF to every address and verify."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for addr in range(8):
        await write_addr(dut, addr, 0x00)
    weights, inputs = snapshot_outputs(dut)
    assert weights == [0, 0, 0, 0], f"weights after zeros: {weights}"
    assert inputs == [0, 0, 0, 0], f"inputs after zeros: {inputs}"

    for addr in range(8):
        await write_addr(dut, addr, 0xFF)
    weights, inputs = snapshot_outputs(dut)
    assert weights == [0xFF]*4, f"weights after 0xFF: {weights}"
    assert inputs == [0xFF]*4, f"inputs after 0xFF: {inputs}"

@cocotb.test()
async def test_alternating_patterns(dut):
    """Test 0xAA and 0x55 alternating writes across all addresses."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for addr in range(8):
        val = 0xAA if (addr % 2 == 0) else 0x55
        await write_addr(dut, addr, val)

    weights, inputs = snapshot_outputs(dut)
    exp_w = [0xAA, 0x55, 0xAA, 0x55]
    exp_i = [0xAA, 0x55, 0xAA, 0x55]
    assert weights == exp_w, f"weights={weights}, expected {exp_w}"
    assert inputs == exp_i, f"inputs={inputs}, expected {exp_i}"

@cocotb.test()
async def test_sign_bit_and_boundaries(dut):
    """Load boundary and sign-bit values to ensure 8-bit correctness (no sign extension)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    values = [0x00, 0x01, 0x80, 0xFE, 0xFF, 0x7F, 0x10, 0xF0]
    assert len(values) == 8
    for addr, val in enumerate(values):
        await write_addr(dut, addr, val)

    weights, inputs = snapshot_outputs(dut)
    assert weights == values[:4], f"weights={weights}, expected {values[:4]}"
    assert inputs == values[4:], f"inputs={inputs}, expected {values[4:]}"

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
async def test_hold_load_en_and_stream(dut):
    """Keep load_en high and change addr/data each cycle (burst write)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    patterns = [0x11, 0x22, 0x33, 0x44, 0x99, 0x88, 0x77, 0x66]
    dut.load_en.value = 1
    for addr, val in enumerate(patterns):
        dut.addr.value = addr
        dut.in_data.value = val
        await ClockCycles(dut.clk, 1)
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 1)

    weights, inputs = snapshot_outputs(dut)
    assert weights == patterns[:4], f"weights={weights}, expected {patterns[:4]}"
    assert inputs == patterns[4:], f"inputs={inputs}, expected {patterns[4:]}"

@cocotb.test()
async def test_state_stability_without_writes(dut):
    """After writing, hold signals steady without load_en and confirm state does not change."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for addr, val in enumerate([5, 6, 7, 8, 9, 10, 11, 12]):
        await write_addr(dut, addr, val)

    weights_before, inputs_before = snapshot_outputs(dut)
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 10)
    weights_after, inputs_after = snapshot_outputs(dut)

    assert weights_before == weights_after, f"weights changed without writes: {weights_before} -> {weights_after}"
    assert inputs_before == inputs_after, f"inputs changed without writes: {inputs_before} -> {inputs_after}"

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

@cocotb.test()
async def test_multiple_resets(dut):
    """Write data, reset, verify clear, write different data, verify."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    for addr, val in enumerate([1,2,3,4,5,6,7,8]):
        await write_addr(dut, addr, val)

    dut.rst.value = 1
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    weights, inputs = snapshot_outputs(dut)
    assert weights == [0,0,0,0], f"after reset1 weights={weights}"
    assert inputs == [0,0,0,0], f"after reset1 inputs={inputs}"

    for addr, val in enumerate([0x10,0x20,0x30,0x40,0xA0,0xB0,0xC0,0xD0]):
        await write_addr(dut, addr, val)

    weights, inputs = snapshot_outputs(dut)
    assert weights == [0x10,0x20,0x30,0x40], f"weights={weights}"
    assert inputs == [0xA0,0xB0,0xC0,0xD0], f"inputs={inputs}"