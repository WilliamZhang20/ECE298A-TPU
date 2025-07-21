import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

@cocotb.test()
async def test_memory_write_and_read(dut):
    dut._log.info("Start memory test")

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.write_en.value = 0
    dut.addr.value = 0
    dut.in_data.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)

    # Define 8 test values
    test_data = list(range(8, 8 + 8))

    # Write each value to corresponding address
    for addr, val in enumerate(test_data):
        dut.write_en.value = 1
        dut.addr.value = addr
        dut.in_data.value = val
        await ClockCycles(dut.clk, 1)

    dut.write_en.value = 0
    await ClockCycles(dut.clk, 2)

    # Read and check outputs
    expected_weights = test_data[0:4]
    expected_inputs = test_data[4:8]

    weights = [int(getattr(dut, f"weight{i}").value) for i in range(4)]
    inputs = [int(getattr(dut, f"input{i}").value) for i in range(4)]

    for i in range(4):
        assert weights[i] == expected_weights[i], \
            f"weight{i} = {weights[i]}, expected {expected_weights[i]}"
        assert inputs[i] == expected_inputs[i], \
            f"input{i} = {inputs[i]}, expected {expected_inputs[i]}"

    dut._log.info("Memory test passed")
