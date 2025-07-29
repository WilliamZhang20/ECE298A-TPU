import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

async def reset_dut(dut):
    """Reset the DUT."""
     # Reset
    dut.input0.value = 0
    dut.input1.value = 0
    dut.input2.value = 0
    dut.input3.value = 0
    dut.weight0.value = 0
    dut.weight1.value = 0
    dut.weight2.value = 0
    dut.weight3.value = 0
    dut.c00.value = 0
    dut.c01.value = 0
    dut.c10.value = 0
    dut.c11.value = 0
    dut.rst.value = 0
    dut.en.value = 0
    dut.rst.value = 1
    await ClockCycles(dut.clk, 1)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

def unsigned_to_signed(value):
    """Convert unsigned value to signed."""
    return value if value < 128 else value - 256

# Regular input and weight values
@cocotb.test()
async def test1(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Reset DUT
    await reset_dut(dut)

    dut._log.info("Enabling feeder module...")
    dut.en.value = 1
    R = [] # List to store results

    # ------------------------------
    # CYCLE 0: Start loading matrices from internal memory
    I = [1, 2, 3, 4]  # row-major: [I00, I01, I10, I11]
    W = [5, 6, 7, 8]  # row-major: [W00, W01, W10, W11]

    dut.mmu_cycle.value = 0

    dut.input0.value = I[0]
    dut.weight0.value = W[0]
    
    await RisingEdge(dut.clk)

    # ------------------------------
    # CYCLE 1: First partial products complete
    dut.mmu_cycle.value = 1

    dut.input1.value = I[1]
    dut.input2.value = I[2]
    dut.weight1.value = W[1]
    dut.weight2.value = W[2]

    dut.c00.value = I[0] * W[0]  # Set c_0 value (data ready from mmu)

    await RisingEdge(dut.clk)

    # ------------------------------
    # CYCLE 2: c00 = a00xb00 ready, c00 outputted
    dut.mmu_cycle.value = 2

    dut._log.info(f"Cycle 2: c00 = {dut.c00.value.integer}")
    dut.input3.value = I[3]
    dut.weight3.value = W[3]

    dut.c01.value = I[1] * W[1]  # Set c_1 value (data ready from mmu)
    dut.c10.value = I[2] * W[2] # Set c_2 value (data ready from mmu)

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_0 to be stable)

    dut._log.info(f"POST Cycle 2: c00 outputted = {dut.host_outdata.value.integer}")
    R.insert(0,dut.host_outdata.value.integer)  # Store host_ output (c00 outputted)

    # ------------------------------
    # CYCLE 3: c01 = a00xb01 ready, c10 = a10xb00 ready, c01 outputted
    dut.mmu_cycle.value = 3

    dut._log.info(f"Cycle 3: c01 = {dut.c01.value.integer}")

    dut.c11.value = I[3] * W[3]  # Set c_3 value (data ready from mmu)

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_1 to be stable)

    dut._log.info(f"POST Cycle 3: c01 outputted = {dut.host_outdata.value.integer}")
    R.insert(1, dut.host_outdata.value.integer)  # Store host_ output (c01 outputted)
    
    # ------------------------------
    # CYCLE 4: c11 = a10xb01 ready, c10 outputted
    dut.mmu_cycle.value = 4

    dut._log.info(f"Cycle 4: c10 = {dut.c10.value.integer}")

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_2 to be stable)

    dut._log.info(f"POST Cycle 4: c10 outputted = {dut.host_outdata.value.integer}")
    R.insert(2, dut.host_outdata.value.integer)  # Store host_ output (c10 outputted)

    # ------------------------------
    # CYCLE 5: c11 outputted
    dut.mmu_cycle.value = 5

    dut._log.info(f"Cycle 5: c11 = {dut.c11.value.integer}")

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_3 to be stable)

    dut._log.info(f"POST Cycle 5: c11 outputted = {dut.host_outdata.value.integer}")
    R.insert(3, dut.host_outdata.value.integer)  # Store host_ output (c11 outputted)

    # ------------------------------
    # CHECK RESULTS
    expected = [5, 12, 21, 32]  # Expected results

    for i in range(4):
        assert R[i] == expected[i], f"R[{i}] = {R[i]} != expected {expected[i]}"
    
    dut._log.info("Test Passed: All results match expected values!")

# Test with edge case inputs
@cocotb.test()
async def test2(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Reset DUT
    await reset_dut(dut)

    dut._log.info("Enabling feeder module...")
    dut.en.value = 1
    R = [] # List to store results

    # ------------------------------
    # CYCLE 0: Start loading matrices from internal memory
    I = [127, -128, 127, -128]  # row-major: [I00, I01, I10, I11]
    W = [127, 127, -128, -128]  # row-major: [W00, W01, W10, W11]

    dut.mmu_cycle.value = 0

    dut.input0.value = I[0]
    dut.weight0.value = W[0]
    
    await RisingEdge(dut.clk)

    # ------------------------------
    # CYCLE 1: First partial products complete
    dut.mmu_cycle.value = 1

    dut.input1.value = I[1]
    dut.input2.value = I[2]
    dut.weight1.value = W[1]
    dut.weight2.value = W[2]

    dut.c00.value = I[0] * W[0]  # Set c_0 value (data ready from mmu)

    await RisingEdge(dut.clk)

    # ------------------------------
    # CYCLE 2: c00 = a00xb00 ready, c00 outputted
    dut.mmu_cycle.value = 2

    dut._log.info(f"Cycle 2: c00 = {dut.c00.value.integer}")
    dut.input3.value = I[3]
    dut.weight3.value = W[3]

    dut.c01.value = I[1] * W[1]  # Set c_1 value (data ready from mmu)
    dut.c10.value = I[2] * W[2] # Set c_2 value (data ready from mmu)

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_0 to be stable)

    val_unsigned = dut.host_outdata.value.integer
    val_signed = val_unsigned if val_unsigned < 128 else val_unsigned - 256
    dut._log.info(f"POST Cycle 2: c00 outputted = {val_signed}")
    R.insert(0,val_signed)  # Store host_ output (c00 outputted)

    # ------------------------------
    # CYCLE 3: c01 = a00xb01 ready, c10 = a10xb00 ready, c01 outputted
    dut.mmu_cycle.value = 3

    dut._log.info(f"Cycle 3: c01 = {dut.c01.value.integer}")

    dut.c11.value = I[3] * W[3]  # Set c_3 value (data ready from mmu)

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_1 to be stable)

    val_unsigned = dut.host_outdata.value.integer
    val_signed = val_unsigned if val_unsigned < 128 else val_unsigned - 256
    dut._log.info(f"POST Cycle 3: c01 outputted = {val_signed}")
    R.insert(1, val_signed)  # Store host_ output (c01 outputted)
    
    # ------------------------------
    # CYCLE 4: c11 = a10xb01 ready, c10 outputted
    dut.mmu_cycle.value = 4

    dut._log.info(f"Cycle 4: c10 = {dut.c10.value.integer}")

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_2 to be stable)

    val_unsigned = dut.host_outdata.value.integer
    val_signed = val_unsigned if val_unsigned < 128 else val_unsigned - 256
    dut._log.info(f"POST Cycle 4: c10 outputted = {val_signed}")
    R.insert(2, val_signed)  # Store host_ output (c10 outputted)

    # ------------------------------
    # CYCLE 5: c11 outputted
    dut.mmu_cycle.value = 5

    dut._log.info(f"Cycle 5: c11 = {dut.c11.value.integer}")

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_3 to be stable)

    val_unsigned = dut.host_outdata.value.integer
    val_signed = val_unsigned if val_unsigned < 128 else val_unsigned - 256
    dut._log.info(f"POST Cycle 5: c11 outputted = {val_signed}")
    R.insert(3, val_signed)  # Store host_ output (c11 outputted)

    # ------------------------------
    # CHECK RESULTS
    expected = [127, -128, -128, 127]  # Expected results

    for i in range(4):
        assert R[i] == expected[i], f"R[{i}] = {R[i]} != expected {expected[i]}"
    
    dut._log.info("Test Passed: All results match expected values!")

# Calculate results by reading from output ports to MMU (mixed signed inputs and weights)
@cocotb.test()
async def test3(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Reset DUT
    await reset_dut(dut)

    I = [1, -2, 3, -4]  # row-major: [I00, I01, I10, I11]
    W = [5, 6, -7, -8]  # row-major: [W00, W01, W10, W11]

    # Load initial values
    dut.input0.value = I[0]
    dut.weight0.value = W[0]
    dut.input1.value = I[1]
    dut.input2.value = I[2]
    dut.weight1.value = W[1]
    dut.weight2.value = W[2]
    dut.input3.value = I[3]
    dut.weight3.value = W[3]

    dut._log.info("Enabling feeder module...")
    dut.en.value = 1

    await RisingEdge(dut.clk)
    R = [] # List to store results

    # ------------------------------
    # CYCLE 0: Start loading matrices from internal memory
    dut.mmu_cycle.value = 0
    
    await RisingEdge(dut.clk)

    # ------------------------------
    # CYCLE 1: First partial products complete
    dut.mmu_cycle.value = 1

    await RisingEdge(dut.clk)

    # ------------------------------
    # CYCLE 2: c00 = a00xb00 ready, c00 outputted

    dut.c00.value = unsigned_to_signed(dut.a_data0.value.integer) * unsigned_to_signed(dut.b_data0.value.integer)
    dut.mmu_cycle.value = 2

    dut._log.info(f"Cycle 2: c00 = {dut.c00.value.integer}")

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_0 to be stable)

    val_signed = unsigned_to_signed(dut.host_outdata.value.integer)
    dut._log.info(f"POST Cycle 2: c00 outputted = {val_signed}")
    R.insert(0,val_signed)  # Store host_ output (c00 outputted)

    # ------------------------------
    # CYCLE 3: c01 = a00xb01 ready, c10 = a10xb00 ready, c01 outputted
    dut.c01.value = unsigned_to_signed(dut.a_data0.value.integer) * unsigned_to_signed(dut.b_data1.value.integer)  # c01 = a0xb1 ready
    dut.c10.value = unsigned_to_signed(dut.a_data1.value.integer) * unsigned_to_signed(dut.b_data0.value.integer) # c10 = a1xb0 ready
    dut.mmu_cycle.value = 3

    dut._log.info(f"Cycle 3: c01 = {dut.c01.value.integer}")

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_1 to be stable)

    val_signed = unsigned_to_signed(dut.host_outdata.value.integer)
    dut._log.info(f"POST Cycle 3: c01 outputted = {val_signed}")
    R.insert(1, val_signed)  # Store host_ output (c01 outputted)
    
    # ------------------------------
    # CYCLE 4: c11 = a10xb01 ready, c10 outputted
    dut.c11.value = unsigned_to_signed(dut.a_data1.value.integer) * unsigned_to_signed(dut.b_data1.value.integer)  # Set c11 = a1xb1 ready

    dut.mmu_cycle.value = 4

    dut._log.info(f"Cycle 4: c10 = {dut.c10.value.integer}")

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_2 to be stable)

    val_signed = unsigned_to_signed(dut.host_outdata.value.integer)
    dut._log.info(f"POST Cycle 4: c10 outputted = {val_signed}")
    R.insert(2, val_signed)  # Store host_ output (c10 outputted)

    # ------------------------------
    # CYCLE 5: c11 outputted
    dut.mmu_cycle.value = 5

    dut._log.info(f"Cycle 5: c11 = {dut.c11.value.integer}")

    await RisingEdge(dut.clk)
    # await RisingEdge(dut.clk) # for latching (wait for c_3 to be stable)

    val_signed = unsigned_to_signed(dut.host_outdata.value.integer)
    dut._log.info(f"POST Cycle 5: c11 outputted = {val_signed}")
    R.insert(3, val_signed)  # Store host_ output (c11 outputted)

    # ------------------------------
    # CHECK RESULTS
    expected = [5, -12, -21, 32]  # Expected results

    for i in range(4):
        assert R[i] == expected[i], f"R[{i}] = {R[i]} != expected {expected[i]}"
    
    dut._log.info("Test Passed: All results match expected values!")