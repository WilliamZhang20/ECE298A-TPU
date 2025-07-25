import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles

@cocotb.test()
async def test_systolic_array_basic(dut):
    """Test basic 2x2 matrix multiplication"""

    cocotb.log.info("Starting systolic array test")

    clock = Clock(dut.clk, 10, units="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.clear.value = 1
    dut.activation.value = 0
    dut.a_data0.value = 0
    dut.a_data1.value = 0
    dut.b_data0.value = 0
    dut.b_data1.value = 0
    await Timer(20, units="ns")

    dut.rst.value = 0
    dut.clear.value = 1
    await RisingEdge(dut.clk)

    dut.clear.value = 0
    await RisingEdge(dut.clk)

    matrix_A = [[1, 2], [3, 4]]
    matrix_B = [[5, 6], [7, 8]]

    weights = [matrix_A[0][0], matrix_A[0][1], matrix_A[1][0], matrix_A[1][1]]  # weight0..3
    inputs = [matrix_B[0][0], matrix_B[0][1], matrix_B[1][0], matrix_B[1][1]]   # input0..3

    # Drive cycle 0: mmu_cycle = 3'b000
    dut.a_data0.value = weights[0]  # weight0
    dut.a_data1.value = 0
    dut.b_data0.value = inputs[0]   # input0
    dut.b_data1.value = 0
    await RisingEdge(dut.clk)

    # Drive cycle 1: mmu_cycle = 3'b001
    dut.a_data0.value = weights[1]  # weight1
    dut.a_data1.value = weights[2]  # weight2
    dut.b_data0.value = inputs[2]   # input2 (not transposed)
    dut.b_data1.value = inputs[1]   # input1
    await RisingEdge(dut.clk)

    # Drive cycle 2: mmu_cycle = 3'b010
    dut.a_data0.value = 0
    dut.a_data1.value = weights[3]  # weight3
    dut.b_data0.value = 0
    dut.b_data1.value = inputs[3]   # input3
    await RisingEdge(dut.clk)

    # Clear inputs for subsequent cycles (3'b011..3'b101)
    dut.a_data0.value = 0
    dut.a_data1.value = 0
    dut.b_data0.value = 0
    dut.b_data1.value = 0

    # Wait for 2 more cycles to let systolic array process
    await ClockCycles(dut.clk, 2)

    # Read outputs
    c00 = dut.c00.value.signed_integer
    c01 = dut.c01.value.signed_integer
    c10 = dut.c10.value.signed_integer
    c11 = dut.c11.value.signed_integer

    cocotb.log.info(f"Output matrix C:")
    cocotb.log.info(f"C00 = {c00}")
    cocotb.log.info(f"C01 = {c01}")
    cocotb.log.info(f"C10 = {c10}")
    cocotb.log.info(f"C11 = {c11}")

    # Check results
    assert c00 == 19, f"C00 expected 19 but got {c00}"
    assert c01 == 22, f"C01 expected 22 but got {c01}"
    assert c10 == 43, f"C10 expected 43 but got {c10}"
    assert c11 == 50, f"C11 expected 50 but got {c11}"

    cocotb.log.info("Systolic array multiplication test passed")
