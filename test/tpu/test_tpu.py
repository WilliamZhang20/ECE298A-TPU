import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import numpy as np

def permute_matrices_order(A, B):
    """
    Reorders A and B matrices for loading based on desired interleaved pattern:
    [a00, b00, a01, a10, b01, b10, a11, b11]
    
    A and B are 4-element lists in row-major order.
    """
    assert len(A) == 4 and len(B) == 4, "Both matrices must be 2x2 in row-major order"

    # Indices: A[0]=a00, A[1]=a01, A[2]=a10, A[3]=a11
    #          B[0]=b00, B[1]=b01, B[2]=b10, B[3]=b11
    return [
        A[0], B[0],  # a00, b00
        A[1], A[2],  # a01, a10
        B[1], B[2],  # b01, b10
        B[3], A[3]   # b11, a11
    ]

def saturate_to_s8(x):
    """Clamp value to 8-bit signed range [-128, 127]."""
    return max(-128, min(127, int(x)))

def get_expected_matmul(A, B, transpose=False, relu=False):
    A_mat = np.array(A).reshape(2, 2)
    B_mat = np.array(B).reshape(2, 2)
    if transpose:
        B_mat = B_mat.T
    result = A_mat @ B_mat
    if relu:
        result = np.maximum(result, 0)
    return [saturate_to_s8(val) for val in result.flatten().tolist()]

async def load_matrices(dut, A, B, transpose=0, relu=0):
    """
    Load two 2x2 matrices into the DUT using the custom 8-cycle pattern.

    Arguments:
        dut: cocotb DUT handle
        A: list of 4 values (row-major)
        B: list of 4 values (row-major)
    """
    combined = permute_matrices_order(A, B)

    for i, val in enumerate(combined):
        # Load enable = 1, sel = 0 for A or 1 for B depending on index
        sel = 0 if i in [0, 2, 3, 6] else 1  # index positions for A values
        dut.ui_in.value = val
        dut.uio_in.value = (transpose << 1) | (relu << 2) | 1
        await RisingEdge(dut.clk)

async def read_signed_output(dut, transpose=0, relu=0):
    results = []
    for i in range(4):
        dut.uio_in.value = (transpose << 1) | (relu << 2)
        await ClockCycles(dut.clk, 1)
        val_unsigned = dut.uo_out.value.integer
        val_signed = val_unsigned if val_unsigned < 128 else val_unsigned - 256
        results.append(val_signed)
        dut._log.info(f"Read C[{i//2}][{i%2}] = {val_signed}")
    return results

@cocotb.test()
async def test_relu_transpose(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 2, units="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)

    A = [5, -6, 7, 8]  # row-major
    B = [8, 9, 6, 8]  # row-major: [B00, B01, B10, B11]

    await load_matrices(dut, A, B, transpose=0, relu=1)

    expected = get_expected_matmul(A, B, transpose=False, relu=True)
    results = await read_signed_output(dut, transpose=0, relu=1)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("First part passed")

    A = [1, 2, 3, 4]
    B = [5, 6, 7, 8]

    await load_matrices(dut, A, B, transpose=1, relu=1)

    expected = get_expected_matmul(A, B, transpose=True, relu=True)
    results = await read_signed_output(dut, transpose=1, relu=1)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("ReLU + Transpose test passed!")

@cocotb.test()
async def test_numeric_limits(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 2, units="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)

    A = [5, -6, 7, 8]  # row-major
    B = [8, 12, 9, -7]  # row-major: [B00, B01, B10, B11]

    await load_matrices(dut, A, B)

    expected = get_expected_matmul(A, B)
    results = []

    # Wait for systolic array to compute
    
    results = await read_signed_output(dut)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Passed large positive values")

    A = [5, -6, 7, 8]  # row-major
    B = [8, -12, 9, -7]  # row-major: [B00, B01, B10, B11]
    
    await load_matrices(dut, A, B)

    expected = get_expected_matmul(A, B)
    results = []

    # Wait for systolic array to compute
    
    results = await read_signed_output(dut)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test passed!")

@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 2, units="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)

    # ------------------------------
    # STEP 1: Load matrix A
    # A = [[1, 2],
    #      [3, 4]]
    A = [1, 2, 3, 4]  # row-major

    # ------------------------------
    # STEP 2: Load matrix B
    # B = [[5, 6],
    #      [7, 8]]
    B = [5, 6, 7, 8]  # row-major: [B00, B01, B10, B11]
    
    await load_matrices(dut, A, B)

    # ------------------------------
    # STEP 4: Read outputs
    expected = get_expected_matmul(A, B)
    results = []

    results = await read_signed_output(dut)

    # ------------------------------
    # STEP 5: Check results
    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test 1 passed!")

    #######################################
    ##### TEST RUN 2 - CHECK CLEARING #####
    
    # ------------------------------
    # STEP 1: Load matrices

    A = [79, -10, 7, 8]  # row-major
    B = [2, 6, 5, 8]  # row-major: [B00, B01, B10, B11]

    await load_matrices(dut, A, B)

    # ------------------------------
    # STEP 4: Read outputs
    expected = get_expected_matmul(A, B)
    results = []

    results = await read_signed_output(dut)

    # ------------------------------
    # STEP 5: Check results
    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test 2 passed!")

    #########################################
    ##### TEST RUN 3 - CHECK SIGNED OPS #####

    A = [5, -6, 7, 8]  # row-major
    B = [1, 2, 3, -4]  # row-major: [B00, B01, B10, B11]

    await load_matrices(dut, A, B)

    expected = get_expected_matmul(A, B)
    results = []

    # Wait for systolic array to compute
    
    results = await read_signed_output(dut)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test 3 passed!")