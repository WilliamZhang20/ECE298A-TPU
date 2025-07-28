import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import numpy as np

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

async def load_matrix(dut, matrix, transpose=0, relu=0):
    """
    Load a 2x2 matrix into the DUT.
    
    Args:
        dut: Device Under Test
        matrix: list of 4 values (row-major)
        sel: 0 for matrix A, 1 for matrix B
    """
    for i in range(4):
        dut.ui_in.value = matrix[i]
        dut.uio_in.value = (transpose << 1) | (relu << 2) | 1  # load_en=1, load_sel_ab=sel, load_index
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

def get_expected_large_matmul(A, B):
    A_saturated = np.vectorize(saturate_to_s8)(A)
    B_saturated = np.vectorize(saturate_to_s8)(B)

    m, n = A.shape
    n_b, p = B.shape
    assert n == n_b, "Incompatible dimensions"

    # Pad dimensions to multiple of 2
    m_padded = ((m + 1) // 2) * 2
    n_padded = ((n + 1) // 2) * 2
    p_padded = ((p + 1) // 2) * 2

    A_pad = np.zeros((m_padded, n_padded), dtype=int)
    B_pad = np.zeros((n_padded, p_padded), dtype=int)
    A_pad[:m, :n] = A_saturated
    B_pad[:n, :p] = B_saturated

    # Initialize output accumulator with 32-bit int to avoid overflow
    result = np.zeros((m_padded, p_padded), dtype=int)

    for i in range(0, m_padded, 2):
        for j in range(0, p_padded, 2):
            # Clear PE accumulators for this 2x2 block:
            # We emulate the reset in hardware by zeroing local accumulators.
            acc_block = np.zeros((2, 2), dtype=int)

            for k in range(0, n_padded, 2):
                # Extract 2x2 sub-blocks
                A_block = A_pad[i:i+2, k:k+2]
                B_block = B_pad[k:k+2, j:j+2]

                # Multiply elementwise (each element 8-bit saturated)
                # Resulting products are 16-bit signed integers
                products = np.zeros((2, 2, 2, 2), dtype=int)  # (A-row, A-col, B-row, B-col)
                for a_r in range(2):
                    for a_c in range(2):
                        for b_r in range(2):
                            for b_c in range(2):
                                if (i + a_r) < m and (k + a_c) < n and (k + b_r) < n and (j + b_c) < p:
                                    # Only valid indices contribute
                                    prod = A_block[a_r, a_c] * B_block[b_r, b_c]
                                    products[a_r, a_c, b_r, b_c] = prod
                                else:
                                    products[a_r, a_c, b_r, b_c] = 0

                # Now sum over the products for matrix multiplication partial sums:
                # The 2x2 block output c_ij = sum over k of a_ik * b_kj
                # For each output element in acc_block:
                for r in range(2):
                    for c in range(2):
                        # sum over a_c (which indexes over k dimension) matching b_r index
                        partial_sum = 0
                        for inner in range(2):  # inner loop over 2 elements in block dimension
                            partial_sum += products[r, inner, inner, c]

                        # Accumulate partial sums with 12-bit wrap (simulate PE accumulator)
                        acc_block[r, c] = (acc_block[r, c] + saturate_to_s8(partial_sum))

            # After summing all k-blocks, saturate final 12-bit values to signed 8-bit and store
            for r in range(2):
                for c in range(2):
                    if (i + r) < m and (j + c) < p:
                        result[i + r, j + c] = acc_block[r, c]

    # Return only the original shape
    return result[:m, :p]

def check_expected(A, B, result):
    """
    Check DUT results against expected matrix multiplication, for big matrices
    """
    print(A @ B)
    print(result)
    expected = get_expected_large_matmul(A, B)
    print(expected)
    np.testing.assert_array_equal(result, expected, err_msg="Matrix multiplication result does not match expected")

async def read_matrix_output(dut, results_large, i, j, transpose=0, relu=0):
    for k in range(4):
        dut.uio_in.value = (transpose << 1) | (relu << 2)
        await ClockCycles(dut.clk, 1)
        val_unsigned = dut.uo_out.value.integer
        val_signed = val_unsigned if val_unsigned < 128 else val_unsigned - 256
        row = i + (k // 2)
        col = j + (k % 2)
        results_large[row, col] += val_signed

async def matmul(dut, A, B):
    """
    Perform matrix multiplication on DUT for matrices of arbitrary dimensions.
    """
    m, n = A.shape
    n_b, p = B.shape
    assert n == n_b, "Matrix dimensions must be compatible for multiplication"
    
    # Compute padded dimensions (next multiple of 2)
    m_padded = ((m + 1) // 2) * 2
    n_padded = ((n + 1) // 2) * 2
    p_padded = ((p + 1) // 2) * 2
    
    # Pad matrices with zeros
    A_padded = np.zeros((m_padded, n_padded), dtype=int)
    B_padded = np.zeros((n_padded, p_padded), dtype=int)
    A_padded[:m, :n] = A
    B_padded[:n, :p] = B
    
    # Initialize result matrix
    results_large = np.zeros((m_padded, p_padded), dtype=int)
    
    # Process in 2x2 blocks
    for i in range(0, m_padded, 2):
        for j in range(0, p_padded, 2):
            for k in range(0, n_padded, 2):
                # Extract 2x2 blocks
                A_block = A_padded[i:i+2, k:k+2].flatten().tolist()
                B_block = B_padded[k:k+2, j:j+2].flatten().tolist()
                
                # Load blocks into DUT
                await load_matrix(dut, A_block)
                await load_matrix(dut, B_block)
                
                # Read partial result directly into results_large
                await read_matrix_output(dut, results_large, i, j)
    
    # Return valid result (m x p)
    return results_large[:m, :p]

@cocotb.test()
async def test_relu_transpose(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 1, units="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    A = [5, -6, 7, 8]  # row-major
    B = [8, 9, 6, 8]  # row-major: [B00, B01, B10, B11]

    await load_matrix(dut, A, transpose=0, relu=1)
    await load_matrix(dut, B, transpose=0, relu=1)

    expected = get_expected_matmul(A, B, transpose=False, relu=True)
    results = await read_signed_output(dut, transpose=0, relu=1)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("First part passed")

    A = [1, 2, 3, 4]
    B = [5, 6, 7, 8]

    await load_matrix(dut, A, transpose=1, relu=1)
    await load_matrix(dut, B, transpose=1, relu=1)

    expected = get_expected_matmul(A, B, transpose=True, relu=True)
    results = await read_signed_output(dut, transpose=1, relu=1)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("ReLU + Transpose test passed!")

@cocotb.test()
async def test_numeric_limits(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 1, units="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    A = [5, -6, 7, 8]  # row-major
    B = [8, 12, 9, -7]  # row-major: [B00, B01, B10, B11]

    await load_matrix(dut, A)
    await load_matrix(dut, B)

    expected = get_expected_matmul(A, B)
    results = []

    # Wait for systolic array to compute
    
    results = await read_signed_output(dut)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Passed large positive values")

    A = [5, -6, 7, 8]  # row-major
    B = [8, -12, 9, -7]  # row-major: [B00, B01, B10, B11]

    await load_matrix(dut, A)
    await load_matrix(dut, B)

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
    clock = Clock(dut.clk, 1, units="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

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
    
    await load_matrix(dut, A)
    await load_matrix(dut, B)

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

    await load_matrix(dut, A)
    await load_matrix(dut, B)

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

    await load_matrix(dut, A)
    await load_matrix(dut, B)

    expected = get_expected_matmul(A, B)
    results = []

    # Wait for systolic array to compute
    
    results = await read_signed_output(dut)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test 3 passed!")

    # ------------------------------
    # TEST RUN 4: Large Matrix Multiplication with Arbitrary Dimensions
    # User-specified size, all elements MUST FIT WITHIN INT8 RANGE
    np.random.seed(42)
    A_large = np.random.randint(-20, 20, size=(5, 3))
    B_large = np.random.randint(-20, 20, size=(3, 6))

    # Perform matrix multiplication on DUT
    result = await matmul(dut, A_large, B_large)
    
    # Check results against expected
    check_expected(A_large, B_large, result)

    dut._log.info("Test 4 (Arbitrary Dimension Matrix) passed!")
   