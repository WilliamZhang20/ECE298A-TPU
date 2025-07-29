import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import numpy as np
import time
import statistics

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
    return result.flatten().tolist()

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

async def parallel_load_read(dut, A, B, transpose=0, relu=0):
    results = []
    dut.uio_in.value = (transpose << 1) | (relu << 2) | 1  # load_en=1

    for inputs in [A, B]:
        for i in range(2):
            idx0 = i * 2
            idx1 = i * 2 + 1
            # Feed either real data or dummy zeros
            dut.ui_in.value = inputs[idx0] if inputs else 0
            await ClockCycles(dut.clk, 1)
            high = dut.uo_out.value.integer

            dut.ui_in.value = inputs[idx1] if inputs else 0
            await ClockCycles(dut.clk, 1)
            low = dut.uo_out.value.integer

            combined = (high << 8) | low
            if combined >= 0x8000:
                combined -= 0x10000

            results.append(combined)
            dut._log.info(f"Read value = {combined}")
    return results

def get_expected_large_matmul(A, B, transpose=False, relu=False):
    # First saturate to emulate hardware's capacity & guard against bad test cases
    A_saturated = np.vectorize(saturate_to_s8)(A)
    B_saturated = np.vectorize(saturate_to_s8)(B)

    if transpose:
        B_saturated = B_saturated.T
    
    result = A_saturated @ B_saturated

    if relu:
        result = np.maximum(result, 0)

    return result

def check_expected(A, B, result):
    """
    Check DUT results against expected matrix multiplication, for big matrices
    """
    print(result)
    expected = get_expected_large_matmul(A, B)
    print(expected)
    np.testing.assert_array_equal(result, expected, err_msg="Matrix multiplication result does not match expected")

async def accumulate_matrix_output(dut, results_large, i, j, transpose=0, relu=0, A_block=None, B_block=None):
    """
    Interleaved output read and input feed for a 2x2 result tile.
    Accumulates results at results_large[i:i+2, j:j+2] by adding partial contributions.
    Loads A_block and B_block in 8 cycles while reading 4 outputs in 8 cycles.
    """
    dut.uio_in.value = (transpose << 1) | (relu << 2) | 1  # load_en=1

    for idx in range(2):
        # Load A and B elements for row idx (2 elements each)
        for k in range(2):
            # Load A[idx*2 + k]
            dut.ui_in.value = A_block[idx * 2 + k] if A_block else 0
            await ClockCycles(dut.clk, 1)
            # Read high byte of output
            high = dut.uo_out.value.integer

            # Load B[idx*2 + k]
            dut.ui_in.value = B_block[idx * 2 + k] if B_block else 0
            await ClockCycles(dut.clk, 1)
            # Read low byte of output
            low = dut.uo_out.value.integer

            # Combine high and low bytes
            combined = (high << 8) | low
            if combined >= 0x8000:
                combined -= 0x10000

            # Accumulate result in correct position (row-major order)
            row = i + idx
            col = j + k
            results_large[row, col] += combined  # Accumulate partial result

    print(results_large)

async def accumulate_matrix_output(dut, results_large, i, j, transpose=0, relu=0, A_block=None, B_block=None):
    """
    Interleaved output read and input feed for a 2x2 result tile.
    Accumulates results at results_large[i:i+2, j:j+2] by adding partial contributions.
    Loads A_block and B_block in 8 cycles while reading 4 outputs in 8 cycles.
    """
    dut.uio_in.value = (transpose << 1) | (relu << 2) | 1  # load_en=1

    # Store outputs for debugging
    outputs = []
    
    for idx in range(2):
        for k in range(2):
            # Load A[idx*2 + k]
            dut.ui_in.value = A_block[idx * 2 + k] if A_block else 0
            await ClockCycles(dut.clk, 1)
            high = dut.uo_out.value.integer

            # Load B[idx*2 + k]
            dut.ui_in.value = B_block[idx * 2 + k] if B_block else 0
            await ClockCycles(dut.clk, 1)
            low = dut.uo_out.value.integer

            # Combine high and low bytes
            combined = (high << 8) | low
            if combined >= 0x8000:
                combined -= 0x10000

            # Map output to result matrix (row-major order: C[i,j], C[i,j+1], C[i+1,j], C[i+1,j+1])
            row = i + (idx if k == 0 else idx + 1)
            col = j + (k if idx == 0 else k + 1)
            results_large[row, col] += combined
            outputs.append(combined)

    return outputs

async def matmul(dut, A, B, transpose=False, relu=False):
    """
    Fully pipelined systolic matrix multiplication using 2x2 blocks.
    Accumulates partial results across k dimension for each (i,j) tile.
    Loads two 2x2 matrices in 8 cycles, collects 4 outputs in 8 cycles while loading next matrices.
    """
    m, n = A.shape
    n_b, p = B.shape
    assert n == n_b, "Matrix dimension mismatch"

    # Pad matrices to be multiples of 2
    m_p, n_p = ((m + 1) // 2) * 2, ((n + 1) // 2) * 2
    p_p = ((p + 1) // 2) * 2

    A_padded = np.zeros((m_p, n_p), dtype=int)
    B_padded = np.zeros((n_p, p_p), dtype=int)
    A_padded[:m, :n] = A
    B_padded[:n, :p] = B
    results_large = np.zeros((m_p, p_p), dtype=int)

    # Generate tile coordinates, processing all j for each (i,k) pair
    tile_coords = [
        (i, j, k)
        for i in range(0, m_p, 2)
        for k in range(0, n_p, 2)
        for j in range(0, p_p, 2)
    ]

    # Step 1: Load the first tile pair
    i0, j0, k0 = tile_coords[0]
    A_block = A_padded[i0:i0+2, k0:k0+2].flatten().tolist()
    B_block = B_padded[k0:k0+2, j0:j0+2].flatten().tolist()
    dut._log.info(f"Loading first tile: i={i0}, j={j0}, k={k0}, A_block={A_block}, B_block={B_block}")
    await load_matrix(dut, A_block, transpose=0, relu=relu)
    await load_matrix(dut, B_block, transpose=transpose, relu=relu)

    # Step 2: Pipelined loop: read output of previous tile + load next tile
    for idx, (i, j, k) in enumerate(tile_coords[1:], 1):
        A_block_next = A_padded[i:i+2, k:k+2].flatten().tolist()
        B_block_next = B_padded[k:k+2, j:j+2].flatten().tolist()
        dut._log.info(f"Processing tile {idx}: i={i}, j={j}, k={k}, A_block={A_block_next}, B_block={B_block_next}")

        # Read output of previous tile and load next tile
        outputs = await accumulate_matrix_output(dut, results_large, i0, j0, transpose, relu, A_block_next, B_block_next)

        # Debug: Compute expected partial result for this tile
        A_tile = np.array(A_block).reshape(2, 2)
        B_tile = np.array(B_block).reshape(2, 2)
        expected_tile = np.dot(A_tile, B_tile).flatten()
        dut._log.info(f"Expected partial result for tile (i={i0}, j={j0}, k={k0}): {expected_tile.tolist()}")
        dut._log.info(f"Actual outputs: {outputs}")

        i0, j0, k0 = i, j, k  # Slide window
        A_block, B_block = A_block_next, B_block_next

    # Step 3: Read final output
    dut._log.info(f"Reading final tile: i={i0}, j={j0}, k={k0}")
    outputs = await accumulate_matrix_output(dut, results_large, i0, j0, transpose, relu)

    # Debug: Compute expected partial result for final tile
    A_tile = np.array(A_block).reshape(2, 2)
    B_tile = np.array(B_block).reshape(2, 2)
    expected_tile = np.dot(A_tile, B_tile).flatten()
    dut._log.info(f"Expected partial result for tile (i={i0}, j={j0}, k={k0}): {expected_tile.tolist()}")
    dut._log.info(f"Actual outputs: {outputs}")

    # Step 4: Wait for pipeline to flush
    await ClockCycles(dut.clk, 8)

    # Apply ReLU if enabled
    if relu:
        results_large = np.maximum(results_large, 0)

    return results_large[:m, :p]

@cocotb.test()
async def test_relu_transpose(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    ## FIRST SET OF MATRICES
    A = [5, -6, 7, 8]  # row-major
    B = [8, 9, 6, 8]  # row-major: [B00, B01, B10, B11]

    await load_matrix(dut, A, transpose=0, relu=1)
    await load_matrix(dut, B, transpose=0, relu=1)

    expected = get_expected_matmul(A, B, transpose=False, relu=True)

    ## SECOND SET OF MATRICES
    A = [1, 2, 3, 4]
    B = [5, 6, 7, 8]
    results = await parallel_load_read(dut, A, B, transpose=1, relu=1)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("First part passed")

    expected = get_expected_matmul(A, B, transpose=True, relu=True)
    results = await parallel_load_read(dut, [], [], transpose=1, relu=1)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("ReLU + Transpose test passed!")

@cocotb.test()
async def test_numeric_limits(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 20, units="ns")
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
    # INPUT NEXT ROUND OF MATRICES
    
    A = [125, -64, 124, 108]  # row-major
    B = [99, -12, 105, -106]  # row-major: [B00, B01, B10, B11]
    
    results = await parallel_load_read(dut, A, B)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Passed large positive values")

    expected = get_expected_matmul(A, B)
    results = []

    # Wait for systolic array to compute
    
    results = await parallel_load_read(dut, [], [])

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test passed!")

@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 20, units="ns")
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
    
    # Test 2 matrices
    A = [79, -10, 7, 8]  # row-major
    B = [2, 6, 5, 8]  # row-major: [B00, B01, B10, B11]

    # Read test 1 matrices
    results = await parallel_load_read(dut, A, B)

    # ------------------------------
    # STEP 5: Check results of test 1
    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test 1 passed!")
    
    #######################################
    ##### TEST RUN 2 - CHECK CLEARING #####

    # ------------------------------
    # STEP 4: Get expected of test 2
    expected = get_expected_matmul(A, B)
    results = []

    # TEST RUN 3 MATRICES
    A = [5, -6, 7, 8]  # row-major
    B = [1, 2, 3, -4]  # row-major: [B00, B01, B10, B11]

    # Read test 2 outputs + load test 3 inputs
    results = await parallel_load_read(dut, A, B)

    # ------------------------------
    # STEP 5: Check results of test 2
    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test 2 passed!")

    #########################################
    ##### TEST RUN 3 - CHECK SIGNED OPS #####

    expected = get_expected_matmul(A, B)
    results = []

    # Wait for systolic array to compute
    
    # Test 4 - ABSOLUTE LIMIT
    A = [-128, -128, -128, -128]  # row-major
    B = [127, 127, 127, 127]  # row-major: [B00, B01, B10, B11]

    results = await parallel_load_read(dut, A, B)

    # Check results of test #3
    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test 3 passed!")

    ## Get expected of test 4
    expected = get_expected_matmul(A, B)
    results = []

    A = [-128, -128, -128, -128]  # row-major
    B = [-128, -128, -128, -128]  # row-major: [B00, B01, B10, B11]

    results = await parallel_load_read(dut, A, B)

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test 4 passed!")

    expected = [-32768, -32768, -32768, -32768] # CAUTION: VERY SPECIAL CASE
    results = []

    # No test 6 matrices!!!
    results = await parallel_load_read(dut, [], [])

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test 5 passed")