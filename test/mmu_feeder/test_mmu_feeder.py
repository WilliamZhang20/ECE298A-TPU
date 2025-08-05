import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import numpy as np

def s16_bytes(x):
    """Split signed 16-bit int into (hi, lo) unsigned bytes."""
    if x < 0:
        x = (1 << 16) + x
    return (x >> 8) & 0xFF, x & 0xFF

def bytesToInt16(x):
    if x >= 0x8000:
        return x - 0x10000
    return x

def unsigned_to_signed(value):
    """Convert unsigned value to signed."""
    return value if value < 128 else value - 256

def assert_equal_fields(dut, expected_dict, conversion=None):
    for name, expected in expected_dict.items():
        actual = getattr(dut, name).value.integer
        if conversion:
            actual = conversion(actual)
        assert actual == expected, f"{name}: got {actual}, expected {expected}"

async def reset_dut(dut):
    dut.rst.value = 1
    dut.en.value = 0
    dut.mmu_cycle.value = 0
    dut.transpose.value = 0
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
    await ClockCycles(dut.clk, 1)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

def random_2x2_matrix(low=-128, high=127):
    return np.random.randint(low, high, size=(2, 2), dtype=np.int8)

@cocotb.test()
async def random_test_vecs(dut):
    """
    Tests the mmu_feeder with multiple random matrices, processed sequentially.
    """
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    
    N = 10 # Number of random test cases

    # Pre-generate all test cases
    testcases = []
    for _ in range(N):
        W_mat = random_2x2_matrix()
        I_mat = random_2x2_matrix()
        transpose = np.random.randint(0, 2)

        A = np.array(W_mat, dtype=np.int16).reshape((2, 2))
        B_raw = np.array(I_mat, dtype=np.int16).reshape((2, 2))
        B = B_raw.T if transpose else B_raw
        C = A @ B

        testcases.append({
            "W": W_mat.flatten().tolist(),
            "I": I_mat.flatten().tolist(),
            "transpose": transpose,
            "C_expected": C.flatten().tolist(),
            "A": A,
            "B_raw": B_raw,
        })

    # Process each test case one by one
    for j, test in enumerate(testcases):
        dut._log.info(f"--- Running Random Test Case {j} ---")
        dut._log.info(f"W:\n{test['A']}\nI:\n{test['B_raw']}\nTranspose: {test['transpose']}\nExpected C:\n{test['C_expected']}")
        
        # Reset DUT for a clean state before each test
        await reset_dut(dut)
        dut.en.value = 1

        # Get data for the current test
        W = test["W"]
        I = test["I"]
        C_expected = test["C_expected"]
        dut.transpose.value = test["transpose"]

        result_bytes = []

        # --- CYCLE 0: Load first inputs ---
        dut.mmu_cycle.value = 0
        dut.weight0.value = W[0]
        dut.input0.value = I[0]
        await RisingEdge(dut.clk)

        # --- CYCLE 1: Load more inputs and first result C[0] ---
        dut.mmu_cycle.value = 1
        dut.weight1.value = W[1]
        dut.weight2.value = W[2]
        dut.input1.value = I[1]
        dut.input2.value = I[2]
        dut.c00.value = int(C_expected[0])
        await RisingEdge(dut.clk)

        # --- CYCLE 2: Load final inputs, more results, capture first output byte ---
        dut.mmu_cycle.value = 2
        dut.weight3.value = W[3]
        dut.input3.value = I[3]
        dut.c01.value = int(C_expected[1])
        dut.c10.value = int(C_expected[2])
        await RisingEdge(dut.clk)
        result_bytes.append(dut.host_outdata.value.integer)

        # --- CYCLE 3: Load final result, capture second output byte ---
        dut.mmu_cycle.value = 3
        dut.c11.value = int(C_expected[3])
        await RisingEdge(dut.clk)
        result_bytes.append(dut.host_outdata.value.integer)
        
        # --- CYCLES 4-7: Capture remaining output bytes ---
        for i in range(4, 8):
            dut.mmu_cycle.value = i
            await RisingEdge(dut.clk)
            result_bytes.append(dut.host_outdata.value.integer)

        # --- Final two clocks to flush pipeline ---
        for i in range(2):
            dut.mmu_cycle.value = i
            await RisingEdge(dut.clk)
            result_bytes.append(dut.host_outdata.value.integer)

        # --- Verification ---
        # Reconstruct 16-bit signed words from the captured bytes
        words = [
            (result_bytes[0] << 8) | result_bytes[1],
            (result_bytes[2] << 8) | result_bytes[3],
            (result_bytes[4] << 8) | result_bytes[5],
            (result_bytes[6] << 8) | result_bytes[7],
        ]
        words = [bytesToInt16(x) for x in words]

        # Assert that the received words match the expected results
        for k, (expected, got) in enumerate(zip(C_expected, words)):
            assert expected == got, f"[Test {j}] C[{k}] = {got} != expected {expected}"
        
        dut._log.info(f"[Test {j}] Passed\n")

@cocotb.test()
async def test_mmu_feeder(dut):
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)

    test_vectors = [
        # Regular test
        {
            "name": "Regular",
            "W": [1, 2, 3, 4],
            "I": [5, 6, 7, 8],
            "transpose": 0,
        },
        # Signed test
        {
            "name": "Signed edge",
            "W": [127, -128, 127, -128],
            "I": [127, 127, 127, 127],
            "transpose": 0,
        },
        # Transpose test
        {
            "name": "Transpose enabled",
            "W": [1, 2, 3, 4],
            "I": [5, 6, 7, 8],
            "transpose": 1,
        },
    ]

    for vec in test_vectors:
        dut._log.info(f"--- Running {vec['name']} ---")
        await reset_dut(dut)
        dut.en.value = 1
        dut.transpose.value = vec["transpose"]
        I = vec["I"]
        W = vec["W"]

        A = np.array(W, dtype=np.int16).reshape((2, 2))
        B_raw = np.array(I, dtype=np.int16).reshape((2, 2))
        B = B_raw.T if vec["transpose"] else B_raw
        dut._log.info(A)
        dut._log.info(B)
        C = A @ B  # Matrix multiply
        dut._log.info(C)

        # Flatten row-major output
        flat = C.flatten()
        
        result_bytes = []

        # --- CYCLE 0 ---
        dut.mmu_cycle.value = 0
        dut.weight0.value = W[0]
        dut.input0.value = I[0]
        await RisingEdge(dut.clk)
        assert_equal_fields(dut, {
            "a_data0": W[0],
            "a_data1": 0,
            "b_data0": I[0],
            "b_data1": 0,
        }, conversion=unsigned_to_signed)
        assert dut.clear.value == 1
        assert dut.done.value == 0

        # --- CYCLE 1 ---
        dut.mmu_cycle.value = 1
        dut.weight1.value = W[1]
        dut.weight2.value = W[2]
        dut.input1.value = I[1]
        dut.input2.value = I[2]
        dut.c00.value = int(flat[0])
        await RisingEdge(dut.clk)

        # Expected b_data depends on transpose
        b0 = I[1] if vec["transpose"] else I[2]
        b1 = I[2] if vec["transpose"] else I[1]
        
        assert_equal_fields(dut, {
            "a_data0": W[1],
            "a_data1": W[2],
            "b_data0": b0,
            "b_data1": b1,
        }, conversion=unsigned_to_signed)

        assert dut.clear.value == 0
        assert dut.done.value == 0

        # --- CYCLE 2 ---
        dut.mmu_cycle.value = 2
        dut.weight3.value = W[3]
        dut.input3.value = I[3]
        dut.c01.value = int(flat[1])
        dut.c10.value = int(flat[2])
        await RisingEdge(dut.clk)
        result_bytes.append(dut.host_outdata.value.integer)

        assert_equal_fields(dut, {
            "a_data0": 0,
            "a_data1": W[3],
            "b_data0": 0,
            "b_data1": I[3],
        }, conversion=unsigned_to_signed)

        assert dut.clear.value == 0
        assert dut.done.value == 1

        # --- CYCLE 3 ---
        dut.mmu_cycle.value = 3
        dut.c11.value = int(flat[3])
        await RisingEdge(dut.clk)
        
        result_bytes.append(dut.host_outdata.value.integer)
        assert dut.done.value == 1

        # --- CYCLES 4–7: read host_outdata bytes ---
        for i in range(4, 8):
            dut.mmu_cycle.value = i
            await RisingEdge(dut.clk)
            result_bytes.append(dut.host_outdata.value.integer)

        # -- Last two values, when mmu cycle reset to 0, 1 for next computation, prev outputs are still appearing --
        for i in range(2):
            dut.mmu_cycle.value = i
            await RisingEdge(dut.clk)
            result_bytes.append(dut.host_outdata.value.integer)

        # Parse high/low byte pairs
        words = [
            (result_bytes[0] << 8) | result_bytes[1],
            (result_bytes[2] << 8) | result_bytes[3],
            (result_bytes[4] << 8) | result_bytes[5],
            (result_bytes[6] << 8) | result_bytes[7],
        ]

        words = [bytesToInt16(x) for x in words]

        for i, (exp, got) in enumerate(zip(flat, words)):
            assert exp == got, f"[{vec['name']}] C[{i}] = {got} != expected {exp}"
        
        dut._log.info(f"{vec['name']} passed")
