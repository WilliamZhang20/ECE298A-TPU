import cocotb
from cocotb.triggers import RisingEdge, ClockCycles
from cocotb.clock import Clock

@cocotb.test()
async def test_control_unit_reset(dut):
    """Test control unit reset functionality"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Apply reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    
    # Check reset state
    assert dut.mem_addr.value == 0, f"mem_addr should be 0 after reset, got {dut.mem_addr.value}"
    assert dut.mmu_en.value == 0, f"mmu_en should be 0 after reset, got {dut.mmu_en.value}"
    assert dut.mmu_cycle.value == 0, f"mmu_cycle should be 0 after reset, got {dut.mmu_cycle.value}"
    
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)
    
    dut._log.info("Reset test passed")

@cocotb.test()
async def test_control_unit_idle_state(dut):
    """Test control unit stays in IDLE when load_en is not asserted"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Stay in idle for several cycles
    for _ in range(5):
        assert dut.mem_addr.value == 0, f"mem_addr should remain 0 in idle, got {dut.mem_addr.value}"
        assert dut.mmu_en.value == 0, f"mmu_en should remain 0 in idle, got {dut.mmu_en.value}"
        assert dut.mmu_cycle.value == 0, f"mmu_cycle should remain 0 in idle, got {dut.mmu_cycle.value}"
        await ClockCycles(dut.clk, 1)
    
    dut._log.info("Idle state test passed")

@cocotb.test()
async def test_control_unit_load_matrices(dut):
    """Test control unit matrix loading phase"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Start loading - first load_en pulse should trigger transition to LOAD_MATS
    dut.load_en.value = 1
    await ClockCycles(dut.clk, 1)

    # Check memory address increments correctly during loading
    expected_addrs = [0, 1, 2, 3, 4, 5, 6, 7]
    for i, expected_addr in enumerate(expected_addrs):
        # Check current state BEFORE the clock edge
        assert int(dut.mem_addr.value) == expected_addr, f"Cycle {i+1}: mem_addr should be {expected_addr}, got {dut.mem_addr.value}"

        current_loaded = int(dut.mem_addr.value)
        # We check when loaded = 6 and not when =5
        # The value is "set" when =5
        # Sequential regs capture the value on the next clock edge
        # So we need to wait another cycle
        if current_loaded >= 6:
            assert dut.mmu_en.value == 1, f"Cycle {i+1}: mmu_en should be 1 when mat_elems_loaded >= 6"
        else:
            assert dut.mmu_en.value == 0, f"Cycle {i+1}: mmu_en should be 0 when mat_elems_loaded < 5"

        # Wait for next clock edge (this is when assignments happen)
        await ClockCycles(dut.clk, 1)

    # After the loop, we've had 8 clock cycles
    # Check final state after all loading is complete
    # At this point, mat_elems_loaded should have been reset to 0 (from the == 7 condition)
    assert dut.mem_addr.value == 0, "mem_addr should be 0 after loading all 8 elements"
    assert dut.mmu_en.value == 1, "mmu_en should be 1 after loading all 8 elements"
    dut.load_en.value = 0

    # mmu_cycle should have incremented - check based on your logic
    # If it starts at 0 and increments when mat_elems_loaded >= 6, 
    # it should be incremented multiple times during the loading
    print(f"Final state: mmu_cycle = {dut.mmu_cycle.value}")

    dut._log.info("Matrix loading test passed")

@cocotb.test()
async def test_control_unit_mmu_compute_phase(dut):
    """Test control unit MMU compute and writeback phase"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Load all 8 elements quickly
    dut.load_en.value = 1
    print("State: ", int(dut.state_out))
    for _ in range(8):
        print("State: ", int(dut.state_out))
        await ClockCycles(dut.clk, 1)
    dut.load_en.value = 0
    print("State: ", int(dut.state_out))
    
    # Now in MMU_FEED_COMPUTE_WB state
    # mmu_cycle should increment from 1 to 5 (0 -> 1 done in S_LOAD_MATS state)
    expected_cycles = [1, 2, 3, 4]
    for expected_cycle in expected_cycles:
        print("State: ", int(dut.state_out))
        assert dut.mmu_en.value == 1, f"mmu_en should remain 1 during compute phase"
        assert dut.mmu_cycle.value == expected_cycle, f"mmu_cycle should be {expected_cycle}, got {dut.mmu_cycle.value}"

        await ClockCycles(dut.clk, 1)
        assert dut.load_en.value == 0, f"load_en should remain 0 during compute phase"
        assert dut.mem_addr.value == 0, f"mem_addr should be 0 during compute phase"
    
    # After mmu_cycle reaches 5, should return to IDLE
    print("State: ", int(dut.state_out))
    await ClockCycles(dut.clk, 2)
    assert dut.mmu_en.value == 0, "mmu_en should be 0 after returning to IDLE"
    assert dut.mmu_cycle.value == 0, "mmu_cycle should reset to 0 in IDLE"
    assert dut.mem_addr.value == 0, "mem_addr should be 0 in IDLE"
    
    dut._log.info("MMU compute phase test passed")

@cocotb.test()
async def test_control_unit_full_cycle(dut):
    """Test complete control unit operation cycle"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Full cycle: IDLE -> LOAD_MATS -> MMU_FEED_COMPUTE_WB -> IDLE
    
    # Phase 1: Load matrices (8 cycles)
    dut.load_en.value = 1
    for cycle in range(8):
        await ClockCycles(dut.clk, 1)
        if cycle < 7:  # During loading
            assert dut.mem_addr.value == cycle + 2, f"mem_addr should be {cycle + 2} during loading cycle {cycle}"
    
    # Phase 2: MMU compute (5 cycles: mmu_cycle 1->5)
    for cycle in range(5):
        assert dut.mmu_en.value == 1, f"mmu_en should be 1 during MMU cycle {cycle}"
        assert dut.mem_addr.value == 0, f"mem_addr should be 0 during MMU cycle {cycle}"
        await ClockCycles(dut.clk, 1)
    
    # Phase 3: Back to IDLE
    assert dut.mmu_en.value == 0, "Should return to IDLE after MMU phase"
    assert dut.mmu_cycle.value == 0, "mmu_cycle should reset in IDLE"
    assert dut.mem_addr.value == 0, "mem_addr should be 0 in IDLE"
    
    dut._log.info("Full cycle test passed")

@cocotb.test()
async def test_control_unit_load_en_during_compute(dut):
    """Test that load_en is ignored during MMU compute phase"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Load matrices
    dut.load_en.value = 1
    for _ in range(8):
        await ClockCycles(dut.clk, 1)

    # Now in MMU compute phase - toggle load_en, should be ignored
    for cycle in range(3):
        dut.load_en.value = 1 if cycle % 2 == 0 else 0
        current_mmu_cycle = int(dut.mmu_cycle.value)
        await ClockCycles(dut.clk, 1)
        # mmu_cycle should still increment normally regardless of load_en
        assert dut.mmu_cycle.value == current_mmu_cycle + 1, f"mmu_cycle should increment normally during compute phase"
        assert dut.mem_addr.value == 0, "mem_addr should remain 0 during compute phase"
    
    dut._log.info("Load enable during compute test passed")

@cocotb.test()
async def test_control_unit_multiple_operations(dut):
    """Test multiple complete operation cycles"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Run two complete cycles
    for operation in range(2):
        dut._log.info(f"Starting operation {operation + 1}")
        
        # Load phase
        dut.load_en.value = 1
        for _ in range(8):
            await ClockCycles(dut.clk, 1)
        
        # MMU phase
        for _ in range(5):
            assert dut.mmu_en.value == 1, f"Operation {operation + 1}: mmu_en should be 1 during MMU phase"
            await ClockCycles(dut.clk, 1)
        
        # Should be back in IDLE
        assert dut.mmu_en.value == 0, f"Operation {operation + 1}: Should return to IDLE"
        assert dut.mmu_cycle.value == 0, f"Operation {operation + 1}: mmu_cycle should reset"
        
        # Wait in idle before next operation
        dut.load_en.value = 0
        await ClockCycles(dut.clk, 2)
    
    dut._log.info("Multiple operations test passed")
