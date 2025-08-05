import cocotb
from cocotb.triggers import RisingEdge, ClockCycles
from cocotb.clock import Clock

@cocotb.test()
async def test_control_unit_reset(dut):
    """Test control unit reset functionality"""
    clock = Clock(dut.clk, 20, units="ns")
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
    clock = Clock(dut.clk, 20, units="ns")
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
    clock = Clock(dut.clk, 20, units="ns")
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
    clock = Clock(dut.clk, 20, units="ns")
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
    # mmu_cycle should increment from 2 to 7 (0 -> 1 done in S_LOAD_MATS state)
    for expected_cycle in range(2, 8):
        print("State: ", int(dut.state_out))
        await ClockCycles(dut.clk, 1)
        assert dut.mmu_en.value == 1, f"mmu_en should remain 1 during compute phase"
        assert dut.mmu_cycle.value.integer == expected_cycle, f"mmu_cycle should be {expected_cycle}, got {dut.mmu_cycle.value}"

        assert dut.load_en.value == 0, f"load_en should remain 0 during compute phase"
        assert dut.mem_addr.value == 0, f"mem_addr should be 0 during compute phase"
    
    # After mmu_cycle reaches 5, should return to IDLE
    # print("State: ", int(dut.state_out))
    await ClockCycles(dut.clk, 1)
    assert dut.mmu_en.value == 1, "mmu_en should be 1 while kept in COMPUTE"
    assert dut.mmu_cycle.value.integer == 0, "mmu_cycle should reset to 0 in COMPUTE"
    assert dut.mem_addr.value == 0, "mem_addr should be 0 in COMPUTE with no LOAD_EN"

    await ClockCycles(dut.clk, 1) 

    # MMU cycle continues cycling although we are not sending anything over....
    
    assert dut.mmu_cycle.value.integer == 1, "mmu_cycle should cycle in compute"
    
    dut._log.info("MMU compute phase test passed")

@cocotb.test()
async def test_control_unit_full_cycle(dut):
    """Test complete control unit operation cycle"""
    clock = Clock(dut.clk, 20, units="ns")
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
            assert dut.mem_addr.value.integer == cycle, f"mem_addr should be {cycle} during loading cycle {cycle}"
    
    # Phase 2: MMU compute (7 cycles: mmu_cycle 2->7->0)
    for cycle in range(2, 8):
        await ClockCycles(dut.clk, 1)
        assert dut.mmu_en.value == 1, f"mmu_en should be 1 during MMU cycle {cycle}"
        assert dut.mem_addr.value == cycle-2, f"mem_addr should be {cycle-2} during MMU cycle {dut.mmu_cycle.value.integer}"
    
    # Phase 3: pretending that we are continuing to load future matrice with load_en high... and mem_addr going up
    await ClockCycles(dut.clk, 1)
    assert dut.mmu_en.value == 1, "mmu_en should be 1 while kept in COMPUTE"
    assert dut.mmu_cycle.value.integer == 0, "mmu_cycle should reset to 0 in COMPUTE"
    assert dut.mem_addr.value.integer == 6, "mem_addr should be 0 in COMPUTE with no LOAD_EN"
    
    dut._log.info("Full cycle test passed")

@cocotb.test()
async def test_control_unit_shaky_load_en(dut):
    """Test that load_en is ignored during MMU compute phase"""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Load matrices, toggle load_en, should only proceed on load_en
    for cycle in range(8):
        dut.load_en.value = 1
        await ClockCycles(dut.clk, 1)
        assert dut.mem_addr.value.integer == cycle

    # Now in MMU compute phase
    for cycle in range(2, 8):
        dut.load_en.value = 1 if cycle % 2 == 0 else 0
        await ClockCycles(dut.clk, 1)
        # mmu_cycle should still increment normally regardless of load_en
        assert dut.mmu_cycle.value.integer == cycle, f"mmu_cycle should increment normally during compute phase"
    
    dut._log.info("Shaky load enable during compute test passed")

@cocotb.test()
async def test_control_unit_multiple_operations(dut):
    """Test multiple complete operation cycles"""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Init Load phase
    dut.load_en.value = 1
    for _ in range(8):
        await ClockCycles(dut.clk, 1)

    # Run 100 complete cycles
    for operation in range(100):
        # Commented below line for clarity of test action
        # dut._log.info(f"Starting operation {operation + 1}")
        
        # output/input phase
        # Note that load_en is still asserted high
        for val in range(8):
            await ClockCycles(dut.clk, 1)
            assert dut.mmu_en.value == 1, f"Operation {operation + 1}: mmu_en should be 1 during MMU phase"
            assert dut.mmu_cycle.value.integer == (val + 2) % 8, f"MMU Cycle not running properly as it should be {val}"
    
        # Wait in idle before next operation
        dut.load_en.value = 0
    
    dut._log.info("Multiple operations test passed")