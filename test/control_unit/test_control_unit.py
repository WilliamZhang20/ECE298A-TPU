import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock

@cocotb.test()
async def minimal_control_unit_test(dut):
    """Minimal test for control_unit FSM"""
    clock = Clock(dut.clk, 5, units="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Load 8 elements by pulsing load_en
    for _ in range(7):
        dut.load_en.value = 1
        await RisingEdge(dut.clk)

     # === Assertions ===

    # 1. MMU should be enabled (mmu_en = 1) in MMU compute state
    assert dut.mmu_en.value == 1, "mmu_en should be high after loading matrices"

    # Log for visibility
    dut._log.info(f"PASS: mmu_en = {dut.mmu_en.value}")
