from pyuvm import uvm_sequence_item
import numpy as np

class MatMulSeqItem(uvm_sequence_item):
    def __init__(self, name, A=None, B=None, transpose=False, relu=False):
        super().__init__(name)
        self.A = A or [0,0,0,0]   # 2x2 row-major
        self.B = B or [0,0,0,0]
        self.transpose = transpose
        self.relu = relu
        self.result = None

    def expected(self):
        A_mat = np.array(self.A).reshape(2,2)
        B_mat = np.array(self.B).reshape(2,2)
        if self.transpose:
            B_mat = B_mat.T
        result = A_mat @ B_mat
        if self.relu:
            result = np.maximum(result, 0)
        return result.flatten().tolist()

from pyuvm import uvm_sequence_item
import numpy as np

class MatMulSeqItem(uvm_sequence_item):
    def __init__(self, name, A=None, B=None, transpose=False, relu=False):
        super().__init__(name)
        self.A = A or [0,0,0,0]   # 2x2 row-major
        self.B = B or [0,0,0,0]
        self.transpose = transpose
        self.relu = relu
        self.result = None

    def expected(self):
        A_mat = np.array(self.A).reshape(2,2)
        B_mat = np.array(self.B).reshape(2,2)
        if self.transpose:
            B_mat = B_mat.T
        result = A_mat @ B_mat
        if self.relu:
            result = np.maximum(result, 0)
        return result.flatten().tolist()

from pyuvm import uvm_driver
from cocotb.triggers import RisingEdge

class MatMulDriver(uvm_driver):
    def build_phase(self):
        self.exp_ap = uvm_analysis_port("exp_ap", self)

    async def run_phase(self):
        dut = cocotb.top
        while True:
            txn = await self.seq_item_port.get_next_item()

            # Load A (4 cycles)
            for val in txn.A:
                dut.ui_in.value = val
                dut.uio_in.value = (txn.transpose << 1) | (txn.relu << 2) | 1
                await RisingEdge(dut.clk)

            # Load B (4 cycles)
            for val in txn.B:
                dut.ui_in.value = val
                dut.uio_in.value = (txn.transpose << 1) | (txn.relu << 2) | 1
                await RisingEdge(dut.clk)

            await ClockCycles(dut.clk, 5) # A safe guess for a small pipeline latency.
            
            # Now, the monitor has had time to read the result.
            # Send expected results to scoreboard.
            self.exp_ap.write(txn.expected())

            self.seq_item_port.item_done()

from pyuvm import uvm_component, uvm_analysis_port
from cocotb.triggers import ClockCycles

class MatMulMonitor(uvm_component):
    def build_phase(self):
        self.ap = uvm_analysis_port("ap", self)
        self.dut = cocotb.top

    async def run_phase(self):
        dut = self.dut
        while True:
            results = []
            
            for _ in range(4): # 4 values to read
                dut.ui_in.value = 0 # Dummy input during read phase
                await ClockCycles(dut.clk, 1)
                high = dut.uo_out.value.integer
                await ClockCycles(dut.clk, 1)
                low = dut.uo_out.value.integer
                
                combined = (high << 8) | low
                if combined >= 0x8000:
                    combined -= 0x10000
                results.append(combined)
                
            # Corrected monitor loop
            all_results = []
            for _ in range(4): # Loop to read 4 final values
                await ClockCycles(dut.clk, 1)
                high = dut.uo_out.value.integer
                await ClockCycles(dut.clk, 1)
                low = dut.uo_out.value.integer
                combined = (high << 8) | low
                if combined >= 0x8000:
                    combined -= 0x10000
                all_results.append(combined)

            self.ap.write(all_results)

from pyuvm import uvm_component, uvm_tlm_analysis_fifo

class MatMulScoreboard(uvm_component):
    def build_phase(self):
        self.exp_fifo = uvm_tlm_analysis_fifo("exp_fifo", self)
        self.out_fifo = uvm_tlm_analysis_fifo("out_fifo", self)
        self.exp_export = self.exp_fifo.analysis_export
        self.out_export = self.out_fifo.analysis_export

    def check_phase(self):
        passed = True
        while self.out_fifo.can_get():
            _, got = self.out_fifo.try_get()
            _, exp = self.exp_fifo.try_get()
            if got != exp:
                self.logger.error(f"FAIL: got {got}, expected {exp}")
                passed = False
            else:
                self.logger.info(f"PASS: {got} == {exp}")
        assert passed

from pyuvm import uvm_env, uvm_sequencer

class MatMulEnv(uvm_env):
    def build_phase(self):
        self.seqr = uvm_sequencer("seqr", self)
        self.driver = MatMulDriver.create("driver", self)
        self.monitor = MatMulMonitor("monitor", self)
        self.scoreboard = MatMulScoreboard("scoreboard", self)

    def connect_phase(self):
        self.driver.seq_item_port.connect(self.seqr.seq_item_export)
        self.monitor.ap.connect(self.scoreboard.out_export)
        self.driver.exp_ap.connect(self.scoreboard.exp_export)

from pyuvm import uvm_sequence, uvm_sequence_item

class MatrixTxn(uvm_sequence_item):
    def __init__(self, name="MatrixTxn"):
        super().__init__(name)
        self.A = [0]*4
        self.B = [0]*4
        self.transpose = 0
        self.relu = 0

    def expected(self):
        import numpy as np
        A = np.array(self.A).reshape(2,2)
        B = np.array(self.B).reshape(2,2)
        if self.transpose:
            B = B.T
        result = A @ B
        if self.relu:
            result = np.maximum(result, 0)
        return result.flatten().tolist()

from pyuvm import uvm_sequence

class SimpleSeq(uvm_sequence):
    async def body(self):
        txn = MatrixTxn("txn1")      # create transaction
        txn.A = [1, 2, 3, 4]
        txn.B = [5, 6, 7, 8]
        txn.transpose = 1
        txn.relu = 1

        await self.start_item(txn)   # pass the transaction to the driver
        await self.finish_item(txn)  # mark it done

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from pyuvm import uvm_root, uvm_test

class MatMulTest(uvm_test):
    def build_phase(self):
        self.env = MatMulEnv("env", self)

    async def run_phase(self):
        self.raise_objection()
        seq = SimpleSeq("seq")
        await seq.start(self.env.seqr)
        self.drop_objection()

@cocotb.test()
async def run_uvm(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # Reset sequence
    dut.rst_n.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1

    await uvm_root().run_test("MatMulTest")