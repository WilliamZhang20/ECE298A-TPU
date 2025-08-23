from pyuvm import uvm_object
from pyuvm import ConfigDB
import cocotb

from pyuvm import uvm_subscriber, UVMConfigItemNotFound
import numpy as np

class MatMulCoverage(uvm_subscriber):
    def end_of_elaboration_phase(self):
        # A set to store the covered combinations (e.g., (transpose, relu))
        self.covered_combos = set()
        # Define all possible combinations you want to cover
        self.all_combos = {(False, False), (True, False), (False, True), (True, True)}

    def write(self, txn):
        # This method is called whenever a transaction is sent to the analysis port
        combination = (txn.transpose, txn.relu)
        self.covered_combos.add(combination)
        self.logger.info(f"COVERAGE: Captured combination: {combination}")

    def report_phase(self):
        # The report phase runs at the end of the test
        missed_combos = self.all_combos - self.covered_combos
        if len(missed_combos) > 0:
            self.logger.error(f"Functional coverage error. Missed combinations: {missed_combos}")
            # TODO: COVER ALL!
            # assert False, "Functional coverage failed"
        else:
            self.logger.info("Covered all specified combinations.")
            # assert True

class MatMulConfig(uvm_object):
    def __init__(self, name="MatMulConfig"):
        super().__init__(name)
        self.dut = cocotb.top
        self.bfm = None

import cocotb
from cocotb.triggers import RisingEdge, ClockCycles

class MatMulBfm:
    def __init__(self, dut):
        self.dut = dut
        self.log = dut._log

    async def send_op(self, A, B, transpose, relu):
        """Drives the input signals of the DUT based on transaction data."""
        self.log.info("BFM: Sending inputs for A and B")
        # Load A
        for val in A:
            self.dut.ui_in.value = val
            self.dut.uio_in.value = (transpose << 1) | (relu << 2) | 1
            await RisingEdge(self.dut.clk)
        
        # Load B
        for val in B:
            self.dut.ui_in.value = val
            self.dut.uio_in.value = (transpose << 1) | (relu << 2) | 1
            await RisingEdge(self.dut.clk)

    async def get_result(self):
        """Passively observes and reads the output signals from the DUT."""
        results = []
        self.log.info("BFM: Reading output")
        # Assuming a fixed latency. In a real design, you'd wait for a signal
        await ClockCycles(self.dut.clk, 5) 

        for _ in range(4): 
            await ClockCycles(self.dut.clk, 1)
            high = self.dut.uo_out.value.integer
            await ClockCycles(self.dut.clk, 1)
            low = self.dut.uo_out.value.integer
            combined = (high << 8) | low
            if combined >= 0x8000:
                combined -= 0x10000
            results.append(combined)
        return results

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

from pyuvm import uvm_driver, uvm_analysis_port
from cocotb.triggers import RisingEdge
import cocotb

class MatMulDriver(uvm_driver):
    def build_phase(self):
        super().build_phase()
        self.cmd_ap = uvm_analysis_port("cmd_ap", self)

    def start_of_simulation_phase(self):
        # Get the config object from the ConfigDB
        self.config = ConfigDB().get(self, "", "matmul_config")
        # Access the BFM from the config object
        self.bfm = self.config.bfm

    async def run_phase(self):
        # ... (same as before, using self.bfm)
        while True:
            txn = await self.seq_item_port.get_next_item()
            await self.bfm.send_op(txn.A, txn.B, txn.transpose, txn.relu)
            self.cmd_ap.write(txn)
            self.seq_item_port.item_done()

from pyuvm import uvm_component, uvm_analysis_port

class MatMulMonitor(uvm_component):
    def build_phase(self):
        super().build_phase()
        self.ap = uvm_analysis_port("ap", self)

    def start_of_simulation_phase(self):
        # Get the config object from the ConfigDB
        self.config = ConfigDB().get(self, "", "matmul_config")
        # Access the BFM from the config object
        self.bfm = self.config.bfm

    async def run_phase(self):
        while True:
            actual_result = await self.bfm.get_result()
            self.ap.write(actual_result)

from pyuvm import uvm_component, uvm_tlm_analysis_fifo

class MatMulScoreboard(uvm_component):
    def build_phase(self):
        super().build_phase()
        self.cmd_fifo = uvm_tlm_analysis_fifo("cmd_fifo", self) # New FIFO for commands
        self.out_fifo = uvm_tlm_analysis_fifo("out_fifo", self)
        self.cmd_export = self.cmd_fifo.analysis_export
        self.out_export = self.out_fifo.analysis_export

    def check_phase(self):
        passed = True
        while self.out_fifo.can_get():
            _, got = self.out_fifo.try_get()
            _, cmd_txn = self.cmd_fifo.try_get() # Get the original transaction
            
            # The scoreboard now calculates the expected value
            exp = cmd_txn.expected()
            
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
        self.config = MatMulConfig("matmul_config")
        self.config.bfm = MatMulBfm(self.config.dut)
        ConfigDB().set(self, "*", "matmul_config", self.config)
        
        self.driver = MatMulDriver("driver", self)
        self.monitor = MatMulMonitor("monitor", self)
        self.scoreboard = MatMulScoreboard("scoreboard", self)
        
        self.coverage = MatMulCoverage("coverage", self)

    def connect_phase(self):
        self.driver.seq_item_port.connect(self.seqr.seq_item_export)
        self.driver.cmd_ap.connect(self.scoreboard.cmd_export)
        self.monitor.ap.connect(self.scoreboard.out_export)
        self.driver.cmd_ap.connect(self.coverage.analysis_export)

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