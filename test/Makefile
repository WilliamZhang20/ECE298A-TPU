# Makefile
# See https://docs.cocotb.org/en/stable/quickstart.html for more info

# defaults
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
SRC_DIR = $(PWD)/../src

TIMESTAMP = $(shell TZ=US/Eastern date +%Y%m%d_%H:%M:%S)
WAVES_ROOT = $(PWD)
COMPILE_ARGS += -DTIMESTAMP=\"$(TIMESTAMP)\"

PROJECT_SOURCES = tpu.v \
					systolic_array_2x2.v \
					control_unit.v \
					PE.v \
					delay_cell.v \
					mmu_feeder.v \
					memory.v

ifneq ($(GATES),yes)

# RTL simulation:
SIM_BUILD				= sim_build/rtl
VERILOG_SOURCES += $(addprefix $(SRC_DIR)/,$(PROJECT_SOURCES))

else

# Gate level simulation:
SIM_BUILD				= sim_build/gl
COMPILE_ARGS    += -DGL_TEST
COMPILE_ARGS    += -DFUNCTIONAL
COMPILE_ARGS    += -DUSE_POWER_PINS
COMPILE_ARGS    += -DSIM
COMPILE_ARGS    += -DUNIT_DELAY=\#1
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/primitives.v
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/sky130_fd_sc_hd.v

# this gets copied in by the GDS action workflow
VERILOG_SOURCES += $(PWD)/gate_level_netlist.v

endif

# Allow sharing configuration between design and testbench via `include`:
COMPILE_ARGS 		+= -I$(SRC_DIR)

.PHONY: all test-uvm test-memory test-systolic-array test-control-unit test-mmu-feeder test-nn test-top

all:
	$(MAKE) test-uvm
	$(MAKE) test-memory
	$(MAKE) test-systolic-array
	$(MAKE) test-control-unit
	$(MAKE) test-mmu-feeder
	$(MAKE) test-nn
	$(MAKE) test-top

test-uvm:
	$(MAKE) clean
	$(MAKE) sim \
		TOPLEVEL=tb \
		MODULE=test_uvm \
		VERILOG_SOURCES="$(WAVES_ROOT)/uvm_test/tb.v $(addprefix $(SRC_DIR)/,tpu.v delay_cell.v systolic_array_2x2.v PE.v mmu_feeder.v control_unit.v memory.v)" \
		PYTHONPATH=$(WAVES_ROOT)/uvm_test \
		WAVES_DIR=$(WAVES_ROOT)/uvm_test/wave \
		COMPILE_ARGS='$(COMPILE_ARGS) -DVCD_PATH="\"$(WAVES_ROOT)/uvm_test/wave/uvm_tb_$(TIMESTAMP).vcd\""'

test-top:
	$(MAKE) clean
	$(MAKE) sim \
		TOPLEVEL=tb \
		MODULE=test_tpu \
		VERILOG_SOURCES="$(WAVES_ROOT)/tpu/tb.v $(addprefix $(SRC_DIR)/,tpu.v delay_cell.v systolic_array_2x2.v PE.v mmu_feeder.v control_unit.v memory.v)" \
		PYTHONPATH=$(WAVES_ROOT)/tpu \
		WAVES_DIR=$(WAVES_ROOT)/tpu/wave \
		COMPILE_ARGS='$(COMPILE_ARGS) -DVCD_PATH="\"$(WAVES_ROOT)/tpu/wave/tpu_tb_$(TIMESTAMP).vcd\""'

test-nn:
	$(MAKE) clean
	$(MAKE) sim \
		TOPLEVEL=tb \
		MODULE=neural_inference \
		VERILOG_SOURCES="$(WAVES_ROOT)/tpu/tb.v $(addprefix $(SRC_DIR)/,tpu.v delay_cell.v systolic_array_2x2.v PE.v mmu_feeder.v control_unit.v memory.v)" \
		PYTHONPATH=$(WAVES_ROOT)/tpu \
		WAVES_DIR=$(WAVES_ROOT)/tpu/wave \
		COMPILE_ARGS='$(COMPILE_ARGS) -DVCD_PATH="\"$(WAVES_ROOT)/tpu/wave/tpu_tb_$(TIMESTAMP).vcd\""'

test-systolic-array:
	$(MAKE) clean
	$(MAKE) sim \
        TOPLEVEL=systolic_array_tb \
        MODULE=test_systolic_array \
        VERILOG_SOURCES="$(WAVES_ROOT)/systolic_array/systolic_array_tb.v $(SRC_DIR)/systolic_array_2x2.v $(SRC_DIR)/PE.v" \
        PYTHONPATH=$(WAVES_ROOT)/systolic_array \
        WAVES_DIR=$(WAVES_ROOT)/systolic_array/wave \
        COMPILE_ARGS='$(COMPILE_ARGS) -DVCD_PATH="\"$(WAVES_ROOT)/systolic_array/wave/systolic_array_tb_$(TIMESTAMP).vcd\""'

test-memory:
	$(MAKE) clean
	$(MAKE) sim \
		TOPLEVEL=memory_tb \
		MODULE=test_memory \
		VERILOG_SOURCES="$(WAVES_ROOT)/memory/memory_tb.v $(SRC_DIR)/memory.v" \
		PYTHONPATH=$(WAVES_ROOT)/memory \
		WAVES_DIR=$(WAVES_ROOT)/memory/wave \
		COMPILE_ARGS='$(COMPILE_ARGS) -DVCD_PATH="\"$(WAVES_ROOT)/memory/wave/memory_tb_$(TIMESTAMP).vcd\""'

test-control-unit:
	$(MAKE) clean
	$(MAKE) sim \
		TOPLEVEL=control_unit_tb \
		MODULE=test_control_unit \
		VERILOG_SOURCES="$(WAVES_ROOT)/control_unit/control_unit_tb.v $(SRC_DIR)/control_unit.v" \
		PYTHONPATH=$(WAVES_ROOT)/control_unit \
		WAVES_DIR=$(WAVES_ROOT)/control_unit/wave \
		COMPILE_ARGS='$(COMPILE_ARGS) -DVCD_PATH="\"$(WAVES_ROOT)/control_unit/wave/control_unit_tb_$(TIMESTAMP).vcd\""'

test-mmu-feeder:
	$(MAKE) clean
	$(MAKE) sim \
		TOPLEVEL=mmu_feeder_tb \
		MODULE=test_mmu_feeder \
		VERILOG_SOURCES="$(WAVES_ROOT)/mmu_feeder/mmu_feeder_tb.v $(SRC_DIR)/mmu_feeder.v" \
		PYTHONPATH=$(WAVES_ROOT)/mmu_feeder \
		WAVES_DIR=$(WAVES_ROOT)/mmu_feeder/wave \
		COMPILE_ARGS='$(COMPILE_ARGS) -DVCD_PATH="\"$(WAVES_ROOT)/mmu_feeder/wave/mmu_feeder_tb_$(TIMESTAMP).vcd\""'

# Phony target for cleaning up
.PHONY: clean
clean::
	rm -rf sim_build* results.xml

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim
