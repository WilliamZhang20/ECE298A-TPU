// TinyHWCodegen.cpp
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/Module.h"
#include "TinyHWDialect.h"
#include "mlir/IR/Builders.h"
#include <fstream>

using namespace mlir;

namespace {
struct TinyHWCodegenPass : public PassWrapper<TinyHWCodegenPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::ofstream py("generated_cocotb.py");
    if (!py.is_open()) {
      module.emitError() << "Failed to open generated_cocotb.py for writing";
      return signalPassFailure();
    }

    // Write header
    py << "import cocotb\nfrom cocotb.clock import Clock\nfrom cocotb_helpers import load_matrix, parallel_load_read, reset_dut\n\n";
    py << "@cocotb.test()\nasync def test_generated_model(dut):\n";
    py << "    clock = Clock(dut.clk, 20, units='ns')\n";
    py << "    cocotb.start_soon(clock.start())\n";
    py << "    await reset_dut(dut)\n\n";

    // Walk and emit per-operation code: when we hit tinyhw.matmul_int8 or tinyhw.relu
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name.equals("tinyhw.matmul_int8")) {
        // For this prototype, assume operands are constant ops containing small 2x2 int tensors.
        // We'll try to print simple placeholders.
        py << "    # tinyhw.matmul_int8\n";
        py << "    A_tile = [0,0,0,0]  # REPLACE with actual tile data or constants\n";
        py << "    B_tile = [0,0,0,0]\n";
        py << "    await load_matrix(dut, A_tile)\n";
        py << "    await load_matrix(dut, B_tile)\n";
        py << "    results = await parallel_load_read(dut, A_tile, B_tile)\n";
        py << "    # results -> use as needed\n\n";
      } else if (name.equals("tinyhw.relu")) {
        py << "    # tinyhw.relu (software fallback)\n";
        py << "    # results = [max(0, v) for v in results]\n\n";
      }
    });

    py << "    dut._log.info('Generated test finished')\n";
    py.close();
  }
};
} // namespace

static PassRegistration<TinyHWCodegenPass> pass("tinyhw-codegen",
    "Emit cocotb Python script from tinyhw dialect");
