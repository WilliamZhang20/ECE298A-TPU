// LowerToTinyHW.cpp
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "TinyHWDialect.h"

using namespace mlir;

namespace {
struct MatmulToTinyHW : public OpRewritePattern<linalg::MatmulOp> {
  MatmulToTinyHW(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    // Minimal checks: shapes must be statically 2x2 and element types i8 -> i32
    auto lhsType = matmul.getOperand(0).getType().dyn_cast<RankedTensorType>();
    auto rhsType = matmul.getOperand(1).getType().dyn_cast<RankedTensorType>();
    auto resType = matmul.getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType || !resType) return failure();

    if (lhsType.getRank() != 2 || rhsType.getRank() != 2) return failure();
    if (lhsType.getDimSize(0) != 2 || lhsType.getDimSize(1) != 2) return failure();
    if (rhsType.getDimSize(0) != 2 || rhsType.getDimSize(1) != 2) return failure();

    // require element types: lhs i8, rhs i8, res i32
    if (!lhsType.getElementType().isInteger(8) ||
        !rhsType.getElementType().isInteger(8) ||
        !resType.getElementType().isInteger(32))
      return failure();

    // Replace with tinyhw.matmul_int8
    Location loc = matmul.getLoc();
    Value lhs = matmul.getOperand(0);
    Value rhs = matmul.getOperand(1);

    // Build op: tinyhw.matmul_int8 %lhs, %rhs : tensor<2x2xi8>, tensor<2x2xi8> -> tensor<2x2xi32>
    auto tinyResType = resType;
    SmallVector<Type, 1> results{tinyResType};
    OperationState state(loc, "tinyhw.matmul_int8", {lhs, rhs}, results);
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(matmul, newOp->getResults());
    return success();
  }
};

struct LowerToTinyHWPass : public PassWrapper<LowerToTinyHWPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<MatmulToTinyHW>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

// registration
static PassRegistration<LowerToTinyHWPass> pass("lower-to-tinyhw",
    "Lower small linalg.matmul ops to tinyhw.matmul_int8");
