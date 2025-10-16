// TinyHWDialect.h
#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace tinyhw {
void registerTinyHWOps();
class TinyHWDialect : public Dialect {
public:
  explicit TinyHWDialect(MLIRContext *ctx);
};
} // namespace tinyhw
} // namespace mlir
