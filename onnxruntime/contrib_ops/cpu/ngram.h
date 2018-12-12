// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <memory>
#include <vector>

namespace onnxruntime {
namespace contrib {

class Ngram final : public OpKernel {
 public:
  explicit Ngram(const OpKernelInfo& info);
  ~Ngram();
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Ngram);

  Status Compute(OpKernelContext* ctx) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace contrib
}  // namespace onnxruntime
