// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <vector>

namespace onnxruntime {
namespace contrib {

class Ngram : public OpKernel {
 public:
  explicit Ngram(const OpKernelInfo& info);
  ~Ngram() = default;

  Status Compute(OpKernelContext* ctx) const override;

 private:
  enum Mode {
    kNone = 0,
    kTF = 1,
    kIDF = 2,
    kTFIDF = 3
  };

  int64_t N_;
  int64_t M_;
  int64_t S_;
  bool all_;
  std::vector<int64_t> ngram_counts_;
  std::vector<int64_t> ngram_indexes_;
  std::vector<float> weights_;
  Mode mode_;
};

}  // namespace contrib
}  // namespace onnxruntime
