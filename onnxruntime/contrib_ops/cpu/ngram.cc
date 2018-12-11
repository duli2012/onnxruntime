// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ngram.h"
#include "onnx/defs/schema.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"

#include <functional>

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    Ngram,
    1,
    string,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<std::string>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    contrib::Ngram);

namespace ngram_details {

// 1-gram element
template <typename T>
struct OneGram {
  size_t index_;
  T elem_;
};

template <typename T>
struct ManyGram {
  size_t index_;
  std::vector<T> elems_;
};

}  // namespace ngram_details
}  // namespace contrib
}  // namespace onnxruntime

using namespace onnxruntime::contrib::ngram_details;

namespace std {
template <typename T>
struct hash<OneGram<T>> {
  typedef OneGram<T> argument_type;
  typedef size_t result_type;
  result_type operator()(const argument_type& a) const {
    return std::hash<T>(a.elem_);
  }
};

template <typename T>
struct hash<ManyGram<T>> {
  typedef ManyGram<T> argument_type;
  typedef size_t result_type;
  result_type operator()(const argument_type& a) const {
    if (a.elems_.empty()) return 0;
    auto first = a.elems_.cbegin();
    auto const end = a.elems_.cend();
    result_type hash = std::hash<T>(*first);
    while (++first != end) {
      hash ^= std::hash<T>(*first) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
  }
};
}  // namespace std

namespace onnxruntime {
namespace contrib {

Ngram::Ngram(const OpKernelInfo& info) : OpKernel(info), mode_(kNone) {
  Status status = info.GetAttr("M", &M_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && M_ > 0, "Positive Attr M is required");
  status = info.GetAttr("N", &N_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && N_ >= M_, "Positive M >= N is required");
  status = info.GetAttr("S", &S_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && N_ >= 0, "Non-negative number of skips S is required");

  int64_t all = 0;
  status = info.GetAttr("all", &all);
  ONNXRUNTIME_ENFORCE(status.IsOK(), "Attribute all is required");
  all_ = (all != 0);

  status = info.GetAttrs(std::string("ngram_counts"), ngram_counts_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && !ngram_counts_.empty(), "Non-empty ngram_counts is required");

  status = info.GetAttrs(std::string("ngram_indexes"), ngram_indexes_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && !ngram_indexes_.empty(), "Non-empty ngram_indexes is required");

  status = info.GetAttrs(std::string("weights"), weights_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && !weights_.empty(), "Non-empty weights is required");
  ONNXRUNTIME_ENFORCE(weights_.size() == ngram_indexes_.size(), "weights and indexes must have equal size");

  std::string mode;
  status = info.GetAttr("mode", &mode);
  ONNXRUNTIME_ENFORCE(status.IsOK(), "mode is required");
  if (mode == "TF") {
    mode_ = kTF;
  } else if (mode == "IDF") {
    mode_ = kIDF;
  } else if (mode == "TFIDF") {
    mode_ = kTFIDF;
  }
  ONNXRUNTIME_ENFORCE(mode_ != kNone, "Unrecognized mode");
}

}  // namespace contrib
}  // namespace onnxruntime
