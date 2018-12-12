// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ngram.h"
#include "onnx/defs/schema.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"

#include <functional>
#include <unordered_set>

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    Ngram,
    1,
    string,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    contrib::Ngram);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    Ngram,
    1,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    contrib::Ngram);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    Ngram,
    1,
    int64_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    contrib::Ngram);

namespace ngram_details {

class NgramElementBase {
  size_t id_;  // id in the pool
 protected:
  NgramElementBase(size_t id) : id_(id) {}
  ~NgramElementBase() = default;

 public:
  size_t id() const { return id_; }
};

template <class T>
class NGramItem : public NgramElementBase {
  std::vector<T> items_;

 public:
  template <typename ForwardIter>
  explicit NGramItem(size_t id, ForwardIter first, ForwardIter last) : NgramElementBase(id),
                                                                       items_(first, last) {
    assert(!items_.empty());
  }
  bool operator==(const NGramItem& o) const {
    return items_ == o.items_;
  }
  size_t hash() const {
    if (items_.empty()) return 0;
    auto first = items_.cbegin();
    auto const end = items_.cend();
    std::hash<T> hf{};
    auto hash = hf(*first);
    while (++first != end) {
      hash ^= hf(*first) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

template <>
class NGramItem<std::string> : NgramElementBase {
 private:
  std::vector<std::reference_wrapper<const std::string>> items_;

 public:
  template <typename ForwardIter>
  explicit NGramItem(size_t id, ForwardIter first, ForwardIter last) : NgramElementBase(id) {
    std::transform(first, last, std::back_inserter(items_),
                   [](const std::string& s) { return std::cref(s); });
    assert(!items_.empty());
  }
  bool operator==(const NGramItem& o) const {
    if (items_.size() == o.items_.size()) {
      return std::equal(items_.cbegin(), items_.cend(),
                        o.items_.cbegin(), o.items_.cend(),
                        std::equal_to<std::string>());
    }
    return false;
  }
  size_t hash() const {
    if (items_.empty()) return 0;
    auto first = items_.cbegin();
    auto const end = items_.cend();
    std::hash<std::string> hf{};
    auto hash = hf(*first);
    while (++first != end) {
      hash ^= hf(*first) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

using IntegerPoolSet = std::unordered_set<NGramItem<int64_t>>;
// Does not own strings, contains references to them. This helps
// to search by string references that point to the current input.
using StringPoolSet = std::unordered_set<NGramItem<std::string>>;

}  // namespace ngram_details
}  // namespace contrib
}  // namespace onnxruntime

using namespace onnxruntime::contrib::ngram_details;

namespace std {
template <typename T>
struct hash<NGramItem<T>> {
  typedef NGramItem<T> argument_type;
  typedef size_t result_type;
  result_type operator()(const argument_type& a) const {
    return a.hash();
  }
};
}  // namespace std

namespace onnxruntime {
namespace contrib {

enum Mode {
  kNone = 0,
  kTF = 1,
  kIDF = 2,
  kTFIDF = 3
};

struct Ngram::Impl {
  Mode mode_ = kNone;
  int64_t N_ = 0;
  int64_t M_ = 0;
  int64_t S_ = 0;
  bool all_ = false;
  std::vector<int64_t> ngram_counts_;
  std::vector<int64_t> ngram_indexes_;
  std::vector<float> weights_;

  std::vector<std::string> pool_strings_;
  StringPoolSet str_set_;
  IntegerPoolSet int_set_;
};

Ngram::Ngram(const OpKernelInfo& info) : OpKernel(info), impl_(new Impl) {
  std::string mode;
  Status status = info.GetAttr("mode", &mode);
  ONNXRUNTIME_ENFORCE(status.IsOK(), "mode is required");
  if (mode == "TF") {
    impl_->mode_ = kTF;
  } else if (mode == "IDF") {
    impl_->mode_ = kIDF;
  } else if (mode == "TFIDF") {
    impl_->mode_ = kTFIDF;
  }
  ONNXRUNTIME_ENFORCE(impl_->mode_ != kNone, "Unrecognized mode");

  status = info.GetAttr("M", &impl_->M_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && impl_->M_ > 0, "Positive Attr M is required");
  status = info.GetAttr("N", &impl_->N_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && impl_->N_ >= impl_->M_, "Positive M >= N is required");
  status = info.GetAttr("S", &impl_->S_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && impl_->N_ >= 0, "Non-negative number of skips S is required");

  int64_t all = 0;
  status = info.GetAttr("all", &all);
  ONNXRUNTIME_ENFORCE(status.IsOK(), "Attribute all is required");
  impl_->all_ = (all != 0);

  status = info.GetAttrs(std::string("ngram_counts"), impl_->ngram_counts_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && !impl_->ngram_counts_.empty(), "Non-empty ngram_counts is required");

  status = info.GetAttrs(std::string("ngram_indexes"), impl_->ngram_indexes_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && !impl_->ngram_indexes_.empty(), "Non-empty ngram_indexes is required");

  status = info.GetAttrs(std::string("weights"), impl_->weights_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && !impl_->weights_.empty(), "Non-empty weights is required");
  ONNXRUNTIME_ENFORCE(impl_->weights_.size() == impl_->ngram_indexes_.size(), "weights and indexes must have equal size");

  status = info.GetAttrs("pool_strings", impl_->pool_strings_);
  if (status.IsOK()) {
    ONNXRUNTIME_ENFORCE(!impl_->pool_strings_.empty(), "pool_strings must not be empty if specified");
  } else {
    std::vector<int64_t> pool_int64;
    status = info.GetAttrs("pool_int64", pool_int64);
    ONNXRUNTIME_ENFORCE(status.IsOK() && !pool_int64.empty(), "non-empty pool_int64 is required if pool_strings not provided");
    // Iterator via the pool. Insert 1 item for 1-grams, 2 items for 2-grams, etc.
    size_t ngram_id = 0;
    size_t ngram_size = 1;
    for (size_t i = 0; i < impl_->ngram_counts_.size(); ++i) {
      size_t start_idx = impl_->ngram_counts_[i];
      size_t end_idx = ((i + 1) < impl_->ngram_counts_.size()) ? impl_->ngram_counts_[i + 1] : pool_int64.size();
      ONNXRUNTIME_ENFORCE(end_idx > start_idx && end_idx < pool_int64.size(),
                          "ngram counts out of bounds for ", std::to_string(ngram_size), "-grams");
      if ((end_idx - start_idx) > 0) {
        ONNXRUNTIME_ENFORCE(((end_idx - start_idx) % ngram_size == 0),
                            "Number of items must compose whole ", std::to_string(ngram_size), "-grams");
        auto ngrams = (end_idx - start_idx) / ngram_size;
        while (start_idx < end_idx) {
        }
      }
      ++ngram_size;
    }

    //impl_->int_set_.
  }
}

Ngram::~Ngram() {
}

}  // namespace contrib
}  // namespace onnxruntime
