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
  NGramItem(std::vector<T>&& sample) : NgramElementBase(0), itmes_(std::move(sample)) {}
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
class NGramItem<std::string> : public NgramElementBase {
 private:
  std::vector<std::reference_wrapper<const std::string>> items_;

 public:
  template <typename ForwardIter>
  explicit NGramItem(size_t id, ForwardIter first, ForwardIter last) : NgramElementBase(id) {
    std::transform(first, last, std::back_inserter(items_),
                   [](const std::string& s) { return std::cref(s); });
    assert(!items_.empty());
  }
  // Used for constructing a key and query the items
  NGramItem(std::vector<std::reference_wrapper<const std::string>>&& sample) : NgramElementBase(0), items_(std::move(sample)) {
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

template <typename ForwardIter, typename Cont>
void Emplace(ForwardIter first, size_t ngrams, size_t ngram_size, size_t& ngram_id, Cont& c) {
  for (; ngrams > 0; --ngrams) {
    c.emplace(ngram_id, first, first + ngram_size);
    first += ngram_size;
    ++ngram_id;
  }
}

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
  // XXX: Add a verification that ngram_counts match the general item count

  status = info.GetAttrs("ngram_indexes", impl_->ngram_indexes_);
  ONNXRUNTIME_ENFORCE(status.IsOK() && !impl_->ngram_indexes_.empty(), "Non-empty ngram_indexes is required");

  status = info.GetAttrs("weights", impl_->weights_);
  if (status.IsOK()) {
    ONNXRUNTIME_ENFORCE(impl_->weights_.size() == impl_->ngram_indexes_.size(),
                        "weights and indexes must have equal size");
  }

  std::vector<int64_t> pool_int64;
  status = info.GetAttrs("pool_strings", impl_->pool_strings_);
  if (status.IsOK()) {
    ONNXRUNTIME_ENFORCE(!impl_->pool_strings_.empty(), "pool_strings must not be empty if specified");
  } else {
    status = info.GetAttrs("pool_int64", pool_int64);
    ONNXRUNTIME_ENFORCE(status.IsOK() && !pool_int64.empty(), "non-empty pool_int64 is required if pool_strings not provided");
  }

  // Iterator via the pool. Insert 1 item for 1-grams, 2 items for 2-grams, etc.
  const auto total_items = (impl_->pool_strings_.empty()) ? pool_int64.size() : impl_->pool_strings_.size();
  size_t ngram_id = 0;
  size_t ngram_size = 1;
  for (size_t i = 0; i < impl_->ngram_counts_.size(); ++i) {
    size_t start_idx = impl_->ngram_counts_[i];
    size_t end_idx = ((i + 1) < impl_->ngram_counts_.size()) ? impl_->ngram_counts_[i + 1] : total_items;
    ONNXRUNTIME_ENFORCE(end_idx >= start_idx && end_idx < total_items,
                        "n-gram counts out of bounds for ", std::to_string(ngram_size), "-grams");
    auto items = end_idx - start_idx;
    if (items > 0) {
      ONNXRUNTIME_ENFORCE((items % ngram_size == 0),
                          "Number of items must compose whole ", std::to_string(ngram_size), "-grams");
      auto ngrams = items / ngram_size;
      if (impl_->pool_strings_.empty()) {
        Emplace(pool_int64.begin() + start_idx, ngrams, ngram_size, ngram_id, impl_->int_set_);
        ONNXRUNTIME_ENFORCE(impl_->int_set_.size() == impl_->ngram_indexes_.size(),
                            "n-grams in the pool does not match ngram_indexes size");
      } else {
        Emplace(impl_->pool_strings_.begin() + start_idx, ngrams, ngram_size, ngram_id, impl_->str_set_);
        ONNXRUNTIME_ENFORCE(impl_->pool_strings_.size() == impl_->ngram_indexes_.size(),
                            "n-grams in the pool does not match ngram_indexes size");
      }
    }
    ++ngram_size;
  }
}

Ngram::~Ngram() {
}

void Ngram::OutputResult(OpKernelContext* ctx, const std::vector<uint32_t>& frequences) const {
  std::vector<int64_t> output_dims;
  output_dims.push_back(frequences.size());

  TensorShape output_shape(output_dims);
  auto Y = ctx->Output(0, output_shape);
  auto output_data = Y->MutableData<float>();
  const auto& w = impl_->weights_;
  switch (impl_->mode_) {
    case kTF: {
      std::transform(frequences.cbegin(), frequences.cend(), output_data,
                     [](uint32_t f) { return static_cast<float>(f); });
    } break;
    case kIDF: {
      if (!w.empty()) {
        assert(frequences.size() == w.size());
        for (size_t i = 0; i < frequences.size(); ++i) {
          *output_data++ = (frequences[i] > 0) ? w[i] : 0;
        }
      } else {
        for (auto f : frequences) {
          *output_data++ = (f > 0) ? 1.0f : 0;
        }
      }
      break;
      case kTFIDF: {
        if (!w.empty()) {
          assert(frequences.size() == w.size());
          for (size_t i = 0; i < frequences.size(); ++i) {
            *output_data++ = frequences[i] * w[i];
          }
        } else {
          std::transform(frequences.cbegin(), frequences.cend(), output_data,
                         [](uint32_t f) { return static_cast<float>(f); });
        }
      } break;
      case kNone:  // fall-through
      default:
        assert(false);
    }
  }
}  // namespace contrib

// General case for int32_t and int64_t
template <typename T>
void Ngram::ComputeImpl(OpKernelContext* ctx, size_t total_items) const {
  const auto& impl = *impl_;
  auto const set_end = impl.int_set_.cend();
  // Frequency holder, init all to zero
  std::vector<uint32_t> frequencies;
  frequencies.resize(impl.int_set_.size(), 0);

  const auto N = impl.N_;
  const auto S = impl.S_;
  const auto n = (impl.all_) ? impl.M_ : impl.N_;

  auto X = ctx->Input<Tensor>(0);
  auto const input_data = X->template Data<T>();
  auto const end_data = input_data + total_items;
  std::vector<int64_t> sample;
  sample.reserve(N);
  for (auto ni = n; ni <= N; ++ni) {
    // Convert skip into distance between n-gram items
    // by adding 1
    for (auto si = 1; si <= S; ++si) {
      auto ngram_start = input_data;
      while (ngram_start < end_data) {
        // we are interested only in a whole n-gram so if the end
        // does not fit, we stop
        auto const ngram_end = ngram_start + si * ni;
        if (ngram_end >= end_data) {
          break;
        }
        auto ngram_item = ngram_start;
        sample.clear();
        while (ngram_item != ngram_end) {
          sample.push_back(*ngram_item);
          ngram_item += si;
        }
        auto hit = impl.int_set_.find({std::move(sample)});
        if (hit != set_end) {
          // record frequency
          auto ngram_id = hit->id();
          assert(ngram_id < impl.ngram_indexes_.size());
          auto output_idx = impl.ngram_indexes_[ngram_id];
          ONNXRUNTIME_ENFORCE(output_idx >= 0, "ngram_indxes has a negative index");
          assert(output_idx < frequencies.size());
          ++frequencies[output_idx];
        }
        ++ngram_start;
      }
    }
  }
  OutputResult(ctx, frequencies);
}

template <>
void Ngram::ComputeImpl<std::string>(OpKernelContext* ctx, size_t total_items) const {
  const auto& impl = *impl_;
  auto const set_end = impl.str_set_.cend();
  // Frequency holder, init all to zero
  std::vector<uint32_t> frequencies;
  frequencies.resize(impl.int_set_.size(), 0);

  const auto N = impl.N_;
  const auto S = impl.S_;
  const auto n = (impl.all_) ? impl.M_ : impl.N_;

  auto X = ctx->Input<Tensor>(0);
  auto const input_data = X->template Data<std::string>();
  auto const end_data = input_data + total_items;
  std::vector<std::reference_wrapper<const std::string>> sample;
  sample.reserve(N);
  for (auto ni = n; ni <= N; ++ni) {
    // Convert skip into distance between n-gram items
    // by adding 1
    for (auto si = 1; si <= S; ++si) {
      auto ngram_start = input_data;
      while (ngram_start < end_data) {
        // we are interested only in a whole n-gram so if the end
        // does not fit, we stop
        auto const ngram_end = ngram_start + si * ni;
        if (ngram_end >= end_data) {
          break;
        }
        auto ngram_item = ngram_start;
        sample.clear();
        while (ngram_item != ngram_end) {
          sample.push_back(std::cref(*ngram_item));
          ngram_item += si;
        }
        auto hit = impl.str_set_.find({std::move(sample)});
        if (hit != set_end) {
          // record frequency
          auto ngram_id = hit->id();
          assert(ngram_id < impl.ngram_indexes_.size());
          auto output_idx = impl.ngram_indexes_[ngram_id];
          ONNXRUNTIME_ENFORCE(output_idx >= 0, "ngram_indxes has a negative index");
          assert(size_t(output_idx) < frequencies.size());
          ++frequencies[output_idx];
        }
        ++ngram_start;
      }
    }
  }
  OutputResult(ctx, frequencies);
}

Status Ngram::Compute(OpKernelContext* ctx) const {
  Status s;

  auto X = ctx->Input<Tensor>(0);
  auto& input_dims = X->Shape().GetDims();
  size_t total_items = 1;
  // Scalar
  if (input_dims.empty() || (input_dims.size() == 1 && input_dims[0] == 0)) {
    total_items = 1;
  } else {
    for (const auto& dim : input_dims) {
      total_items *= dim;
    }
  }

  return s;
}

}  // namespace contrib
}  // namespace onnxruntime
