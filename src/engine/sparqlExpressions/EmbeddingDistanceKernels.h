// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <sebastian.walter98@gmail.com>

#ifndef QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_EMBEDDINGDISTANCEKERNELS_H
#define QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_EMBEDDINGDISTANCEKERNELS_H

#include <cstddef>

#include "backports/span.h"

namespace sparqlExpression::detail {

// Raw distance kernels shared by the `qlef:distance` expression
// (`EmbeddingExpression.cpp`) and the embedding benchmark
// (`benchmark/EmbeddingBenchmark.cpp`). Kept header-only and free of the
// metric/sign conventions so the benchmark can measure the arithmetic in
// isolation; the metric selection, the negated-dot and `1 - cos` conventions,
// the `qle:normalized` fast path and the strict validation all stay in
// `EmbeddingExpression.cpp`.
//
// `double` accumulation is used for numerical stability. The operands are
// `span`s, so for the `mmap`/`in-memory` sidecar backends they point straight
// into the storage with no per-row copy.
inline double dotProduct(ql::span<const float> a, ql::span<const float> b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return sum;
}

inline double squaredL2(ql::span<const float> a, ql::span<const float> b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += diff * diff;
  }
  return sum;
}

}  // namespace sparqlExpression::detail

#endif  // QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_EMBEDDINGDISTANCEKERNELS_H
