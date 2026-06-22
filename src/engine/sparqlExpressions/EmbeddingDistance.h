// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <sebastian.walter98@gmail.com>

#ifndef QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_EMBEDDINGDISTANCE_H
#define QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_EMBEDDINGDISTANCE_H

#include <absl/strings/str_cat.h>

#include <cmath>
#include <optional>
#include <stdexcept>

#include "backports/span.h"
#include "engine/sparqlExpressions/EmbeddingDistanceKernels.h"
#include "global/Id.h"
#include "index/EmbeddingSetRegistry.h"
#include "rdfTypes/EmbeddingVector.h"

namespace sparqlExpression::detail {

// Compute the distance between two operand vectors under the set's metric, or
// throw a hard query error (per `docs/embedding-query-spec.md`) if an operand
// is not an embedding vector or has the wrong dimension. This is the metric
// policy of the `qlef:distance` expression; it lives in a header so the
// embedding benchmark can measure exactly the production function (not a
// reimplementation) — see `benchmark/EmbeddingBenchmark.cpp`.
inline Id computeDistance(
    const EmbeddingSetConfig& config,
    const std::optional<ad_utility::MaybeOwnedVector>& optA,
    const std::optional<ad_utility::MaybeOwnedVector>& optB) {
  if (!optA.has_value() || !optB.has_value()) {
    throw std::runtime_error{
        "qlef:distance: an argument is not an embedding vector literal of the "
        "set's precision (qle:fp32-vector)"};
  }
  ql::span<const float> a = optA->span();
  ql::span<const float> b = optB->span();
  if (a.size() != config.dimension_ || b.size() != config.dimension_) {
    throw std::runtime_error{absl::StrCat(
        "qlef:distance: a vector's dimension (", a.size(), " resp. ", b.size(),
        ") does not match the embedding set's qle:dimension (",
        config.dimension_, ")")};
  }
  double distance;
  switch (config.metric_) {
    case EmbeddingMetric::DotProduct:
      // Negated so that `ORDER BY ASC` still yields the nearest neighbours.
      distance = -dotProduct(a, b);
      break;
    case EmbeddingMetric::SquaredL2:
      distance = squaredL2(a, b);
      break;
    case EmbeddingMetric::L2:
      distance = std::sqrt(squaredL2(a, b));
      break;
    case EmbeddingMetric::Cosine: {
      double dot = dotProduct(a, b);
      if (config.normalized_) {
        // Pre-normalized vectors: ‖a‖ = ‖b‖ = 1, so cosine = dot.
        distance = 1.0 - dot;
      } else {
        double denom =
            std::sqrt(dotProduct(a, a)) * std::sqrt(dotProduct(b, b));
        if (denom == 0.0) {
          throw std::runtime_error{
              "qlef:distance: cosine distance is undefined for a zero-norm "
              "vector"};
        }
        distance = 1.0 - dot / denom;
      }
      break;
    }
  }
  return Id::makeFromDouble(distance);
}

}  // namespace sparqlExpression::detail

#endif  // QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_EMBEDDINGDISTANCE_H
