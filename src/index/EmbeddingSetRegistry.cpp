// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <sebastian.walter98@gmail.com>

#include "index/EmbeddingSetRegistry.h"

#include "global/Constants.h"
#include "util/Exception.h"

// _____________________________________________________________________________
std::optional<EmbeddingMetric> embeddingMetricFromString(
    std::string_view metric) {
  if (metric == EMBEDDING_METRIC_COSINE) {
    return EmbeddingMetric::Cosine;
  }
  if (metric == EMBEDDING_METRIC_L2) {
    return EmbeddingMetric::L2;
  }
  if (metric == EMBEDDING_METRIC_SQUARED_L2) {
    return EmbeddingMetric::SquaredL2;
  }
  if (metric == EMBEDDING_METRIC_DOT_PRODUCT) {
    return EmbeddingMetric::DotProduct;
  }
  return std::nullopt;
}

// _____________________________________________________________________________
void EmbeddingSetRegistry::addSet(Id setIri, EmbeddingSetConfig config) {
  auto [it, inserted] = sets_.try_emplace(setIri, std::move(config));
  AD_CORRECTNESS_CHECK(inserted,
                       "An embedding set was registered more than once");
}

// _____________________________________________________________________________
const EmbeddingSetConfig* EmbeddingSetRegistry::getConfig(Id setIri) const {
  auto it = sets_.find(setIri);
  return it == sets_.end() ? nullptr : &it->second;
}

// _____________________________________________________________________________
const EmbeddingSetConfig* EmbeddingSetRegistry::getUniqueSetConfig() const {
  if (sets_.size() != 1) {
    return nullptr;
  }
  return &sets_.begin()->second;
}
