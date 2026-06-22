// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <sebastian.walter98@gmail.com>

#ifndef QLEVER_SRC_INDEX_EMBEDDINGSETREGISTRY_H
#define QLEVER_SRC_INDEX_EMBEDDINGSETREGISTRY_H

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "global/Id.h"
#include "util/HashMap.h"

// The similarity metric of an embedding set. Each is computed as an exact
// *distance* by `qlef:distance` (smaller = closer); see
// `docs/embedding-query-spec.md`.
enum class EmbeddingMetric {
  Cosine,      // `1 − dot/(‖a‖·‖b‖)`  (or `1 − dot` when the set is normalized)
  L2,          // true Euclidean distance `√Σ(aᵢ−bᵢ)²`
  SquaredL2,   // `Σ(aᵢ−bᵢ)²` (no `√`; same kNN ranking as L2)
  DotProduct,  // `−dot` (negated so `ORDER BY ASC` still yields nearest)
};

// Map a `qle:metric` string to its `EmbeddingMetric`, or `std::nullopt` if it is
// not one of the (MVP-)supported metrics. The single place that knows the
// metric vocabulary (used by the load-time validation).
std::optional<EmbeddingMetric> embeddingMetricFromString(std::string_view metric);

// The validated metadata of a single embedding set (see
// `docs/embedding-storage-spec.md` §3). All fields are mandatory; the
// `EmbeddingSetRegistry` only ever holds configs that passed strict validation
// at index load time.
struct EmbeddingSetConfig {
  // The vector dimension every member of the set must have.
  uint64_t dimension_;
  // The element precision (MVP: always `"fp32"`).
  std::string precision_;
  // The similarity metric (already validated/parsed from `qle:metric`).
  EmbeddingMetric metric_;
  // Provenance: which model produced the stored vectors.
  std::string model_;
  // Whether the vectors are pre-normalized (lets cosine skip norm division).
  bool normalized_;

  bool operator==(const EmbeddingSetConfig&) const = default;
};

// An in-memory map `embedding-set IRI → EmbeddingSetConfig`, built at index
// **load** time (not persisted) by scanning the `qle:EmbeddingSet` declarations
// from the loaded permutations. The query-side `qle:distance` function resolves
// a set IRI to its config through this registry (see
// `docs/embedding-query-spec.md`). The set IRI is keyed by its `VocabIndex`
// `Id` so the expression can look it up cheaply from a constant operand.
class EmbeddingSetRegistry {
 private:
  ad_utility::HashMap<Id, EmbeddingSetConfig> sets_;

 public:
  // Register a validated set. The `setIri` is the `Id` of the user-chosen
  // predicate that doubles as the set IRI. Throws if the same set is added
  // twice (a builder invariant, not user-facing).
  void addSet(Id setIri, EmbeddingSetConfig config);

  // Look up the config for a set IRI `Id`, or `nullptr` if it is not a declared
  // embedding set. Used for the explicit `qle:distance(set, a, b)` form.
  const EmbeddingSetConfig* getConfig(Id setIri) const;

  // For the implicit `qle:distance(a, b)` form: the single declared set's
  // config iff exactly one set exists, else `nullptr` (the caller turns 0 vs.
  // >1 into the appropriate "no embeddings" / "ambiguous" error via `numSets`).
  const EmbeddingSetConfig* getUniqueSetConfig() const;

  // The number of declared embedding sets (drives the implicit-form error
  // messages: 0 → no embeddings, >1 → ambiguous).
  size_t numSets() const { return sets_.size(); }

  bool empty() const { return sets_.empty(); }
};

#endif  // QLEVER_SRC_INDEX_EMBEDDINGSETREGISTRY_H
