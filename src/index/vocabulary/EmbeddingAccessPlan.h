// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <sebastian.walter98@gmail.com>

#ifndef QLEVER_SRC_INDEX_VOCABULARY_EMBEDDINGACCESSPLAN_H
#define QLEVER_SRC_INDEX_VOCABULARY_EMBEDDINGACCESSPLAN_H

#include "rdfTypes/EmbeddingVector.h"
#include "util/MmapVector.h"

namespace ad_utility {

// How to fetch a batch of embedding vectors from the `.embvec` sidecar. The
// plan is derived from the read backend plus two cheap signals measured on the
// set of candidate indices the caller already holds (e.g. the `qlef:distance`
// fast path holds them as an `IdTable` column): whether the indices are already
// sorted by storage offset, and how dense the access is (`k / N`). The
// rationale and cost model are in `docs/mmap-on-disk-data.md`.
struct EmbeddingAccessPlan {
  // Sort the indices (and scatter the results back to their input rows) before
  // fetching, so the sidecar is touched monotonically. Only worth its
  // materialize + gather/scatter cost when the order is not already usable.
  bool reorder;
  // `madvise` hint to apply to the mmap mapping before the scan. `None` for the
  // `Pread`/`InMemory` backends, which have no mapping to advise.
  AccessPattern adviseHint;

  bool operator==(const EmbeddingAccessPlan&) const = default;
};

// Decide the access plan. `density` is the fraction of the set's vectors that
// will be fetched (`k / N`); `denseThreshold` is the crossover above which a
// monotonic sequential sweep is expected to beat per-vector random reads (a
// heuristic — see the cost model in the docs).
//
// The decision tree (under the backend gate):
//   InMemory                  -> {reorder=false, hint=None}   (resident;
//   nothing to tune) sorted                    -> {reorder=false,
//   hint=dense?SEQUENTIAL:RANDOM} unsorted && dense         -> {reorder=true,
//   hint=SEQUENTIAL} unsorted && sparse        -> {reorder=false, hint=RANDOM}
// where `hint` is only `SEQUENTIAL`/`RANDOM` for the `Mmap` backend (`Pread`
// has no mapping, so its `hint` stays `None`, but it still benefits from
// `reorder`, which turns random seeks into sequential ones).
inline EmbeddingAccessPlan planEmbeddingAccess(EmbeddingVectorAccess backend,
                                               bool indicesSorted,
                                               double density,
                                               double denseThreshold = 0.05) {
  using Enum = EmbeddingVectorAccess::Enum;
  // The operator chose `InMemory` => the whole set is resident, so neither the
  // access order nor any `madvise` hint can change the (RAM-speed) cost.
  if (backend.value() == Enum::InMemory) {
    return {false, AccessPattern::None};
  }
  const bool dense = density >= denseThreshold;
  // Reorder only when the order is not already monotonic AND we will touch
  // enough that a sequential sweep beats random per-vector reads; a sparse
  // scattered subset is left in place (sorting wouldn't pay for itself).
  const bool reorder = !indicesSorted && dense;
  // `madvise` is only meaningful for the mmap mapping. A dense scan wants
  // readahead on; a sparse one wants it off so it doesn't pull the gaps.
  AccessPattern hint = AccessPattern::None;
  if (backend.value() == Enum::Mmap) {
    hint = dense ? AccessPattern::Sequential : AccessPattern::Random;
  }
  return {reorder, hint};
}

}  // namespace ad_utility

#endif  // QLEVER_SRC_INDEX_VOCABULARY_EMBEDDINGACCESSPLAN_H
