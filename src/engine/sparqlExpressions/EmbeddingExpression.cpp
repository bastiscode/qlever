// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <sebastian.walter98@gmail.com>

#include "engine/sparqlExpressions/EmbeddingExpression.h"

#include <absl/strings/str_cat.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "backports/algorithm.h"
#include "engine/sparqlExpressions/EmbeddingDistance.h"
#include "engine/sparqlExpressions/LiteralExpression.h"
#include "engine/sparqlExpressions/SparqlExpressionGenerators.h"
#include "engine/sparqlExpressions/SparqlExpressionValueGetters.h"
#include "global/RuntimeParameters.h"
#include "index/EmbeddingSetRegistry.h"
#include "index/vocabulary/EmbeddingAccessPlan.h"
#include "util/Exception.h"
#include "util/HashMap.h"

namespace sparqlExpression {
namespace {
using namespace detail;

using OptVector = std::optional<ad_utility::MaybeOwnedVector>;

// `computeDistance` (the metric policy) lives in `EmbeddingDistance.h`, and the
// raw `dotProduct`/`squaredL2` kernels in `EmbeddingDistanceKernels.h`, so the
// embedding benchmark measures exactly these production functions.

// The maximum number of distinct query vectors the many-vs-(few) fast path will
// cache. A `Variable` column with at most this many distinct Ids is treated as
// the "query" side (its decodes are hoisted); more than this and it is the
// high-cardinality "candidate" side. `n = 1` is the dominant kNN case.
constexpr size_t kMaxDistinctQueries = 32;

// The distinct `ValueId`s of `ids`, or `nullopt` if there are more than `cap`
// of them. The high-cardinality candidate side therefore bails after ~`cap`
// rows; the linear membership test stays cheap because `distinct` never exceeds
// `cap`.
std::optional<std::vector<ValueId>> distinctCapped(ql::span<const ValueId> ids,
                                                   size_t cap) {
  std::vector<ValueId> distinct;
  for (ValueId id : ids) {
    if (ql::ranges::find(distinct, id) == distinct.end()) {
      distinct.push_back(id);
      if (distinct.size() > cap) {
        return std::nullopt;
      }
    }
  }
  return distinct;
}

// Decide whether to fetch a candidate column's embeddings in sorted storage
// order (then scatter the results back to input-row order) so the sidecar is
// touched monotonically — the access-pattern policy of `EmbeddingAccessPlan.h`
// / `docs/mmap-on-disk-data.md`. Only the precomputed-sidecar case (every Id a
// `VocabIndex`) is eligible; the planner then keys off the column's sortedness
// and its density over the touched key span (`k / (maxKey - minKey + 1)`, the
// signal the cost model uses). The persistent `madvise` hint is deliberately
// NOT toggled here: the mapping is shared across concurrent queries, so the
// readahead this relies on is the vocab's open-time default, and reordering is
// what makes a scattered-but-dense access monotonic enough to benefit from it.
bool shouldReorderCandidates(ql::span<const ValueId> candidates) {
  if (candidates.empty()) {
    return false;
  }
  uint64_t minKey = std::numeric_limits<uint64_t>::max();
  uint64_t maxKey = 0;
  uint64_t previous = 0;
  bool sorted = true;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].getDatatype() != Datatype::VocabIndex) {
      // A non-`VocabIndex` candidate goes through the parse fallback, not the
      // sidecar, so the storage-offset reasoning does not apply: leave as-is.
      return false;
    }
    uint64_t key = candidates[i].getVocabIndex().get();
    if (i != 0 && key < previous) {
      sorted = false;
    }
    previous = key;
    minKey = std::min(minKey, key);
    maxKey = std::max(maxKey, key);
  }
  double density = static_cast<double>(candidates.size()) /
                   static_cast<double>(maxKey - minKey + 1);
  auto backend =
      getRuntimeParameter<&RuntimeParameters::embeddingVectorAccess_>();
  return ad_utility::planEmbeddingAccess(backend, sorted, density).reorder;
}

// The `qlef:distance` expression. Holds the two vector child expressions and
// (for the 3-arg form) the constant set IRI; the metric/dimension come from the
// set's config, resolved against the index's `EmbeddingSetRegistry` at
// evaluation time.
class EmbeddingDistanceExpression : public SparqlExpression {
 private:
  std::array<Ptr, 2> children_;        // [0] = a, [1] = b
  std::optional<std::string> setIri_;  // nullopt => the unique declared set

 public:
  EmbeddingDistanceExpression(Ptr a, Ptr b, std::optional<std::string> setIri)
      : children_{std::move(a), std::move(b)}, setIri_{std::move(setIri)} {}

  // __________________________________________________________________________
  ExpressionResult evaluate(EvaluationContext* context) const override {
    // Resolve (and validate) the set once per evaluation; this surfaces
    // unknown-set / ambiguity errors at the start of evaluation, not per row.
    const EmbeddingSetConfig& config = resolveSet(context);

    ExpressionResult resultA = children_[0]->evaluate(context);
    ExpressionResult resultB = children_[1]->evaluate(context);

    auto computeOnOperands = [context, &config](
                                 auto&& opA, auto&& opB) -> ExpressionResult {
      using A = std::decay_t<decltype(opA)>;
      using B = std::decay_t<decltype(opB)>;
      size_t targetSize = getResultSize(*context, opA, opB);
      constexpr bool resultIsConstant =
          isConstantResult<A> && isConstantResult<B>;

      // Many-vs-(few) fast path. When both operands are `Variable` columns and
      // one of them has only a few distinct values, that side is the "query"
      // set and the other is the "candidate" set we stream. This covers the
      // dominant kNN shapes:
      //   * n = 1: a query bound by a single-subject triple
      //     (`<X> <emb> ?q . ?e <emb> ?ev . distance(?q, ?ev)`), whose `?q`
      //     column is constant-valued at runtime — invisible to the
      //     compile-time `resultIsConstant` check above.
      //   * 2 <= n <= kMaxDistinctQueries: a handful of queries crossed with
      //   the
      //     candidates (e.g. `VALUES ?q { … } . ?e <emb> ?ev`).
      // The generic per-row path would re-decode the query operand on every row
      // (a vocab lookup + sidecar read/parse). Instead we decode the few
      // distinct query vectors *once* and reuse them; the candidate side is
      // streamed with the access-pattern policy applied.
      //
      // A literal / constant-folded query operand is NOT a `Variable`, so it
      // does not reach here; it is already decoded once by the constant
      // `resultGenerator` in the generic path below.
      if constexpr (ad_utility::isSimilar<::Variable, A> &&
                    ad_utility::isSimilar<::Variable, B>) {
        ql::span<const ValueId> idsA = getIdsFromVariable(opA, context);
        ql::span<const ValueId> idsB = getIdsFromVariable(opB, context);
        AD_CORRECTNESS_CHECK(idsA.size() == targetSize &&
                             idsB.size() == targetSize);

        // Distinct Ids of each column, capped: `nullopt` == "more than
        // kMaxDistinctQueries distinct" (the high-cardinality candidate side,
        // detected after only ~kMaxDistinctQueries rows).
        auto distinctA = distinctCapped(idsA, kMaxDistinctQueries);
        auto distinctB = distinctCapped(idsB, kMaxDistinctQueries);

        EmbeddingValueGetter getter{};

        // Both columns single-valued: one distance, replicated for every row.
        if (distinctA && distinctA->size() == 1 && distinctB &&
            distinctB->size() == 1) {
          context->cancellationHandle_->throwIfCancelled();
          Id distance = computeDistance(config, getter(idsA.front(), context),
                                        getter(idsB.front(), context));
          VectorWithMemoryLimit<Id> result{context->_allocator};
          result.insert(result.end(), targetSize, distance);
          return result;
        }

        // Compute distances for the candidate column, fetching in sorted
        // storage order when the access-pattern policy says so (then scattering
        // results back to row order). `getQuery(row)` yields the
        // already-decoded query operand for that row; `queryIsFirst` keeps the
        // operand order of `computeDistance` (and the dimension-mismatch
        // message) identical to the textual argument order.
        auto computeColumn =
            [&config, context, &getter](
                ql::span<const ValueId> candidates, auto&& getQuery,
                bool queryIsFirst) -> VectorWithMemoryLimit<Id> {
          auto one = [&](size_t row) {
            const OptVector& query = getQuery(row);
            OptVector candidate = getter(candidates[row], context);
            return queryIsFirst ? computeDistance(config, query, candidate)
                                : computeDistance(config, candidate, query);
          };
          VectorWithMemoryLimit<Id> out{context->_allocator};
          size_t k = candidates.size();
          if (shouldReorderCandidates(candidates)) {
            std::vector<uint32_t> order(k);
            std::iota(order.begin(), order.end(), uint32_t{0});
            ql::ranges::sort(order, [&candidates](uint32_t a, uint32_t b) {
              return candidates[a].getVocabIndex().get() <
                     candidates[b].getVocabIndex().get();
            });
            out.resize(k);
            for (uint32_t i : order) {
              context->cancellationHandle_->throwIfCancelled();
              out[i] = one(i);
            }
          } else {
            out.reserve(k);
            for (size_t i = 0; i < k; ++i) {
              context->cancellationHandle_->throwIfCancelled();
              out.push_back(one(i));
            }
          }
          return out;
        };

        // Pick the low-cardinality "query" side (fewer distinct values; prefer
        // A on a tie). If neither side is low-cardinality, fall through to the
        // generic both-vary path.
        const bool aIsQuery =
            distinctA && (!distinctB || distinctA->size() <= distinctB->size());
        const bool bIsQuery = !aIsQuery && distinctB.has_value();
        if (aIsQuery || bIsQuery) {
          ql::span<const ValueId> queryIds = aIsQuery ? idsA : idsB;
          ql::span<const ValueId> candidateIds = aIsQuery ? idsB : idsA;
          const std::vector<ValueId>& distinct =
              aIsQuery ? *distinctA : *distinctB;
          if (distinct.size() == 1) {
            // n == 1: decode the single shared query once, no per-row lookup.
            OptVector query = getter(queryIds.front(), context);
            return computeColumn(
                candidateIds,
                [&query](size_t) -> const OptVector& { return query; },
                aIsQuery);
          }
          // 2 <= n <= kMaxDistinctQueries: decode the distinct query vectors
          // once into a small cache, then look each row's query up by its Id.
          ad_utility::HashMap<ValueId, OptVector> queryCache;
          queryCache.reserve(distinct.size());
          for (ValueId id : distinct) {
            queryCache.emplace(id, getter(id, context));
          }
          return computeColumn(
              candidateIds,
              [&queryCache, queryIds](size_t row) -> const OptVector& {
                return queryCache.at(queryIds[row]);
              },
              aIsQuery);
        }
      }

      auto genA = valueGetterGenerator(targetSize, context, AD_FWD(opA),
                                       EmbeddingValueGetter{});
      auto genB = valueGetterGenerator(targetSize, context, AD_FWD(opB),
                                       EmbeddingValueGetter{});
      auto distanceFn = [&config](OptVector a, OptVector b) -> Id {
        return computeDistance(config, a, b);
      };
      auto resultGenerator = applyFunction(distanceFn, targetSize,
                                           std::move(genA), std::move(genB));

      VectorWithMemoryLimit<Id> result{context->_allocator};
      result.reserve(targetSize);
      for (auto&& element : resultGenerator) {
        result.push_back(element);
      }
      if constexpr (resultIsConstant) {
        AD_CORRECTNESS_CHECK(result.size() == 1);
        return result[0];
      } else {
        return result;
      }
    };
    return ad_utility::visitWithVariantsAndParameters(
        computeOnOperands, std::move(resultA), std::move(resultB));
  }

  // __________________________________________________________________________
  std::string getCacheKey(const VariableToColumnMap& varColMap) const override {
    return absl::StrCat(
        "EMBEDDING_DISTANCE set=", setIri_.value_or("<unique-set>"),
        " a=", children_[0]->getCacheKey(varColMap),
        " b=", children_[1]->getCacheKey(varColMap));
  }

 private:
  // __________________________________________________________________________
  ql::span<Ptr> childrenImpl() override {
    return {children_.data(), children_.size()};
  }

  // Resolve the embedding set against the index's registry, throwing a hard
  // query error if it cannot be resolved (unknown / no sets / ambiguous).
  const EmbeddingSetConfig& resolveSet(EvaluationContext* context) const {
    const auto& index = context->_qec.getIndex();
    const auto& registry = index.getEmbeddingSetRegistry();
    if (setIri_.has_value()) {
      const EmbeddingSetConfig* config = nullptr;
      VocabIndex idx;
      if (index.getVocab().getId(*setIri_, &idx)) {
        config = registry.getConfig(Id::makeFromVocabIndex(idx));
      }
      if (config == nullptr) {
        throw std::runtime_error{absl::StrCat(
            "qlef:distance: ", *setIri_,
            " is not a declared embedding set (a qle:EmbeddingSet)")};
      }
      return *config;
    }
    if (registry.numSets() == 0) {
      throw std::runtime_error{
          "qlef:distance: the index declares no embedding sets, so the "
          "implicit "
          "two-argument form cannot be used"};
    }
    const EmbeddingSetConfig* config = registry.getUniqueSetConfig();
    if (config == nullptr) {
      throw std::runtime_error{
          "qlef:distance: the index declares more than one embedding set; name "
          "the set explicitly as the first argument"};
    }
    return *config;
  }
};
}  // namespace

// ____________________________________________________________________________
SparqlExpression::Ptr makeEmbeddingDistanceExpression(
    SparqlExpression::Ptr child1, SparqlExpression::Ptr child2,
    std::optional<SparqlExpression::Ptr> child3) {
  if (child3.has_value()) {
    // 3-arg form: child1 = set IRI (must be a constant IRI), child2/child3 =
    // a/b.
    const auto* iriExpression =
        dynamic_cast<const IriExpression*>(child1.get());
    if (iriExpression == nullptr) {
      throw std::runtime_error{
          "qlef:distance: the first argument (the embedding set) must be a "
          "constant IRI"};
    }
    std::string setIri = iriExpression->value().toStringRepresentation();
    return std::make_unique<EmbeddingDistanceExpression>(
        std::move(child2), std::move(child3.value()), std::move(setIri));
  }
  // 2-arg form: child1/child2 = a/b, the unique declared set is used.
  return std::make_unique<EmbeddingDistanceExpression>(
      std::move(child1), std::move(child2), std::nullopt);
}

}  // namespace sparqlExpression
