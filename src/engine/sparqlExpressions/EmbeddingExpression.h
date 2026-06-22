// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <sebastian.walter98@gmail.com>

#ifndef QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_EMBEDDINGEXPRESSION_H
#define QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_EMBEDDINGEXPRESSION_H

#include <optional>

#include "engine/sparqlExpressions/SparqlExpression.h"

namespace sparqlExpression {

// Factory for the `qlef:distance` SPARQL expression function (see
// `docs/embedding-query-spec.md`). Two forms, distinguished by whether `child3`
// is present:
//
//   qlef:distance(a, b)       -> `child3 == nullopt`; the embedding set is the
//                                unique declared set (error if 0 or >1 sets).
//   qlef:distance(set, a, b)  -> `child1` is the set IRI, `child2`/`child3` the
//                                two vectors. `child1` must be a constant IRI
//                                (else a parse-time error is thrown here).
//
// The result is a cosine / L2 / squared-L2 / dot-product *distance* (smaller =
// closer) per the set's `qle:metric`. Set resolution and all per-row validation
// happen at evaluation time against the index's `EmbeddingSetRegistry`.
SparqlExpression::Ptr makeEmbeddingDistanceExpression(
    SparqlExpression::Ptr child1, SparqlExpression::Ptr child2,
    std::optional<SparqlExpression::Ptr> child3);

}  // namespace sparqlExpression

#endif  // QLEVER_SRC_ENGINE_SPARQLEXPRESSIONS_EMBEDDINGEXPRESSION_H
