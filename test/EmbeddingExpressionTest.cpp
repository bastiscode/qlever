// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <sebastian.walter98@gmail.com>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>
#include <gmock/gmock.h>

#include <cmath>
#include <string>
#include <vector>

#include "engine/ExportQueryExecutionTrees.h"
#include "engine/QueryPlanner.h"
#include "index/EncodedIriManager.h"
#include "parser/SparqlParser.h"
#include "util/CancellationHandle.h"
#include "util/GTestHelpers.h"
#include "util/IndexTestHelpers.h"
#include "util/Timer.h"
#include "util/http/MediaTypes.h"

using namespace ad_utility::testing;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

namespace {
// The `fp32-vector` datatype IRI (without brackets).
constexpr std::string_view kVecType =
    "http://qlever.cs.uni-freiburg.de/embeddings/fp32-vector";

// SPARQL prefix declarations used by all test queries.
constexpr std::string_view kPrefixes =
    "PREFIX qle: <http://qlever.cs.uni-freiburg.de/embeddings/>\n"
    "PREFIX qlef: <http://qlever.cs.uni-freiburg.de/embeddings/functions/>\n"
    "PREFIX qled: <http://qlever.cs.uni-freiburg.de/embeddings/>\n"
    "PREFIX ex: <http://example.org/>\n";

// Serialize an inline `fp32-vector` literal for an array body like `[1, 0]`.
std::string vec(std::string_view body) {
  return absl::StrCat("\"", body, "\"^^<", kVecType, ">");
}

// A one-set knowledge graph: set `ex:emb` uses the given `metric`/`normalized`,
// with three stored 2-D vectors a=[1,0], b=[0,1], c=[1,1].
std::string makeKg(std::string_view metric, bool normalized) {
  return absl::StrCat(
      "@prefix qle: <http://qlever.cs.uni-freiburg.de/embeddings/> .\n"
      "@prefix ex: <http://example.org/> .\n"
      "ex:emb a qle:EmbeddingSet ; qle:dimension 2 ; qle:precision \"fp32\" ;\n"
      "  qle:metric \"",
      metric, "\" ; qle:model \"m\" ; qle:normalized ",
      normalized ? "true" : "false", " .\n", "ex:a ex:emb ", vec("[1, 0]"),
      " .\n", "ex:b ex:emb ", vec("[0, 1]"), " .\n", "ex:c ex:emb ",
      vec("[1, 1]"), " .\n");
}

// Run a SELECT `query` against the Turtle `kg` and return the result rows (CSV,
// header dropped). Throws (propagating the query error) if execution fails.
std::vector<std::string> runSelect(const std::string& kg,
                                   const std::string& query) {
  TestIndexConfig config{kg};
  auto qec = getQec(std::move(config));
  qec->clearCacheUnpinnedOnly();
  auto handle = std::make_shared<ad_utility::CancellationHandle<>>();
  QueryPlanner qp{qec, handle};
  static EncodedIriManager evM;
  auto pq = SparqlParser::parseQuery(&evM, query, {});
  auto qet = qp.createExecutionTree(pq);
  ad_utility::Timer timer{ad_utility::Timer::Started};
  std::string out;
  for (const auto& block : ExportQueryExecutionTrees::computeResult(
           pq, qet, ad_utility::MediaType::csv, timer, std::move(handle))) {
    out += block;
  }
  std::vector<std::string> rows = absl::StrSplit(out, '\n', absl::SkipEmpty());
  if (!rows.empty()) {
    rows.erase(rows.begin());  // drop the CSV header row
  }
  return rows;
}

// Convenience: the single scalar result of a one-row, one-column SELECT.
double singleDouble(const std::string& kg, const std::string& query) {
  auto rows = runSelect(kg, query);
  EXPECT_EQ(rows.size(), 1u);
  return std::stod(rows.at(0));
}
}  // namespace

// _____________________________________________________________________________
TEST(EmbeddingExpression, KnnOrderingImplicitSet) {
  // Nearest to [1, 0] under cosine: a (same direction) < c (45°) < b (90°).
  auto rows = runSelect(
      makeKg("cosine", false),
      absl::StrCat(kPrefixes,
                   "SELECT ?x { ?x ex:emb ?e . BIND(qlef:distance(?e, ",
                   vec("[1, 0]"), ") AS ?d) } ORDER BY ASC(?d)"));
  EXPECT_THAT(rows,
              ElementsAre(HasSubstr("/a"), HasSubstr("/c"), HasSubstr("/b")));
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, ManyVsOneTwoVariables) {
  // The dominant kNN shape: the query vector is bound by a single-subject
  // triple (`ex:a ex:emb ?q`), so BOTH operands of `qlef:distance` are
  // `Variable` columns and `?q` is constant-valued at runtime. This exercises
  // the many-vs-one fast path (decode the shared query once); the result must
  // match the literal-query form in `KnnOrderingImplicitSet`. Query = a =
  // [1,0], so nearest is a (0) < c (45°) < b (90°).
  std::string kg = makeKg("cosine", false);
  // `constB` branch: distance(?e, ?q) with ?e varying, ?q constant.
  EXPECT_THAT(
      runSelect(kg, absl::StrCat(kPrefixes,
                                 "SELECT ?x { ?x ex:emb ?e . ex:a ex:emb ?q . "
                                 "BIND(qlef:distance(?e, ?q) AS ?d) } "
                                 "ORDER BY ASC(?d)")),
      ElementsAre(HasSubstr("/a"), HasSubstr("/c"), HasSubstr("/b")));
  // `constA` branch: arguments swapped, distance is symmetric for cosine.
  EXPECT_THAT(
      runSelect(kg, absl::StrCat(kPrefixes,
                                 "SELECT ?x { ?x ex:emb ?e . ex:a ex:emb ?q . "
                                 "BIND(qlef:distance(?q, ?e) AS ?d) } "
                                 "ORDER BY ASC(?d)")),
      ElementsAre(HasSubstr("/a"), HasSubstr("/c"), HasSubstr("/b")));
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, ManyVsOneUnsortedCandidates) {
  // Scramble the candidate order with `VALUES` so the candidate column is not
  // in storage order, exercising the access-pattern reorder + scatter path. The
  // result must still be correct: `ORDER BY ASC(?d)` ranks by distance, so a
  // mis-scattered `?d` would corrupt the `?x` order. Query = a = [1,0], so the
  // ranking is a (0) < c (45°) < b (90°), independent of the input row order.
  std::string kg = makeKg("cosine", false);
  EXPECT_THAT(
      runSelect(kg, absl::StrCat(kPrefixes,
                                 "SELECT ?x { VALUES ?x { ex:c ex:b ex:a } "
                                 "?x ex:emb ?e . ex:a ex:emb ?q . "
                                 "BIND(qlef:distance(?e, ?q) AS ?d) } "
                                 "ORDER BY ASC(?d)")),
      ElementsAre(HasSubstr("/a"), HasSubstr("/c"), HasSubstr("/b")));
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, ManyVsFewCachedQueries) {
  // 2 <= n <= kMaxDistinctQueries: VALUES binds three query subjects, crossed
  // with all candidates. Both operands are `Variable` columns with three
  // distinct values, so neither is the n=1 constant case — this exercises the
  // small-n query-decode cache. The cosine ranking per query must be correct:
  //   from a=[1,0]: a<c<b ;  from b=[0,1]: b<c<a ;  from c=[1,1]: c<{a,b}
  // (the a/b tie at distance 1-1/√2 is broken by ?other, so a before b).
  std::string kg = makeKg("cosine", false);
  auto rows = runSelect(
      kg, absl::StrCat(kPrefixes,
                       "SELECT ?other { VALUES ?qs { ex:a ex:b ex:c } "
                       "?qs ex:emb ?q . ?other ex:emb ?e . "
                       "BIND(qlef:distance(?q, ?e) AS ?d) } "
                       "ORDER BY ?qs ASC(?d) ?other"));
  EXPECT_THAT(rows,
              ElementsAre(HasSubstr("/a"), HasSubstr("/c"), HasSubstr("/b"),
                          HasSubstr("/b"), HasSubstr("/c"), HasSubstr("/a"),
                          HasSubstr("/c"), HasSubstr("/a"), HasSubstr("/b")));
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, ManyVsOneBothColumnsConstant) {
  // Two single-subject triples → both operand columns are constant-valued, so a
  // single distance is computed and replicated. cosine([1,0],[0,1]) = 1.
  std::string kg = makeKg("cosine", false);
  EXPECT_NEAR(
      singleDouble(kg,
                   absl::StrCat(kPrefixes,
                                "SELECT ?d { ex:a ex:emb ?qa . ex:b ex:emb "
                                "?qb . BIND(qlef:distance(?qa, ?qb) AS ?d) }")),
      1.0, 1e-5);
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, ManyVsOneStrictErrorsTwoVariables) {
  // The fast path must keep the strict-error semantics: if the (constant) query
  // operand is not an embedding vector, it is still a hard query error. Here
  // the query Variable `?q` is bound to a plain string literal.
  std::string kg =
      absl::StrCat(makeKg("cosine", false), "ex:q ex:label \"hello\" .\n");
  AD_EXPECT_THROW_WITH_MESSAGE(
      runSelect(kg,
                absl::StrCat(kPrefixes,
                             "SELECT ?x { ?x ex:emb ?e . ex:q ex:label ?q . "
                             "BIND(qlef:distance(?e, ?q) AS ?d) }")),
      HasSubstr("not an embedding vector"));
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, MetricValuesCosine) {
  // cosine([1,0],[0,1]) = 1 - 0 = 1; cosine([1,0],[1,1]) = 1 - 1/√2 ≈ 0.2929.
  std::string kg = makeKg("cosine", false);
  EXPECT_NEAR(singleDouble(kg, absl::StrCat(kPrefixes, "SELECT ?d { BIND(",
                                            "qlef:distance(", vec("[1, 0]"),
                                            ", ", vec("[0, 1]"), ") AS ?d) }")),
              1.0, 1e-5);
  EXPECT_NEAR(singleDouble(kg, absl::StrCat(kPrefixes, "SELECT ?d { BIND(",
                                            "qlef:distance(", vec("[1, 0]"),
                                            ", ", vec("[1, 1]"), ") AS ?d) }")),
              1.0 - 1.0 / std::sqrt(2.0), 1e-5);
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, MetricValuesL2AndSquaredL2) {
  // ‖[1,0]-[0,1]‖² = 2, so l2 = √2, squared-l2 = 2.
  EXPECT_NEAR(singleDouble(makeKg("l2", false),
                           absl::StrCat(kPrefixes, "SELECT ?d { BIND(",
                                        "qlef:distance(", vec("[1, 0]"), ", ",
                                        vec("[0, 1]"), ") AS ?d) }")),
              std::sqrt(2.0), 1e-5);
  EXPECT_NEAR(singleDouble(makeKg("squared-l2", false),
                           absl::StrCat(kPrefixes, "SELECT ?d { BIND(",
                                        "qlef:distance(", vec("[1, 0]"), ", ",
                                        vec("[0, 1]"), ") AS ?d) }")),
              2.0, 1e-5);
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, MetricValuesDotProduct) {
  // dot([1,0],[1,1]) = 1, returned negated: -1.
  EXPECT_NEAR(singleDouble(makeKg("dot-product", false),
                           absl::StrCat(kPrefixes, "SELECT ?d { BIND(",
                                        "qlef:distance(", vec("[1, 0]"), ", ",
                                        vec("[1, 1]"), ") AS ?d) }")),
              -1.0, 1e-5);
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, NormalizedCosineSkipsNorms) {
  // With normalized=true, cosine is computed as 1 - dot. For the already-unit
  // vectors [1,0] and [0,1] that is 1 - 0 = 1 (same as the un-normalized path).
  EXPECT_NEAR(singleDouble(makeKg("cosine", true),
                           absl::StrCat(kPrefixes, "SELECT ?d { BIND(",
                                        "qlef:distance(", vec("[1, 0]"), ", ",
                                        vec("[0, 1]"), ") AS ?d) }")),
              1.0, 1e-5);
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, ExplicitSetForm) {
  EXPECT_NEAR(singleDouble(makeKg("l2", false),
                           absl::StrCat(kPrefixes, "SELECT ?d { BIND(",
                                        "qlef:distance(ex:emb, ", vec("[1, 0]"),
                                        ", ", vec("[0, 1]"), ") AS ?d) }")),
              std::sqrt(2.0), 1e-5);
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, StrictErrors) {
  std::string kg = makeKg("cosine", false);
  // Unknown set in the explicit form.
  AD_EXPECT_THROW_WITH_MESSAGE(
      runSelect(kg, absl::StrCat(kPrefixes, "SELECT ?d { BIND(qlef:distance(",
                                 "ex:nope, ", vec("[1, 0]"), ", ",
                                 vec("[0, 1]"), ") AS ?d) }")),
      HasSubstr("not a declared embedding set"));
  // Dimension mismatch (3-D vector against a 2-D set).
  AD_EXPECT_THROW_WITH_MESSAGE(
      runSelect(kg, absl::StrCat(kPrefixes, "SELECT ?d { BIND(qlef:distance(",
                                 vec("[1, 0, 0]"), ", ", vec("[0, 1]"),
                                 ") AS ?d) }")),
      HasSubstr("dimension"));
  // An argument that is not an embedding vector literal.
  AD_EXPECT_THROW_WITH_MESSAGE(
      runSelect(kg, absl::StrCat(kPrefixes,
                                 "SELECT ?d { BIND(qlef:distance(\"hello\", ",
                                 vec("[0, 1]"), ") AS ?d) }")),
      HasSubstr("not an embedding vector"));
}

// _____________________________________________________________________________
TEST(EmbeddingExpression, AmbiguousImplicitSetWithTwoSets) {
  std::string kg = absl::StrCat(
      "@prefix qle: <http://qlever.cs.uni-freiburg.de/embeddings/> .\n"
      "@prefix ex: <http://example.org/> .\n"
      "ex:e1 a qle:EmbeddingSet ; qle:dimension 2 ; qle:precision \"fp32\" ;\n"
      "  qle:metric \"cosine\" ; qle:model \"m\" ; qle:normalized false .\n"
      "ex:e2 a qle:EmbeddingSet ; qle:dimension 2 ; qle:precision \"fp32\" ;\n"
      "  qle:metric \"l2\" ; qle:model \"m\" ; qle:normalized false .\n"
      "ex:a ex:e1 ",
      vec("[1, 0]"), " .\n");
  AD_EXPECT_THROW_WITH_MESSAGE(
      runSelect(kg,
                absl::StrCat(kPrefixes, "SELECT ?d { BIND(qlef:distance(",
                             vec("[1, 0]"), ", ", vec("[0, 1]"), ") AS ?d) }")),
      HasSubstr("more than one embedding set"));
}
