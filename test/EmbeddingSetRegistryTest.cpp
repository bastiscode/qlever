// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <sebastian.walter98@gmail.com>

#include <gmock/gmock.h>

#include "index/EmbeddingSetRegistry.h"
#include "index/Index.h"
#include "util/GTestHelpers.h"
#include "util/IndexTestHelpers.h"

using namespace ad_utility::testing;
using ::testing::HasSubstr;

namespace {
// The QLever-owned `qle:` prefix that the embedding-set metadata vocabulary
// lives under (deliberately not under `builtin-functions/`, so the declaration
// triples survive index building; see `Constants.h`).
constexpr std::string_view kQlePrefix =
    "@prefix qle: <http://qlever.cs.uni-freiburg.de/embeddings/> .\n"
    "@prefix ex: <http://example.org/> .\n";

// A complete, valid single-set declaration.
std::string oneValidSet() {
  return absl::StrCat(kQlePrefix,
                      "ex:labels a qle:EmbeddingSet ;\n"
                      "  qle:dimension 3 ;\n"
                      "  qle:precision \"fp32\" ;\n"
                      "  qle:metric \"cosine\" ;\n"
                      "  qle:model \"test-model\" ;\n"
                      "  qle:normalized true .\n"
                      "ex:s1 ex:labels \"a vector goes here\" .\n");
}
}  // namespace

// _____________________________________________________________________________
TEST(EmbeddingSetRegistry, ParsesValidDeclaration) {
  auto index = makeTestIndex(oneValidSet());
  const auto& registry = index.getEmbeddingSetRegistry();

  EXPECT_FALSE(registry.empty());
  EXPECT_EQ(registry.numSets(), 1);

  auto getId = makeGetId(index);
  Id setId = getId("<http://example.org/labels>");
  const EmbeddingSetConfig* config = registry.getConfig(setId);
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->dimension_, 3u);
  EXPECT_EQ(config->precision_, "fp32");
  EXPECT_EQ(config->metric_, EmbeddingMetric::Cosine);
  EXPECT_EQ(config->model_, "test-model");
  EXPECT_TRUE(config->normalized_);

  // With exactly one set, the implicit-form lookup returns it.
  EXPECT_EQ(registry.getUniqueSetConfig(), config);

  // A non-set IRI is not a known embedding set.
  EXPECT_EQ(registry.getConfig(getId("<http://example.org/s1>")), nullptr);
}

// _____________________________________________________________________________
TEST(EmbeddingSetRegistry, AllSupportedMetricsAreParsed) {
  auto check = [](std::string_view metric, EmbeddingMetric expected) {
    std::string turtle = absl::StrCat(
        kQlePrefix, "ex:labels a qle:EmbeddingSet ;\n  qle:dimension 3 ;\n",
        "  qle:precision \"fp32\" ;\n  qle:metric \"", metric, "\" ;\n",
        "  qle:model \"m\" ;\n  qle:normalized false .\n"
        "ex:s1 ex:labels \"a vector\" .\n");
    auto index = makeTestIndex(std::move(turtle));
    const auto& registry = index.getEmbeddingSetRegistry();
    auto getId = makeGetId(index);
    const auto* config =
        registry.getConfig(getId("<http://example.org/labels>"));
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->metric_, expected);
  };
  check("cosine", EmbeddingMetric::Cosine);
  check("l2", EmbeddingMetric::L2);
  check("squared-l2", EmbeddingMetric::SquaredL2);
  check("dot-product", EmbeddingMetric::DotProduct);
}

// _____________________________________________________________________________
TEST(EmbeddingSetRegistry, MultipleSetsAreAmbiguousForImplicitForm) {
  std::string turtle = absl::StrCat(
      kQlePrefix,
      "ex:labels a qle:EmbeddingSet ;\n"
      "  qle:dimension 3 ; qle:precision \"fp32\" ; qle:metric \"cosine\" ;\n"
      "  qle:model \"m1\" ; qle:normalized true .\n"
      "ex:images a qle:EmbeddingSet ;\n"
      "  qle:dimension 5 ; qle:precision \"fp32\" ; qle:metric \"cosine\" ;\n"
      "  qle:model \"m2\" ; qle:normalized false .\n");
  auto index = makeTestIndex(std::move(turtle));
  const auto& registry = index.getEmbeddingSetRegistry();

  EXPECT_EQ(registry.numSets(), 2);
  // Ambiguous: the implicit form cannot pick a set.
  EXPECT_EQ(registry.getUniqueSetConfig(), nullptr);

  auto getId = makeGetId(index);
  const auto* labels = registry.getConfig(getId("<http://example.org/labels>"));
  const auto* images = registry.getConfig(getId("<http://example.org/images>"));
  ASSERT_NE(labels, nullptr);
  ASSERT_NE(images, nullptr);
  EXPECT_EQ(labels->dimension_, 3u);
  EXPECT_TRUE(labels->normalized_);
  EXPECT_EQ(images->dimension_, 5u);
  EXPECT_FALSE(images->normalized_);
}

// _____________________________________________________________________________
TEST(EmbeddingSetRegistry, NoEmbeddingSets) {
  auto index = makeTestIndex("<http://example.org/a> <http://example.org/b> "
                             "<http://example.org/c> .");
  const auto& registry = index.getEmbeddingSetRegistry();
  EXPECT_TRUE(registry.empty());
  EXPECT_EQ(registry.numSets(), 0);
  EXPECT_EQ(registry.getUniqueSetConfig(), nullptr);
}

// _____________________________________________________________________________
TEST(EmbeddingSetRegistry, MissingFieldIsLoadError) {
  // The set omits `qle:metric`.
  std::string turtle = absl::StrCat(
      kQlePrefix,
      "ex:labels a qle:EmbeddingSet ;\n"
      "  qle:dimension 3 ; qle:precision \"fp32\" ;\n"
      "  qle:model \"m\" ; qle:normalized true .\n");
  AD_EXPECT_THROW_WITH_MESSAGE(
      makeTestIndex(std::move(turtle)),
      HasSubstr("missing the mandatory field qle:metric"));
}

// _____________________________________________________________________________
TEST(EmbeddingSetRegistry, UnsupportedPrecisionIsLoadError) {
  std::string turtle = absl::StrCat(
      kQlePrefix,
      "ex:labels a qle:EmbeddingSet ;\n"
      "  qle:dimension 3 ; qle:precision \"fp16\" ; qle:metric \"cosine\" ;\n"
      "  qle:model \"m\" ; qle:normalized true .\n");
  AD_EXPECT_THROW_WITH_MESSAGE(makeTestIndex(std::move(turtle)),
                               HasSubstr("unsupported qle:precision"));
}

// _____________________________________________________________________________
TEST(EmbeddingSetRegistry, UnsupportedMetricIsLoadError) {
  std::string turtle = absl::StrCat(
      kQlePrefix,
      "ex:labels a qle:EmbeddingSet ;\n"
      "  qle:dimension 3 ; qle:precision \"fp32\" ; qle:metric \"euclidean\" ;\n"
      "  qle:model \"m\" ; qle:normalized true .\n");
  AD_EXPECT_THROW_WITH_MESSAGE(makeTestIndex(std::move(turtle)),
                               HasSubstr("unsupported qle:metric"));
}

// _____________________________________________________________________________
TEST(EmbeddingSetRegistry, InvalidDimensionIsLoadError) {
  std::string turtle = absl::StrCat(
      kQlePrefix,
      "ex:labels a qle:EmbeddingSet ;\n"
      "  qle:dimension 0 ; qle:precision \"fp32\" ; qle:metric \"cosine\" ;\n"
      "  qle:model \"m\" ; qle:normalized true .\n");
  AD_EXPECT_THROW_WITH_MESSAGE(makeTestIndex(std::move(turtle)),
                               HasSubstr("invalid qle:dimension"));
}
