// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <swalter@informatik.uni-freiburg.de>

#include <absl/cleanup/cleanup.h>
#include <absl/strings/str_cat.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "global/Constants.h"
#include "global/RuntimeParameters.h"
#include "index/vocabulary/CompressedVocabulary.h"
#include "index/vocabulary/EmbeddingAccessPlan.h"
#include "index/vocabulary/EmbeddingVocabulary.h"
#include "index/vocabulary/VocabularyInMemory.h"
#include "index/vocabulary/VocabularyInternalExternal.h"
#include "rdfTypes/EmbeddingVector.h"
#include "util/File.h"

namespace {

using namespace ad_utility;

// Wrap an array body (e.g. `[1, 2, 3]`) into a serialized `fp32-vector`
// literal, i.e. `"[1, 2, 3]"^^<...fp32-vector>`.
std::string makeVectorLiteral(std::string_view arrayBody) {
  return absl::StrCat("\"", arrayBody, EMBEDDING_FP32_LITERAL_SUFFIX);
}

// Typed test suite: an `EmbeddingVocabulary` must behave identically regardless
// of its underlying vocabulary implementation.
using EmbeddingVocabularyUnderlyingVocabTypes =
    ::testing::Types<VocabularyInMemory,
                     CompressedVocabulary<VocabularyInternalExternal>>;

template <typename T>
class EmbeddingVocabularyTypedTest : public ::testing::Test {
 public:
  // The vocabulary must round-trip identically regardless of the read backend
  // (`pread`/`mmap`/`in-memory`), which is chosen at `open()` from the
  // `embedding-vector-access` runtime parameter.
  void testEmbeddingVocabulary(ad_utility::EmbeddingVectorAccess mode) {
    setRuntimeParameter<&RuntimeParameters::embeddingVectorAccess_>(mode);
    absl::Cleanup resetMode{[]() {
      setRuntimeParameter<&RuntimeParameters::embeddingVectorAccess_>(
          ad_utility::EmbeddingVectorAccess::Pread);
    }};
    using EV = EmbeddingVocabulary<T>;
    EV vocab;
    const std::string fn = "embvocab-test1.dat";
    auto ww = vocab.makeDiskWriterPtr(fn);
    ww->readableName() = "test";

    // Pairs of (literal, expected decoded vector). All values are exactly
    // representable in `float32` so equality holds. The vectors have different
    // dimensions on purpose, to exercise the variable-length sidecar.
    std::vector<std::pair<std::string, std::vector<float>>> testData{
        {makeVectorLiteral("[1, 2, 3]"), {1.0f, 2.0f, 3.0f}},
        {makeVectorLiteral("[0.5, -0.25]"), {0.5f, -0.25f}},
        {makeVectorLiteral("[10, 20, 30, 40]"), {10.0f, 20.0f, 30.0f, 40.0f}},
        {makeVectorLiteral("[-1.5]"), {-1.5f}},
        {makeVectorLiteral("[0, 0, 0, 0, 0]"), {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
    };
    // The vocabulary requires its words to be inserted in sorted order.
    std::sort(testData.begin(), testData.end());

    for (size_t i = 0; i < testData.size(); i++) {
      auto idx = (*ww)(testData[i].first, true);
      ASSERT_EQ(i, idx);
    }
    ww->finish();

    vocab.open(fn);

    ASSERT_EQ(vocab.size(), testData.size());
    for (size_t i = 0; i < testData.size(); i++) {
      // The original literal string is preserved (first-class RDF term).
      ASSERT_EQ(vocab[i], testData[i].first);
      ASSERT_EQ(vocab.getUnderlyingVocabulary()[i], testData[i].first);
      // The decoded vector round-trips exactly, with the correct dimension.
      auto emb = vocab.getEmbedding(i);
      ASSERT_TRUE(emb.has_value());
      EXPECT_THAT(emb.value(), ::testing::ElementsAreArray(testData[i].second));
    }

    // `lower_bound`/`upper_bound` forward to the underlying vocabulary.
    auto wI = vocab.lower_bound(testData.front().first, ql::ranges::less{});
    ASSERT_EQ(wI.index(), 0);
    wI = vocab.upper_bound("\"~~~", ql::ranges::less{});
    ASSERT_TRUE(wI.isEnd());

    vocab.close();
  }
};

TYPED_TEST_SUITE(EmbeddingVocabularyTypedTest,
                 EmbeddingVocabularyUnderlyingVocabTypes);

// _____________________________________________________________________________
TYPED_TEST(EmbeddingVocabularyTypedTest, TypedTest) {
  for (auto mode : ad_utility::EmbeddingVectorAccess::all()) {
    this->testEmbeddingVocabulary(mode);
  }
}

// A malformed vector literal is a hard build error (strict), unlike the
// `GeoVocabulary`, which stores a placeholder for invalid geometries.
TEST(EmbeddingVocabularyTest, MalformedLiteralIsBuildError) {
  using EV = EmbeddingVocabulary<VocabularyInMemory>;
  EV vocab;
  const std::string fn = "embvocab-test-malformed.dat";
  auto ww = vocab.makeDiskWriterPtr(fn);
  // Not finite.
  EXPECT_ANY_THROW((*ww)(makeVectorLiteral("[1, nan, 3]"), true));
}

TEST(EmbeddingVocabularyTest, MalformedLiteralEmptyArrayIsBuildError) {
  using EV = EmbeddingVocabulary<VocabularyInMemory>;
  EV vocab;
  const std::string fn = "embvocab-test-empty.dat";
  auto ww = vocab.makeDiskWriterPtr(fn);
  EXPECT_ANY_THROW((*ww)(makeVectorLiteral("[]"), true));
}

// The access-pattern policy (`planEmbeddingAccess`) covers the full decision
// tree from `docs/mmap-on-disk-data.md`.
TEST(EmbeddingAccessPlan, Policy) {
  using ad_utility::AccessPattern;
  using ad_utility::EmbeddingAccessPlan;
  using ad_utility::EmbeddingVectorAccess;
  using ad_utility::planEmbeddingAccess;
  constexpr double kDense = 0.5;     // above the default 0.05 threshold
  constexpr double kSparse = 0.001;  // below it

  // InMemory: resident, so neither order nor hint ever matters.
  for (bool sorted : {true, false}) {
    for (double density : {kSparse, kDense, 1.0}) {
      EXPECT_EQ(
          planEmbeddingAccess(EmbeddingVectorAccess::InMemory, sorted, density),
          (EmbeddingAccessPlan{false, AccessPattern::None}));
    }
  }

  // Mmap: hint follows density; reorder only when unsorted AND dense.
  EXPECT_EQ(planEmbeddingAccess(EmbeddingVectorAccess::Mmap, true, kDense),
            (EmbeddingAccessPlan{false, AccessPattern::Sequential}));
  EXPECT_EQ(planEmbeddingAccess(EmbeddingVectorAccess::Mmap, true, kSparse),
            (EmbeddingAccessPlan{false, AccessPattern::Random}));
  EXPECT_EQ(planEmbeddingAccess(EmbeddingVectorAccess::Mmap, false, kDense),
            (EmbeddingAccessPlan{true, AccessPattern::Sequential}));
  EXPECT_EQ(planEmbeddingAccess(EmbeddingVectorAccess::Mmap, false, kSparse),
            (EmbeddingAccessPlan{false, AccessPattern::Random}));

  // Pread: same reorder decision, but never a madvise hint (no mapping).
  EXPECT_EQ(planEmbeddingAccess(EmbeddingVectorAccess::Pread, false, kDense),
            (EmbeddingAccessPlan{true, AccessPattern::None}));
  EXPECT_EQ(planEmbeddingAccess(EmbeddingVectorAccess::Pread, true, kDense),
            (EmbeddingAccessPlan{false, AccessPattern::None}));
  EXPECT_EQ(planEmbeddingAccess(EmbeddingVectorAccess::Pread, false, kSparse),
            (EmbeddingAccessPlan{false, AccessPattern::None}));

  // The `denseThreshold` is the crossover knob.
  EXPECT_EQ(
      planEmbeddingAccess(EmbeddingVectorAccess::Mmap, false, 0.2, 0.1).reorder,
      true);
  EXPECT_EQ(
      planEmbeddingAccess(EmbeddingVectorAccess::Mmap, false, 0.2, 0.3).reorder,
      false);
}

// An index with a mismatching sidecar version is rejected on load.
TEST(EmbeddingVocabularyTest, InvalidEmbeddingVectorVersion) {
  using EV = EmbeddingVocabulary<VocabularyInMemory>;
  const std::string fn = "embvocab-test-version.dat";
  {
    EV vocab;
    auto ww = vocab.makeDiskWriterPtr(fn);
    (*ww)(makeVectorLiteral("[1, 2, 3]"), true);
    ww->finish();
  }
  // Overwrite the version header of the offsets file with a wrong value.
  {
    ad_utility::File offsetsFile{EV::getOffsetsFilename(fn), "r+"};
    uint32_t wrongVersion = ad_utility::EMBEDDING_VECTOR_VERSION + 1;
    offsetsFile.write(&wrongVersion, sizeof(wrongVersion));
    offsetsFile.close();
  }
  EV vocab;
  EXPECT_ANY_THROW(vocab.open(fn));
}

}  // namespace
