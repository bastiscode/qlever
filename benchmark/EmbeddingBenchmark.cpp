// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <sebastian.walter98@gmail.com>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <fcntl.h>
#include <unistd.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "../benchmark/infrastructure/Benchmark.h"
#include "../benchmark/infrastructure/BenchmarkMeasurementContainer.h"
#include "../benchmark/infrastructure/BenchmarkMetadata.h"
#include "engine/sparqlExpressions/EmbeddingDistance.h"
#include "global/Constants.h"
#include "global/RuntimeParameters.h"
#include "index/EmbeddingSetRegistry.h"
#include "index/vocabulary/EmbeddingAccessPlan.h"
#include "index/vocabulary/EmbeddingVocabulary.h"
#include "index/vocabulary/VocabularyInMemory.h"
#include "rdfTypes/EmbeddingVector.h"
#include "util/ConfigManager/ConfigManager.h"
#include "util/Random.h"

// Benchmarks for the embedding MVP (`qle:fp32-vector` storage + the
// `qlef:distance` function). See `docs/benchmarking.md` for the methodology and
// `docs/embedding-distance-simd.md` / `docs/mmap-on-disk-data.md` for the
// models these numbers should be read against.
//
// Every measured unit is *production* code: `EmbeddingVocabulary::getEmbedding`
// (decode), `computeDistance` (the metric policy, from `EmbeddingDistance.h`),
// and `EmbeddingVocabulary::setAccessPattern` / `planEmbeddingAccess`. Nothing
// here reimplements engine logic, so the numbers reflect what QLever actually
// runs. Three scenarios, each a table with one column per embedding dimension:
//   1. sidecar decode  — `getEmbedding` per read backend
//   (pread/mmap/in-memory).
//   2. distance metric — `computeDistance` per metric, vectors in memory.
//   3. kNN scan        — the 1-vs-many hot loop (`getEmbedding` +
//   `computeDistance`)
//                        per read backend.
//   4. access pattern  — the `planEmbeddingAccess` policy: a cold mmap scan
//                        under the old `MADV_RANDOM`, the new `MADV_NORMAL`
//                        default, and the `MADV_SEQUENTIAL` ideal.

namespace ad_benchmark {
using namespace sparqlExpression::detail;  // computeDistance
using ad_utility::AccessPattern;
using ad_utility::EmbeddingVectorAccess;
using ad_utility::MaybeOwnedVector;
using ad_utility::planEmbeddingAccess;
using OptVector = std::optional<MaybeOwnedVector>;

namespace {
// Embedding dimensions to sweep (typical text/image embedding sizes).
const std::vector<size_t> kDimensions{128, 384, 768, 1536};

// Sink that prevents the compiler from optimizing away an otherwise-unused
// computation (QLever's benchmark framework has no `DoNotOptimize`; see
// `benchmark/Usage.md`). Each measured lambda folds its result into this.
volatile double gSink = 0.0;

// A validated `EmbeddingSetConfig` (as the registry would hold) for the given
// metric/dimension, so the benchmark can call the real `computeDistance`.
EmbeddingSetConfig makeConfig(size_t dim, EmbeddingMetric metric,
                              bool normalized) {
  return EmbeddingSetConfig{dim, "fp32", metric, "m", normalized};
}

// Borrow a `float` span as the `computeDistance` operand type (zero-copy), the
// same shape the value getter hands the expression.
OptVector asOperand(ql::span<const float> vec) {
  return OptVector{MaybeOwnedVector{vec}};
}

// Format a `float32` embedding-vector literal `"[v0, v1, ...]"^^<…fp32-vector>`
// for `dim` values drawn from `gen`.
std::string makeVectorLiteral(size_t dim,
                              ad_utility::RandomDoubleGenerator& gen) {
  std::vector<float> values;
  values.reserve(dim);
  for (size_t i = 0; i < dim; ++i) {
    values.push_back(static_cast<float>(gen()));
  }
  return absl::StrCat("\"[", absl::StrJoin(values, ", "), "]",
                      EMBEDDING_FP32_LITERAL_SUFFIX);
}

// Build an on-disk `EmbeddingVocabulary` of `numVectors` random `dim`-D vectors
// and return its base filename. The three sidecar files
// (`<fn>`, `<fn>.embvec`, `<fn>.embvec.idx`) are left on disk for the caller to
// open under each access mode; `removeVocabularyFiles` cleans them up.
std::string buildVocabulary(size_t dim, size_t numVectors) {
  std::string filename =
      absl::StrCat("embedding-benchmark-d", dim, "-n", numVectors, ".dat");
  EmbeddingVocabulary<VocabularyInMemory> vocab;
  auto writer = vocab.makeDiskWriterPtr(filename);
  writer->readableName() = "benchmark";
  // The literals only need distinct, valid `fp32-vector` strings; the benchmark
  // addresses vectors by index, so insertion order does not matter here.
  ad_utility::RandomDoubleGenerator gen{-1.0, 1.0};
  for (size_t i = 0; i < numVectors; ++i) {
    (*writer)(makeVectorLiteral(dim, gen), true);
  }
  writer->finish();
  return filename;
}

void removeVocabularyFiles(const std::string& filename) {
  for (std::string_view suffix : {"", ".embvec", ".embvec.idx"}) {
    std::error_code ec;
    std::filesystem::remove(absl::StrCat(filename, suffix), ec);
  }
}

// Evict a file's pages from the OS page cache so the next scan is cold.
// `POSIX_FADV_DONTNEED` needs no privileges; the `fsync` first flushes the
// freshly-written (dirty) pages, which `DONTNEED` would otherwise skip. Used so
// the access-pattern comparison is measured cold, where the hint actually
// bites.
void dropFileCache(const std::string& filename) {
  int fd = ::open(filename.c_str(), O_RDONLY);
  if (fd >= 0) {
    ::fsync(fd);
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
  }
}
}  // namespace

class EmbeddingBenchmark : public BenchmarkInterface {
  // Number of stored candidate vectors streamed per scan. Configurable so the
  // scan can be made long enough for a stable single-shot measurement.
  int numVectors_ = 50'000;

 public:
  EmbeddingBenchmark() {
    getConfigManager().addOption("num-vectors",
                                 "Number of stored embedding vectors that each "
                                 "scan streams over.",
                                 &numVectors_, 50'000);
  }

  std::string name() const final { return "Embedding storage + qlef:distance"; }

  BenchmarkResults runAllBenchmarks() final {
    BenchmarkResults results{};
    const size_t numVectors = static_cast<size_t>(numVectors_);
    getGeneralMetadata().addKeyValuePair("num-vectors", numVectors_);
    getGeneralMetadata().addKeyValuePair("metric-of-times", "seconds per scan");

    // Column headers: the row-name column plus one column per dimension.
    std::vector<std::string> dimColumns{"backend / metric"};
    for (size_t dim : kDimensions) {
      dimColumns.push_back(absl::StrCat("d=", dim));
    }

    benchmarkSidecarDecode(results, numVectors, dimColumns);
    benchmarkDistanceMetrics(results, numVectors, dimColumns);
    benchmarkKnnScanEndToEnd(results, numVectors, dimColumns);
    benchmarkAccessPatternPolicy(results, numVectors, dimColumns);
    return results;
  }

 private:
  // Scenario 1: cost of decoding every stored vector once via
  // `EmbeddingVocabulary::getEmbedding`, per read backend. This is the
  // candidate-streaming cost of a kNN scan with the arithmetic removed, and is
  // where the `mmap`/`in-memory` zero-copy backends should beat `pread`.
  void benchmarkSidecarDecode(BenchmarkResults& results, size_t numVectors,
                              const std::vector<std::string>& dimColumns) {
    auto& table = results.addTable(
        absl::StrCat("1. sidecar decode (getEmbedding), ", numVectors,
                     " vectors decoded, per access mode"),
        {"pread", "mmap", "in-memory"}, dimColumns);

    for (size_t col = 0; col < kDimensions.size(); ++col) {
      size_t dim = kDimensions[col];
      std::string filename = buildVocabulary(dim, numVectors);

      size_t row = 0;
      for (auto mode : EmbeddingVectorAccess::all()) {
        setRuntimeParameter<&RuntimeParameters::embeddingVectorAccess_>(mode);
        EmbeddingVocabulary<VocabularyInMemory> vocab;
        vocab.open(filename);
        table.addMeasurement(row, col + 1, [&vocab, numVectors]() {
          double acc = 0.0;
          for (size_t i = 0; i < numVectors; ++i) {
            auto embedding = vocab.getEmbedding(i);
            acc += embedding->span().front();
          }
          gSink = acc;
        });
        vocab.close();
        ++row;
      }
      setRuntimeParameter<&RuntimeParameters::embeddingVectorAccess_>(
          EmbeddingVectorAccess::Pread);
      removeVocabularyFiles(filename);
    }
  }

  // Scenario 2: cost of the production `computeDistance` per metric, over
  // vectors already in a contiguous in-memory buffer (decode excluded). One
  // held query vs `numVectors` candidates — the dominant 1-vs-many shape. This
  // isolates the arithmetic a SIMD kernel would target, and shows that
  // un-normalized cosine costs ~3x a single dot product (it needs both
  // self-norms) while the other metrics are one pass — i.e. `qle:normalized
  // true` is the configuration to prefer.
  void benchmarkDistanceMetrics(BenchmarkResults& results, size_t numVectors,
                                const std::vector<std::string>& dimColumns) {
    auto& table = results.addTable(
        absl::StrCat("2. computeDistance per metric (1 query x ", numVectors,
                     " candidates, in-memory, no decode)"),
        {"dot-product", "squared-l2", "l2", "cosine (normalized)",
         "cosine (un-normalized)"},
        dimColumns);

    for (size_t col = 0; col < kDimensions.size(); ++col) {
      size_t dim = kDimensions[col];
      std::vector<float> data(numVectors * dim);
      ad_utility::RandomDoubleGenerator gen{-1.0, 1.0};
      for (float& value : data) {
        value = static_cast<float>(gen());
      }
      auto vectorAt = [&data, dim](size_t i) {
        return ql::span<const float>{data.data() + i * dim, dim};
      };

      auto sweep = [&](size_t row, EmbeddingMetric metric, bool normalized) {
        EmbeddingSetConfig config = makeConfig(dim, metric, normalized);
        OptVector query = asOperand(vectorAt(0));  // held once
        table.addMeasurement(row, col + 1, [&, config]() {
          double acc = 0.0;
          for (size_t i = 0; i < numVectors; ++i) {
            acc += computeDistance(config, query, asOperand(vectorAt(i)))
                       .getDouble();
          }
          gSink = acc;
        });
      };
      sweep(0, EmbeddingMetric::DotProduct, false);
      sweep(1, EmbeddingMetric::SquaredL2, false);
      sweep(2, EmbeddingMetric::L2, false);
      sweep(3, EmbeddingMetric::Cosine, /*normalized=*/true);
      sweep(4, EmbeddingMetric::Cosine, /*normalized=*/false);
    }
  }

  // Scenario 3: the realistic 1-vs-many hot loop, exactly what the
  // `qlef:distance` fast path does per chunk: decode the query once
  // (`getEmbedding`), then for each candidate decode it and call the real
  // `computeDistance` (un-normalized cosine here). Combines scenarios 1 and 2
  // and is reported per read backend.
  void benchmarkKnnScanEndToEnd(BenchmarkResults& results, size_t numVectors,
                                const std::vector<std::string>& dimColumns) {
    auto& table = results.addTable(
        absl::StrCat("3. kNN scan end-to-end (1 query x ", numVectors,
                     " candidates, getEmbedding + computeDistance) per access "
                     "mode"),
        {"pread", "mmap", "in-memory"}, dimColumns);

    for (size_t col = 0; col < kDimensions.size(); ++col) {
      size_t dim = kDimensions[col];
      EmbeddingSetConfig config =
          makeConfig(dim, EmbeddingMetric::Cosine, /*normalized=*/false);
      std::string filename = buildVocabulary(dim, numVectors);

      size_t row = 0;
      for (auto mode : EmbeddingVectorAccess::all()) {
        setRuntimeParameter<&RuntimeParameters::embeddingVectorAccess_>(mode);
        EmbeddingVocabulary<VocabularyInMemory> vocab;
        vocab.open(filename);
        table.addMeasurement(row, col + 1, [&vocab, numVectors, config]() {
          // Decode the held query once (the many-vs-one optimization).
          OptVector query = vocab.getEmbedding(0);
          double acc = 0.0;
          for (size_t i = 0; i < numVectors; ++i) {
            acc += computeDistance(config, query, vocab.getEmbedding(i))
                       .getDouble();
          }
          gSink = acc;
        });
        vocab.close();
        ++row;
      }
      setRuntimeParameter<&RuntimeParameters::embeddingVectorAccess_>(
          EmbeddingVectorAccess::Pread);
      removeVocabularyFiles(filename);
    }
  }

  // Scenario 4: the access-pattern policy in action on the mmap backend. A full
  // sequential scan (sorted + dense → `planEmbeddingAccess` advises sequential)
  // is measured on a COLD cache (the file is evicted before each run) under the
  // old fixed `MADV_RANDOM`, the new `MADV_NORMAL` open-time default, and the
  // `MADV_SEQUENTIAL` ideal. This isolates the readahead effect — the cause of
  // the pathological large-set mmap time. The gap grows with the sidecar size
  // relative to RAM; with everything warm the rows collapse to RAM speed.
  void benchmarkAccessPatternPolicy(
      BenchmarkResults& results, size_t numVectors,
      const std::vector<std::string>& dimColumns) {
    auto& table = results.addTable(
        absl::StrCat("4. mmap full scan (", numVectors,
                     " vectors, cold cache) by madvise hint"),
        {"MADV_RANDOM (was default)", "MADV_NORMAL (is default)",
         "MADV_SEQUENTIAL (planner ideal)"},
        dimColumns);

    setRuntimeParameter<&RuntimeParameters::embeddingVectorAccess_>(
        EmbeddingVectorAccess::Mmap);
    // A full in-order scan is sorted and dense, so the planner's advisory hint
    // is sequential (the single-threaded ideal). Production instead opens with
    // `None`/`MADV_NORMAL` (the middle row) and relies on readahead + local
    // reordering, because the mapping is shared across queries.
    AccessPattern planned =
        planEmbeddingAccess(EmbeddingVectorAccess::Mmap, /*indicesSorted=*/true,
                            /*density=*/1.0)
            .adviseHint;

    for (size_t col = 0; col < kDimensions.size(); ++col) {
      size_t dim = kDimensions[col];
      std::string filename = buildVocabulary(dim, numVectors);
      std::string dataFile =
          EmbeddingVocabulary<VocabularyInMemory>::getDataFilename(filename);

      auto coldScan = [&](size_t row, AccessPattern hint) {
        dropFileCache(dataFile);  // force the scan to fault from disk
        EmbeddingVocabulary<VocabularyInMemory> vocab;
        vocab.open(filename);
        vocab.setAccessPattern(hint);
        table.addMeasurement(row, col + 1, [&vocab, numVectors]() {
          double acc = 0.0;
          for (size_t i = 0; i < numVectors; ++i) {
            // Read the WHOLE vector so every page of the sidecar is touched.
            for (float value : vocab.getEmbedding(i)->span()) {
              acc += value;
            }
          }
          gSink = acc;
        });
        vocab.close();
      };
      coldScan(0, AccessPattern::Random);
      coldScan(1, AccessPattern::None);  // MADV_NORMAL — the production default
      coldScan(2, planned);              // MADV_SEQUENTIAL
      removeVocabularyFiles(filename);
    }
    setRuntimeParameter<&RuntimeParameters::embeddingVectorAccess_>(
        EmbeddingVectorAccess::Pread);
  }
};

AD_REGISTER_BENCHMARK(EmbeddingBenchmark);
}  // namespace ad_benchmark
