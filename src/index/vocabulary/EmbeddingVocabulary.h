// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <swalter@informatik.uni-freiburg.de>

#ifndef QLEVER_SRC_INDEX_VOCABULARY_EMBEDDINGVOCABULARY_H
#define QLEVER_SRC_INDEX_VOCABULARY_EMBEDDINGVOCABULARY_H

#include <absl/strings/str_cat.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "index/vocabulary/VocabularyTypes.h"
#include "rdfTypes/EmbeddingVector.h"
#include "util/File.h"
#include "util/MmapVector.h"
#include "util/Serializer/Serializer.h"

// An `EmbeddingVocabulary` holds embedding-vector literals (datatype
// `qle:*-vector`). Like the `GeoVocabulary`, it stores both the original
// literal strings (in the underlying vocabulary, so they remain first-class,
// exportable RDF terms) and a precomputed, decoded form (here: the raw
// `float32` vectors) in a sidecar, so that query-time access is a cheap binary
// read instead of re-parsing a decimal string.
//
// In contrast to the `GeoVocabulary`, whose per-word `GeometryInfo` records are
// fixed-size, embedding vectors vary in length (different sets have different
// dimensions). The sidecar is therefore variable-length: an in-memory offset
// table (`VocabIndex -> byte offset`) plus an on-disk blob of concatenated
// `float32`. The blob is precision-agnostic raw bytes; the element type is a
// consumer-side concern resolved from the embedding set, not stored here. This
// vocabulary is only suitable for embedding literals and should be used as part
// of a `SplitVocabulary`.
template <typename UnderlyingVocabulary>
class EmbeddingVocabulary {
 private:
  UnderlyingVocabulary literals_;

  // On-disk blob of concatenated raw vector bytes (MVP: `float32`). Used by the
  // `Pread` backend (and to load the `InMemory` buffer at `open()`); for the
  // `Mmap` backend it is closed after the mapping is set up.
  ad_utility::File dataFile_;

  // The read backend chosen at `open()` from the `embedding-vector-access`
  // runtime parameter. Only the storage matching `mode_` is populated.
  ad_utility::EmbeddingVectorAccess mode_;
  // `Mmap`: a read-only memory map of the `.embvec` blob (raw bytes; the float
  // element type is applied consumer-side, so this stays precision-agnostic).
  ad_utility::MmapVectorView<std::byte> mapping_;
  // `InMemory`: the whole blob held in a private buffer.
  std::vector<std::byte> inMemory_;

  // In-memory offset table with `size() + 1` entries. `offsets_[i]` is the
  // start byte of vector `i` in `dataFile_`; `offsets_[size()]` is the total
  // size, so `length(i) = offsets_[i + 1] - offsets_[i]`.
  std::vector<uint64_t> offsets_;

  // Filename suffixes for the two sidecar files.
  static constexpr std::string_view dataSuffix = ".embvec";
  static constexpr std::string_view offsetsSuffix = ".embvec.idx";

  // Header of the offsets file: just the format version.
  static constexpr size_t offsetsHeader =
      sizeof(ad_utility::EMBEDDING_VECTOR_VERSION);

 public:
  EmbeddingVocabulary() = default;

  // Load the decoded `float32` vector for the literal with the given index. The
  // result borrows from the mapping/in-memory buffer for the `Mmap`/`InMemory`
  // backends (zero-copy) and owns a fresh copy for the `Pread` backend.
  std::optional<ad_utility::MaybeOwnedVector> getEmbedding(
      uint64_t index) const;

  // Filenames for the sidecar files, derived from the underlying vocabulary's
  // filename.
  static std::string getDataFilename(std::string_view filename) {
    return absl::StrCat(filename, dataSuffix);
  }
  static std::string getOffsetsFilename(std::string_view filename) {
    return absl::StrCat(filename, offsetsSuffix);
  }

  // Forward all the standard operations to the underlying literal vocabulary.

  // ___________________________________________________________________________
  decltype(auto) operator[](uint64_t id) const { return literals_[id]; }

  // ___________________________________________________________________________
  [[nodiscard]] uint64_t size() const { return literals_.size(); }

  // ___________________________________________________________________________
  template <typename InternalStringType, typename Comparator>
  WordAndIndex lower_bound(const InternalStringType& word,
                           Comparator comparator) const {
    return literals_.lower_bound(word, comparator);
  }

  // ___________________________________________________________________________
  template <typename InternalStringType, typename Comparator>
  WordAndIndex upper_bound(const InternalStringType& word,
                           Comparator comparator) const {
    return literals_.upper_bound(word, comparator);
  }

  // ___________________________________________________________________________
  UnderlyingVocabulary& getUnderlyingVocabulary() { return literals_; }

  // ___________________________________________________________________________
  const UnderlyingVocabulary& getUnderlyingVocabulary() const {
    return literals_;
  }

  // ___________________________________________________________________________
  void open(const std::string& filename);

  // Re-advise the read backend for an upcoming access pattern (a sequential
  // scan vs. random point lookups). Only the `Mmap` backend has a mapping to
  // advise; for `Pread`/`InMemory` this is a no-op. The hint is typically
  // chosen via `planEmbeddingAccess` (see `EmbeddingAccessPlan.h`).
  void setAccessPattern(ad_utility::AccessPattern pattern);

  // Custom word writer that decodes each vector literal and writes its raw
  // `float32` bytes (plus an offset) to the sidecar.
  class WordWriter : public WordWriterBase {
   private:
    std::unique_ptr<typename UnderlyingVocabulary::WordWriter>
        underlyingWordWriter_;
    ad_utility::File dataFile_;
    std::string offsetsFilename_;
    // Accumulated offset table; starts as `{0}` and appends an end offset per
    // vector.
    std::vector<uint64_t> offsets_{0};

   public:
    // Open a writer on the underlying vocabulary and the sidecar data file.
    WordWriter(const UnderlyingVocabulary& vocabulary,
               const std::string& filename);

    // Decode the next vector literal, append its `float32` bytes to the
    // sidecar, and return the literal's new index. Throws on a malformed
    // literal.
    uint64_t operator()(std::string_view word, bool isExternal) override;

    // Finish the underlying writer and flush the offsets file.
    void finishImpl() override;

    ~WordWriter() override;
  };

  // ___________________________________________________________________________
  std::unique_ptr<WordWriter> makeDiskWriterPtr(
      const std::string& filename) const {
    return std::make_unique<WordWriter>(literals_, filename);
  }

  // ___________________________________________________________________________
  void close();

  // Generic serialization support.
  AD_SERIALIZE_FRIEND_FUNCTION(EmbeddingVocabulary) {
    (void)serializer;
    (void)arg;
    throw std::runtime_error(
        "Generic serialization is not implemented for EmbeddingVocabulary.");
  }
};

#endif  // QLEVER_SRC_INDEX_VOCABULARY_EMBEDDINGVOCABULARY_H
