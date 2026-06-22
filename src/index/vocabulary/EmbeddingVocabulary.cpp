// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <swalter@informatik.uni-freiburg.de>

#include "index/vocabulary/EmbeddingVocabulary.h"

#include <stdexcept>

#include "global/RuntimeParameters.h"
#include "index/vocabulary/CompressedVocabulary.h"
#include "index/vocabulary/VocabularyInMemory.h"
#include "index/vocabulary/VocabularyInternalExternal.h"
#include "rdfTypes/EmbeddingVector.h"
#include "util/Exception.h"
#include "util/ExceptionHandling.h"
#include "util/MmapVector.h"

// ____________________________________________________________________________
template <typename V>
void EmbeddingVocabulary<V>::open(const std::string& filename) {
  literals_.open(filename);
  dataFile_.open(getDataFilename(filename).c_str(), "r");

  // Read and check the header of the offsets file, then load the whole offset
  // table into memory.
  ad_utility::File offsetsFile{getOffsetsFilename(filename), "r"};
  std::decay_t<decltype(ad_utility::EMBEDDING_VECTOR_VERSION)> versionOfFile =
      0;
  offsetsFile.read(&versionOfFile, offsetsHeader, 0);
  if (versionOfFile != ad_utility::EMBEDDING_VECTOR_VERSION) {
    throw std::runtime_error(absl::StrCat(
        "The embedding vector version of ", getOffsetsFilename(filename),
        " is ", versionOfFile, ", which is incompatible with version ",
        ad_utility::EMBEDDING_VECTOR_VERSION,
        " as required by this version of QLever. Please rebuild your index."));
  }

  size_t numWords = literals_.size();
  offsets_.resize(numWords + 1);
  offsetsFile.read(offsets_.data(), (numWords + 1) * sizeof(uint64_t),
                   offsetsHeader);
  offsetsFile.close();

  // Choose the read backend once, at load time (see `EmbeddingVectorAccess`).
  mode_ = getRuntimeParameter<&RuntimeParameters::embeddingVectorAccess_>();
  uint64_t numDataBytes = offsets_.empty() ? 0 : offsets_.back();
  switch (mode_.value()) {
    case ad_utility::EmbeddingVectorAccess::Enum::Pread:
      // Keep `dataFile_` open for positioned reads in `getEmbedding`.
      break;
    case ad_utility::EmbeddingVectorAccess::Enum::Mmap:
      // Memory-map the data blob (precision-agnostic, so map raw bytes). The
      // trailing `MmapVectorMetaData` written by the `WordWriter` lets
      // `MmapVectorView` open the `.embvec` sidecar directly.
      //
      // Open with `None` (`MADV_NORMAL`), i.e. the kernel's default readahead —
      // NOT `Random`, which disables readahead and makes a cold full scan
      // degenerate into one synchronous page read per vector (the pathological
      // mmap time observed for large sets; see `docs/mmap-on-disk-data.md`).
      // The persistent hint is a shared, query-independent default on purpose:
      // it must not be toggled per query because the mapping is shared across
      // concurrent queries. A query that knows its access is scattered makes it
      // *monotonic* locally instead (the `reorder` step in
      // `EmbeddingExpression.cpp`), which lets this default readahead work.
      mapping_.open(getDataFilename(filename), ad_utility::AccessPattern::None);
      dataFile_.close();
      break;
    case ad_utility::EmbeddingVectorAccess::Enum::InMemory:
      // Load the whole blob into a private buffer once, then drop the file.
      inMemory_.resize(numDataBytes);
      dataFile_.read(inMemory_.data(), numDataBytes, 0);
      dataFile_.close();
      break;
  }
}

// ____________________________________________________________________________
template <typename V>
void EmbeddingVocabulary<V>::setAccessPattern(
    ad_utility::AccessPattern pattern) {
  // Only the memory-mapped backend has a live mapping to re-advise.
  if (mode_.value() == ad_utility::EmbeddingVectorAccess::Enum::Mmap) {
    mapping_.setAccessPattern(pattern);
  }
}

// ____________________________________________________________________________
template <typename V>
void EmbeddingVocabulary<V>::close() {
  literals_.close();
  dataFile_.close();
  mapping_.close();
  inMemory_.clear();
  inMemory_.shrink_to_fit();
  offsets_.clear();
}

// ____________________________________________________________________________
template <typename V>
EmbeddingVocabulary<V>::WordWriter::WordWriter(const V& vocabulary,
                                               const std::string& filename)
    : underlyingWordWriter_{vocabulary.makeDiskWriterPtr(filename)},
      dataFile_{getDataFilename(filename), "w"},
      offsetsFilename_{getOffsetsFilename(filename)} {}

// ____________________________________________________________________________
template <typename V>
uint64_t EmbeddingVocabulary<V>::WordWriter::operator()(std::string_view word,
                                                        bool isExternal) {
  // Store the literal string in the underlying vocabulary.
  uint64_t index = (*underlyingWordWriter_)(word, isExternal);

  // Decode the vector and append its raw `float32` bytes to the data file. A
  // malformed literal is a hard build error (strict).
  auto vec = ad_utility::parseFp32VectorLiteral(word);
  if (!vec.has_value()) {
    throw std::runtime_error(absl::StrCat(
        "Malformed embedding vector literal could not be parsed as a finite "
        "fp32 array: ",
        word));
  }
  dataFile_.write(vec->data(), vec->size() * sizeof(float));
  offsets_.push_back(offsets_.back() + vec->size() * sizeof(float));

  return index;
}

// ____________________________________________________________________________
template <typename V>
void EmbeddingVocabulary<V>::WordWriter::finishImpl() {
  underlyingWordWriter_->finish();

  // Append the `MmapVectorMetaData` trailer to the data blob so it can be
  // opened directly by `MmapVectorView<std::byte>` at query time. The element
  // type for the view is `std::byte`, so the element count, capacity, and byte
  // size all equal the total number of data bytes (`offsets_.back()`).
  uint64_t numDataBytes = offsets_.back();
  ad_utility::MmapVectorMetaData{numDataBytes, numDataBytes, numDataBytes}
      .writeToFile(dataFile_);
  dataFile_.close();

  // Write the offsets file: header (version) followed by the offset table.
  ad_utility::File offsetsFile{offsetsFilename_, "w"};
  offsetsFile.write(&ad_utility::EMBEDDING_VECTOR_VERSION,
                    sizeof(ad_utility::EMBEDDING_VECTOR_VERSION));
  offsetsFile.write(offsets_.data(), offsets_.size() * sizeof(uint64_t));
  offsetsFile.close();
}

// ____________________________________________________________________________
template <typename V>
EmbeddingVocabulary<V>::WordWriter::~WordWriter() {
  if (!finishWasCalled()) {
    ad_utility::terminateIfThrows(
        [this]() { this->finish(); },
        "Calling `finish` from the destructor of `EmbeddingVocabulary`");
  }
}

// ____________________________________________________________________________
template <typename V>
std::optional<ad_utility::MaybeOwnedVector>
EmbeddingVocabulary<V>::getEmbedding(uint64_t index) const {
  AD_CONTRACT_CHECK(index + 1 < offsets_.size());
  uint64_t start = offsets_[index];
  uint64_t numBytes = offsets_[index + 1] - start;
  size_t numFloats = numBytes / sizeof(float);
  // Reinterpret a byte region of the chosen storage as a `float` span. This is
  // the fp32 decode step; `start` is a multiple of `sizeof(float)` and the
  // storage base is suitably aligned, so the reinterpret is well-defined.
  auto asFloatSpan = [numFloats](const std::byte* base) {
    return ql::span<const float>{reinterpret_cast<const float*>(base),
                                 numFloats};
  };
  switch (mode_.value()) {
    case ad_utility::EmbeddingVectorAccess::Enum::Mmap:
      return ad_utility::MaybeOwnedVector{asFloatSpan(mapping_.data() + start)};
    case ad_utility::EmbeddingVectorAccess::Enum::InMemory:
      return ad_utility::MaybeOwnedVector{
          asFloatSpan(inMemory_.data() + start)};
    case ad_utility::EmbeddingVectorAccess::Enum::Pread: {
      std::vector<float> result(numFloats);
      dataFile_.read(result.data(), numBytes, start);
      return ad_utility::MaybeOwnedVector{std::move(result)};
    }
  }
  AD_FAIL();
}

// Explicit template instantiations
template class EmbeddingVocabulary<
    CompressedVocabulary<VocabularyInternalExternal>>;
template class EmbeddingVocabulary<VocabularyInMemory>;
