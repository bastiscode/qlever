// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Sebastian Walter <swalter@informatik.uni-freiburg.de>

#ifndef QLEVER_SRC_RDFTYPES_EMBEDDINGVECTOR_H
#define QLEVER_SRC_RDFTYPES_EMBEDDINGVECTOR_H

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include "backports/span.h"
#include "util/EnumWithStrings.h"

namespace ad_utility {

// A vector of `float`s that is either *borrowed* — a `span` into storage that
// outlives the access (a memory map or an in-memory buffer held by the
// `EmbeddingVocabulary`, valid for the index's lifetime) — or *owned* (a
// `std::vector`, e.g. a `pread` copy or a freshly parsed query vector). This
// lets `getEmbedding` be zero-copy for the `mmap`/`in-memory` backends while
// still supporting the owning `pread`/parse paths behind one type.
//
// The view is always computed on demand by `span()` from whichever alternative
// is active, so the two alternatives stay mutually exclusive and there is never
// a `span` member pointing into a sibling owned `vector` (which a move would
// dangle).
class MaybeOwnedVector {
 private:
  std::variant<ql::span<const float>, std::vector<float>> data_;

 public:
  // Read-only contiguous range of `float` (the typedefs let it be used directly
  // as a container, e.g. with gtest's `ElementsAre`).
  using value_type = float;
  using const_iterator = typename ql::span<const float>::iterator;
  using iterator = const_iterator;

  MaybeOwnedVector() = default;
  explicit MaybeOwnedVector(ql::span<const float> borrowed) : data_{borrowed} {}
  explicit MaybeOwnedVector(std::vector<float> owned)
      : data_{std::move(owned)} {}

  // The vector contents, regardless of ownership.
  ql::span<const float> span() const {
    return std::visit(
        [](const auto& d) -> ql::span<const float> {
          return ql::span<const float>{d};
        },
        data_);
  }

  size_t size() const { return span().size(); }

  // Range access (iterates the underlying storage; the `span` temporary's
  // iterators are plain pointers into that storage, so they stay valid).
  auto begin() const { return span().begin(); }
  auto end() const { return span().end(); }
};

// On-disk format version of the `EmbeddingVocabulary` sidecar. Bump this
// whenever the sidecar layout changes; an index with a mismatching version is
// rejected at load and must be rebuilt.
//
// Version 2: the `.embvec` data blob carries a trailing `MmapVectorMetaData`
// record (so it can be opened directly via `MmapVectorView`), in addition to
// the raw `float32` bytes.
constexpr inline uint32_t EMBEDDING_VECTOR_VERSION = 2;

// How `EmbeddingVocabulary` reads the vectors out of the `.embvec` sidecar at
// query time. This is a load-time choice (read once when the vocabulary is
// opened), selected via the `embedding-vector-access` runtime parameter:
// - `Pread`    (default): positioned reads into an owned buffer; WASM-safe,
//                         predictable RSS, consistent with the main vocabulary.
// - `Mmap`:               memory-map the sidecar and borrow `span`s into it
//                         (zero-copy); not available under WASM.
// - `InMemory`:           load the whole blob into a private RAM buffer once
//                         and borrow `span`s into it (fastest, non-reclaimable).
namespace detail {
enum struct EmbeddingVectorAccessEnum { Pread, Mmap, InMemory };
}
class EmbeddingVectorAccess
    : public EnumWithStrings<EmbeddingVectorAccess,
                             detail::EmbeddingVectorAccessEnum> {
 public:
  using Enum = detail::EmbeddingVectorAccessEnum;

  static constexpr std::array<std::pair<Enum, std::string_view>, 3>
      descriptions_{{{Enum::Pread, "pread"},
                     {Enum::Mmap, "mmap"},
                     {Enum::InMemory, "in-memory"}}};
  static const EmbeddingVectorAccess Pread;
  static const EmbeddingVectorAccess Mmap;
  static const EmbeddingVectorAccess InMemory;

  static constexpr std::string_view typeName() {
    return "embedding vector access mode";
  }

  using EnumWithStrings::EnumWithStrings;
};

const inline EmbeddingVectorAccess EmbeddingVectorAccess::Pread{
    EmbeddingVectorAccess::Enum::Pread};
const inline EmbeddingVectorAccess EmbeddingVectorAccess::Mmap{
    EmbeddingVectorAccess::Enum::Mmap};
const inline EmbeddingVectorAccess EmbeddingVectorAccess::InMemory{
    EmbeddingVectorAccess::Enum::InMemory};

// Parse the body of an embedding-vector literal, i.e. a string of the form
// `[0.1, -0.2, 1.0]`, into a vector of `float`s. Returns `std::nullopt` if the
// string is not a well-formed, non-empty array of finite decimal numbers.
// Shared by the `EmbeddingVocabulary` builder (decoding the stored literal) and
// the query-time value getter (decoding an inline query vector).
std::optional<std::vector<float>> parseFloatVectorArrayBody(
    std::string_view body);

// Parse a full serialized `fp32-vector` literal of the form
// `"[...]"^^<...fp32-vector>` into a vector of `float`s: strip the surrounding
// quote and datatype suffix, then delegate to `parseFloatVectorArrayBody`.
// Returns `std::nullopt` if the suffix does not match or the body is malformed.
std::optional<std::vector<float>> parseFp32VectorLiteral(std::string_view word);

}  // namespace ad_utility

#endif  // QLEVER_SRC_RDFTYPES_EMBEDDINGVECTOR_H
