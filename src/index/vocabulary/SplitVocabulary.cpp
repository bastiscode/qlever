// Copyright 2025 University of Freiburg
// Chair of Algorithms and Data Structures
// Author: Christoph Ullinger <ullingec@cs.uni-freiburg.de>

#include "index/vocabulary/EmbeddingVocabulary.h"
#include "index/vocabulary/GeoVocabulary.h"
#include "index/vocabulary/SplitVocabularyImpl.h"

// Explicit template instantiations
using namespace detail::splitVocabulary;
template class SplitVocabulary<
    GeoSplitFunc, GeoFilenameFunc,
    CompressedVocabulary<VocabularyInternalExternal>,
    GeoVocabulary<CompressedVocabulary<VocabularyInternalExternal>>>;
template class SplitVocabulary<GeoSplitFunc, GeoFilenameFunc,
                               VocabularyInMemory,
                               GeoVocabulary<VocabularyInMemory>>;

// The combined "special" split (`regular | geo | embedding`) used by the main
// index, for both relevant underlying vocabulary types.
template class SplitVocabulary<
    SpecialSplitFunc, SpecialFilenameFunc,
    CompressedVocabulary<VocabularyInternalExternal>,
    GeoVocabulary<CompressedVocabulary<VocabularyInternalExternal>>,
    EmbeddingVocabulary<CompressedVocabulary<VocabularyInternalExternal>>>;
template class SplitVocabulary<
    SpecialSplitFunc, SpecialFilenameFunc, VocabularyInMemory,
    GeoVocabulary<VocabularyInMemory>, EmbeddingVocabulary<VocabularyInMemory>>;
