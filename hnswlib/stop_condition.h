// ==================================================================
// hnswlib/stop_condition.h
// ==================================================================
// Documentation for custom search stop conditions.
// This file provides advanced stop conditions used to control
// nearest neighbor search beyond simple k‑NN. It includes:
//   - Multi‑vector search: each "document" may have multiple vectors,
//     and we want k distinct documents (not k vectors).
//   - Epsilon (radius) search: retrieve all points within a distance
//     threshold, with optional limits on result set size.
//
// The stop condition mechanism allows early termination and dynamic
// result set pruning during the HNSW base layer search.
//
// Author:   hnswlib contributors
// License:  Apache 2.0
// ==================================================================

#pragma once
#include "space_l2.h"
#include "space_ip.h"
#include <assert.h>
#include <unordered_map>

namespace hnswlib {

// ------------------------------------------------------------------
// BaseMultiVectorSpace : abstract space for documents with multiple vectors
// ------------------------------------------------------------------

/**
 * @brief Abstract interface for a space where each stored "point"
 *        consists of a vector (float array) plus a document ID.
 *
 * This is used when one logical document can be represented by several
 * vectors (e.g., multiple embeddings per article). The HNSW index stores
 * each vector separately, but search results should return unique documents.
 * The document ID is stored together with the vector data.
 *
 * @tparam DOCIDTYPE Type used for document identifiers (e.g., int, size_t).
 */
template<typename DOCIDTYPE>
class BaseMultiVectorSpace : public SpaceInterface<float> {
 public:
    /**
     * @brief Extracts the document ID from a stored data point.
     * @param datapoint Pointer to the raw data (vector + doc ID).
     * @return Document ID.
     */
    virtual DOCIDTYPE get_doc_id(const void *datapoint) = 0;

    /**
     * @brief Sets the document ID in a data point buffer.
     * @param datapoint Pointer to the raw data buffer.
     * @param doc_id    Document ID to write.
     */
    virtual void set_doc_id(void *datapoint, DOCIDTYPE doc_id) = 0;
};

// ------------------------------------------------------------------
// MultiVectorL2Space : L2 space for documents with multiple vectors
// ------------------------------------------------------------------

/**
 * @brief Concrete multi‑vector space using squared L2 distance.
 *
 * Each stored element is: [float vector (dim dimensions)] + [DOCIDTYPE doc_id].
 * The distance is computed only on the vector part, ignoring the doc ID.
 * This class automatically selects SIMD optimizations for L2 based on dimension.
 *
 * @tparam DOCIDTYPE Type for document identifiers.
 */
template<typename DOCIDTYPE>
class MultiVectorL2Space : public BaseMultiVectorSpace<DOCIDTYPE> {
    DISTFUNC<float> fstdistfunc_;   ///< Distance function (L2, possibly SIMD)
    size_t data_size_;              ///< Total bytes per element: vector + doc ID
    size_t vector_size_;            ///< Bytes per vector: dim * sizeof(float)
    size_t dim_;                    ///< Number of dimensions

 public:
    /**
     * @brief Constructs a multi‑vector L2 space.
     * @param dim Dimensionality of each vector.
     */
    MultiVectorL2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        else if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = L2SqrSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = L2SqrSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = L2SqrSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        dim_ = dim;
        vector_size_ = dim * sizeof(float);
        data_size_ = vector_size_ + sizeof(DOCIDTYPE);
    }

    size_t get_data_size() override {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() override {
        return fstdistfunc_;
    }

    void *get_dist_func_param() override {
        return &dim_;
    }

    /**
     * @brief Extracts document ID from the end of the data block.
     */
    DOCIDTYPE get_doc_id(const void *datapoint) override {
        return *(DOCIDTYPE *)((char *)datapoint + vector_size_);
    }

    /**
     * @brief Writes document ID at the end of the data block.
     */
    void set_doc_id(void *datapoint, DOCIDTYPE doc_id) override {
        *(DOCIDTYPE*)((char *)datapoint + vector_size_) = doc_id;
    }

    ~MultiVectorL2Space() {}
};

// ------------------------------------------------------------------
// MultiVectorInnerProductSpace : inner product space for multi‑vector documents
// ------------------------------------------------------------------

/**
 * @brief Concrete multi‑vector space using inner product distance (1 - dot).
 *
 * Similar to MultiVectorL2Space, but uses 1‑dot as distance.
 *
 * @tparam DOCIDTYPE Type for document identifiers.
 */
template<typename DOCIDTYPE>
class MultiVectorInnerProductSpace : public BaseMultiVectorSpace<DOCIDTYPE> {
    DISTFUNC<float> fstdistfunc_;   ///< Distance function (inner product based)
    size_t data_size_;              ///< Total bytes per element
    size_t vector_size_;            ///< Bytes per vector
    size_t dim_;                    ///< Dimensionality

 public:
    /**
     * @brief Constructs a multi‑vector inner product space.
     * @param dim Dimensionality of each vector.
     */
    MultiVectorInnerProductSpace(size_t dim) {
        fstdistfunc_ = InnerProductDistance;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
        } else if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #elif defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #endif
    #if defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
            InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
        }
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif
        vector_size_ = dim * sizeof(float);
        data_size_ = vector_size_ + sizeof(DOCIDTYPE);
    }

    size_t get_data_size() override {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() override {
        return fstdistfunc_;
    }

    void *get_dist_func_param() override {
        return &dim_;
    }

    DOCIDTYPE get_doc_id(const void *datapoint) override {
        return *(DOCIDTYPE *)((char *)datapoint + vector_size_);
    }

    void set_doc_id(void *datapoint, DOCIDTYPE doc_id) override {
        *(DOCIDTYPE*)((char *)datapoint + vector_size_) = doc_id;
    }

    ~MultiVectorInnerProductSpace() {}
};

// ------------------------------------------------------------------
// MultiVectorSearchStopCondition : stop condition for multi‑vector search
// ------------------------------------------------------------------

/**
 * @brief Stop condition that retrieves a set of distinct documents,
 *        each possibly represented by multiple vectors.
 *
 * This condition is used when the index contains many vectors per document.
 * The search returns the k most relevant documents, where a document's
 * relevance is determined by its closest vector to the query (the distance
 * of the best matching vector for that document). The internal algorithm
 * uses a collection candidate set of size `ef_collection` (≥ k) to gather
 * enough vectors before filtering distinct documents.
 *
 * @tparam DOCIDTYPE Type for document IDs.
 * @tparam dist_t    Type for distance values (float/double).
 */
template<typename DOCIDTYPE, typename dist_t>
class MultiVectorSearchStopCondition : public BaseSearchStopCondition<dist_t> {
    size_t curr_num_docs_;                               ///< Current number of distinct documents in result set
    size_t num_docs_to_search_;                          ///< Target number of documents (k)
    size_t ef_collection_;                               ///< Size of collection candidate list (≥ k)
    std::unordered_map<DOCIDTYPE, size_t> doc_counter_;  ///< How many vectors per doc are currently in results
    std::priority_queue<std::pair<dist_t, DOCIDTYPE>> search_results_; ///< Heap of (distance, doc_id) for pruning
    BaseMultiVectorSpace<DOCIDTYPE>& space_;             ///< Reference to the space (to extract doc IDs)

 public:
    /**
     * @brief Constructs a multi‑vector stop condition.
     * @param space            Multi‑vector space used to extract doc IDs from points.
     * @param num_docs_to_search Number of distinct documents to retrieve (k).
     * @param ef_collection      Size of the candidate collection (≥ k). Larger values improve recall.
     */
    MultiVectorSearchStopCondition(
        BaseMultiVectorSpace<DOCIDTYPE>& space,
        size_t num_docs_to_search,
        size_t ef_collection = 10)
        : space_(space) {
            curr_num_docs_ = 0;
            num_docs_to_search_ = num_docs_to_search;
            ef_collection_ = std::max(ef_collection, num_docs_to_search);
        }

    /**
     * @brief Called when a point is added to the result set.
     *        Increments the counter for its document ID.
     * @param label      Point label (unused here).
     * @param datapoint  Raw data (vector + doc ID).
     * @param dist       Distance from query.
     */
    void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) override {
        DOCIDTYPE doc_id = space_.get_doc_id(datapoint);
        if (doc_counter_[doc_id] == 0) {
            curr_num_docs_ += 1;
        }
        search_results_.emplace(dist, doc_id);
        doc_counter_[doc_id] += 1;
    }

    /**
     * @brief Called when a point is removed from the result set (pruned).
     *        Decrements the counter for its document ID.
     */
    void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) override {
        DOCIDTYPE doc_id = space_.get_doc_id(datapoint);
        doc_counter_[doc_id] -= 1;
        if (doc_counter_[doc_id] == 0) {
            curr_num_docs_ -= 1;
        }
        search_results_.pop();
    }

    /**
     * @brief Determines whether the search should stop.
     *        Stops when the candidate distance exceeds the lower bound and
     *        we already have `ef_collection_` distinct documents.
     */
    bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) override {
        bool stop_search = candidate_dist > lowerBound && curr_num_docs_ == ef_collection_;
        return stop_search;
    }

    /**
     * @brief Determines whether a candidate point should be considered.
     *        Considers if we have less than ef_collection_ docs or the distance
     *        is better than the current worst distance.
     */
    bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) override {
        bool flag_consider_candidate = curr_num_docs_ < ef_collection_ || lowerBound > candidate_dist;
        return flag_consider_candidate;
    }

    /**
     * @brief Indicates whether extra results should be removed.
     *        Removal occurs when we have more than `ef_collection_` distinct documents.
     */
    bool should_remove_extra() override {
        bool flag_remove_extra = curr_num_docs_ > ef_collection_;
        return flag_remove_extra;
    }

    /**
     * @brief Final filtering: reduces the result set from `ef_collection_`
     *        down to exactly `num_docs_to_search_` distinct documents.
     * @param candidates Vector of (distance, label) sorted farthest to closest? Actually
     *                   the search returns farthest first, so we pop from the back.
     */
    void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) override {
        while (curr_num_docs_ > num_docs_to_search_) {
            dist_t dist_cand = candidates.back().first;
            dist_t dist_res = search_results_.top().first;
            assert(dist_cand == dist_res);   // consistency check
            DOCIDTYPE doc_id = search_results_.top().second;
            doc_counter_[doc_id] -= 1;
            if (doc_counter_[doc_id] == 0) {
                curr_num_docs_ -= 1;
            }
            search_results_.pop();
            candidates.pop_back();           // remove the farthest result
        }
    }

    ~MultiVectorSearchStopCondition() {}
};

// ------------------------------------------------------------------
// EpsilonSearchStopCondition : radius (threshold) search
// ------------------------------------------------------------------

/**
 * @brief Stop condition for epsilon (radius) search: retrieve all points
 *        within a given distance threshold `epsilon`, optionally limited
 *        by minimum and maximum number of candidates.
 *
 * This condition stops when:
 *   - The candidate distance exceeds the current worst result (`lowerBound`)
 *     and we already have `max_num_candidates_` results; OR
 *   - The candidate distance exceeds `epsilon` and we have at least
 *     `min_num_candidates_` results.
 *
 * After search, points with distance > epsilon are filtered out.
 *
 * @tparam dist_t Type for distance values.
 */
template<typename dist_t>
class EpsilonSearchStopCondition : public BaseSearchStopCondition<dist_t> {
    float epsilon_;                 ///< Distance threshold (points with distance > epsilon are discarded)
    size_t min_num_candidates_;     ///< Minimum number of points to return (even if some exceed epsilon)
    size_t max_num_candidates_;     ///< Maximum number of points to keep during search
    size_t curr_num_items_;         ///< Current number of items in the result set

 public:
    /**
     * @brief Constructs an epsilon stop condition.
     * @param epsilon            Distance threshold. Only points with distance ≤ epsilon survive final filtering.
     * @param min_num_candidates Minimum number of points to return (if fewer exist within epsilon, more may be kept).
     * @param max_num_candidates Maximum number of points to maintain in the candidate set during search.
     */
    EpsilonSearchStopCondition(float epsilon, size_t min_num_candidates, size_t max_num_candidates) {
        assert(min_num_candidates <= max_num_candidates);
        epsilon_ = epsilon;
        min_num_candidates_ = min_num_candidates;
        max_num_candidates_ = max_num_candidates;
        curr_num_items_ = 0;
    }

    void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) override {
        curr_num_items_ += 1;
    }

    void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) override {
        curr_num_items_ -= 1;
    }

    /**
     * @brief Determines whether to stop the search.
     */
    bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) override {
        if (candidate_dist > lowerBound && curr_num_items_ == max_num_candidates_) {
            // No way to improve the result set (candidate is worse than the worst kept result)
            return true;
        }
        if (candidate_dist > epsilon_ && curr_num_items_ >= min_num_candidates_) {
            // We are already outside epsilon radius and have at least the minimum required points
            return true;
        }
        return false;
    }

    /**
     * @brief Determines whether a candidate point should be considered.
     */
    bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) override {
        bool flag_consider_candidate = curr_num_items_ < max_num_candidates_ || lowerBound > candidate_dist;
        return flag_consider_candidate;
    }

    /**
     * @brief Indicates whether extra results should be removed.
     */
    bool should_remove_extra() {
        bool flag_remove_extra = curr_num_items_ > max_num_candidates_;
        return flag_remove_extra;
    }

    /**
     * @brief Final filtering: discards points with distance > epsilon,
     *        and also truncates to max_num_candidates_ if necessary.
     * @param candidates Vector of (distance, label) sorted by increasing distance
     *                   (closest first) due to later conversion.
     */
    void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) override {
        // Remove points that exceed epsilon (from the back, since farthest first)
        while (!candidates.empty() && candidates.back().first > epsilon_) {
            candidates.pop_back();
        }
        // Also ensure we don't exceed max_num_candidates_ (should already hold, but safe)
        while (candidates.size() > max_num_candidates_) {
            candidates.pop_back();
        }
    }

    ~EpsilonSearchStopCondition() {}
};

}  // namespace hnswlib
