// ==================================================================
// hnswlib/bruteforce.h
// ==================================================================
// Professional documentation for the BruteforceSearch class.
// This file is part of the hnswlib library (https://github.com/nmslib/hnswlib)
//
// Description:
//   Provides an exact brute-force search implementation for nearest neighbor
//   queries. It is primarily used for debugging, benchmarking, and as a
//   reference baseline when comparing with approximate methods (e.g., HNSW).
//   All operations are O(N) and thread-safe thanks to a mutex.
//
// Author:   hnswlib contributors
// License:  Apache 2.0
// ==================================================================

#pragma once
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <assert.h>

namespace hnswlib {

/**
 * @brief Exact brute‑force nearest neighbor search index.
 *
 * This class implements the AlgorithmInterface by storing all points in a
 * contiguous memory block and performing a linear scan for every search.
 * It is useful as a ground‑truth comparator or for very small datasets.
 *
 * @tparam dist_t Type used for distance values (usually float or double).
 */
template<typename dist_t>
class BruteforceSearch : public AlgorithmInterface<dist_t> {
 public:
    // ------------------------------------------------------------------
    // Public member variables (exposed for performance reasons)
    // ------------------------------------------------------------------
    char *data_;                    ///< Raw memory buffer: [vector | label] repeated.
    size_t maxelements_;            ///< Capacity of the index (maximum number of points).
    size_t cur_element_count;       ///< Current number of stored points.
    size_t size_per_element_;       ///< Bytes per element = data_size_ + sizeof(labeltype).

    size_t data_size_;              ///< Bytes per vector (depends on space and dims).
    DISTFUNC <dist_t> fstdistfunc_; ///< Distance function pointer (e.g., L2, IP).
    void *dist_func_param_;         ///< Extra parameters for the distance function.
    std::mutex index_lock;          ///< Mutex protecting concurrent modifications.

    /// Map from external user-defined label to internal array index.
    std::unordered_map<labeltype, size_t> dict_external_to_internal;

    // ------------------------------------------------------------------
    // Constructors / Destructor
    // ------------------------------------------------------------------

    /**
     * @brief Default constructor (does not allocate memory).
     * @param s Pointer to a SpaceInterface defining distance and data layout.
     *          Only needed for loadIndex later.
     */
    BruteforceSearch(SpaceInterface <dist_t> *s)
        : data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
    }

    /**
     * @brief Constructor that loads an existing index from disk.
     * @param s        SpaceInterface (distance + data size).
     * @param location Path to the file previously saved by saveIndex().
     */
    BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
        : data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
        loadIndex(location, s);
    }

    /**
     * @brief Constructs an empty brute‑force index with a given capacity.
     * @param s           SpaceInterface providing data size and distance function.
     * @param maxElements Maximum number of points the index can hold.
     * @throws std::runtime_error if memory allocation fails.
     */
    BruteforceSearch(SpaceInterface <dist_t> *s, size_t maxElements) {
        maxelements_ = maxElements;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        size_per_element_ = data_size_ + sizeof(labeltype);
        data_ = (char *) malloc(maxElements * size_per_element_);
        if (data_ == nullptr)
            throw std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
        cur_element_count = 0;
    }

    /**
     * @brief Destructor – frees the internal memory buffer.
     */
    ~BruteforceSearch() {
        free(data_);
    }

    // ------------------------------------------------------------------
    // Core operations
    // ------------------------------------------------------------------

    /**
     * @brief Inserts a new point into the index, or replaces an existing one.
     *
     * If the label already exists, the point is overwritten. Otherwise, the point
     * is appended. This operation is thread‑safe.
     *
     * @param datapoint      Pointer to the vector data (size data_size_ bytes).
     * @param label          External identifier for the point (must be unique).
     * @param replace_deleted Ignored (kept for interface compatibility).
     * @throws std::runtime_error if the capacity (maxelements_) is exceeded.
     */
    void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) {
        int idx;
        {
            std::unique_lock<std::mutex> lock(index_lock);

            auto search = dict_external_to_internal.find(label);
            if (search != dict_external_to_internal.end()) {
                idx = search->second;               // overwrite existing
            } else {
                if (cur_element_count >= maxelements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit\n");
                }
                idx = cur_element_count;
                dict_external_to_internal[label] = idx;
                cur_element_count++;
            }
        }
        // Write label at the end of the element block.
        memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
        // Write vector data at the beginning.
        memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
    }

    /**
     * @brief Removes a point by its label.
     *
     * The removal is performed by swapping the to‑be‑removed element with the last
     * element in the array, then decrementing the element count. This maintains
     * compact storage but changes the internal index of the swapped point.
     *
     * @param cur_external Label of the point to remove.
     */
    void removePoint(labeltype cur_external) {
        std::unique_lock<std::mutex> lock(index_lock);

        auto found = dict_external_to_internal.find(cur_external);
        if (found == dict_external_to_internal.end()) {
            return;   // nothing to remove
        }

        dict_external_to_internal.erase(found);
        size_t cur_c = found->second;                // internal index to remove

     
        size_t cur_c = found->second;
        labeltype label = *((labeltype*)(data_ + size_per_element_ * (cur_element_count-1) + data_size_));
        dict_external_to_internal[label] = cur_c;
        memcpy(data_ + size_per_element_ * cur_c,
                data_ + size_per_element_ * (cur_element_count-1),
                data_size_+sizeof(labeltype));


        cur_element_count--;
    }

    /**
     * @brief Performs a k‑nearest neighbor search using brute‑force scanning.
     *
     * Computes distances from the query to every stored point, keeps the k best
     * results (smallest distances) using a max‑heap. Optionally filters points
     * via a user‑provided functor.
     *
     * @param query_data   Pointer to the query vector.
     * @param k            Number of nearest neighbors to retrieve.
     * @param isIdAllowed  Optional filter: if not null, only points for which
     *                     (*isIdAllowed)(label) returns true are considered.
     * @return A max‑heap (priority_queue) of pairs (distance, label).
     *         The top of the heap contains the worst distance among the k results.
     *
     * @note This method is thread‑safe for concurrent reads.
     */
    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        assert(k <= cur_element_count);
        std::priority_queue<std::pair<dist_t, labeltype >> topResults;
        dist_t lastdist = std::numeric_limits<dist_t>::max();
        for (int i = 0; i < cur_element_count; i++) {
            dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
            if (dist <= lastdist || topResults.size() < k) {
                labeltype label = *((labeltype *) (data_ + size_per_element_ * i + data_size_));
                if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                    topResults.emplace(dist, label);
                    if (topResults.size() > k)
                        topResults.pop();
                    if (!topResults.empty())
                        lastdist = topResults.top().first;
                }
            }
        }
        return topResults;
    }

    // ------------------------------------------------------------------
    // Persistence
    // ------------------------------------------------------------------

    /**
     * @brief Saves the entire index to a binary file.
     *
     * The file contains (in order):
     *   - maxelements_  (size_t)
     *   - size_per_element_ (size_t)
     *   - cur_element_count (size_t)
     *   - raw data buffer (maxelements_ * size_per_element_ bytes)
     *
     * @param location File system path where the index will be stored.
     */
    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, maxelements_);
        writeBinaryPOD(output, size_per_element_);
        writeBinaryPOD(output, cur_element_count);

        output.write(data_, maxelements_ * size_per_element_);

        output.close();
    }

    /**
     * @brief Loads an index previously saved with saveIndex().
     *
     * The space interface must be compatible with the stored data layout
     * (same dimensions and distance type).
     *
     * @param location Path to the binary file.
     * @param s        SpaceInterface that provides the distance function and
     *                 data_size_ needed to interpret the buffer.
     * @throws std::runtime_error if memory allocation fails.
     */
    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {
        std::ifstream input(location, std::ios::binary);
        std::streampos position;

        readBinaryPOD(input, maxelements_);
        readBinaryPOD(input, size_per_element_);
        readBinaryPOD(input, cur_element_count);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        size_per_element_ = data_size_ + sizeof(labeltype);
        data_ = (char *) malloc(maxelements_ * size_per_element_);
        if (data_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate data");

        input.read(data_, maxelements_ * size_per_element_);

        input.close();
    }
};

}  // namespace hnswlib
