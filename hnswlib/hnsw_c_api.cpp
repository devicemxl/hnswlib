// ==================================================================
// hnswlib/hnsw_c_api.cpp
// ==================================================================
// Documentation for the C API wrapper of hnswlib.
// This file provides a C interface to the HNSW index, allowing
// usage from languages that support C bindings (Python, C#, Rust, etc.).
// It wraps the C++ classes (HierarchicalNSW, L2Space, InnerProductSpace)
// into opaque handles and simple C functions.
//
// The API includes:
//   - Index creation with L2 or Inner Product distance.
//   - Insertion (with or without overwriting deleted entries).
//   - Deletion (logical deletion by label).
//   - k‑NN search with configurable ef_search.
//   - Persistence (save/load to/from disk).
//   - Resizing and status queries.
//
// Thread safety: The underlying C++ index is thread‑safe for
// concurrent searches and concurrent inserts (except insert+search).
// This C API does not add additional locking; caller must ensure
// appropriate synchronization when mixing operations.
//
// Author:   hnswlib contributors (C API wrapper)
// License:  Apache 2.0
// ==================================================================

#include "hnsw_c_api.h"
#include "hnswlib.h"   // include the hnswlib headers
#include <string>
#include <cstring>
#include <new>

// ------------------------------------------------------------------
// Internal structure: opaque handle for C API
// ------------------------------------------------------------------

/**
 * @brief Internal structure holding the actual hnswlib index and its space.
 *
 * This structure is hidden from C clients; they only see a `HnswHandle`
 * (void*). It owns the space and algorithm objects and manages their
 * lifetimes. The `owns_space` flag indicates whether the space was
 * allocated by this object (true for new indices) or was provided
 * externally (currently always true, kept for potential extensions).
 */
struct HnswInternal {
    hnswlib::SpaceInterface<float>* space;   ///< Distance space (L2 or IP)
    hnswlib::AlgorithmInterface<float>* alg;  ///< HNSW index (HierarchicalNSW)
    int dim;                                  ///< Dimensionality of vectors
    hnswlib::labeltype next_label;            ///< For auto label assignment (not used if user provides)
    bool owns_space;                          ///< Whether this object owns `space` (to delete it)

    HnswInternal(hnswlib::SpaceInterface<float>* sp, hnswlib::AlgorithmInterface<float>* a, int d)
        : space(sp), alg(a), dim(d), next_label(0), owns_space(true) {}

    ~HnswInternal() {
        if (owns_space && space) delete space;
        if (alg) delete alg;
    }
};

// ------------------------------------------------------------------
// Error conversion helper (simplistic)
// ------------------------------------------------------------------

/**
 * @brief Converts a C++ exception to an HnswError code.
 *
 * This is a simple mapping. For more detailed error reporting,
 * the function could inspect the exception message.
 *
 * @param e Reference to the caught exception.
 * @return Corresponding error code (generic INVALID_PARAM for now).
 */
static HnswError wrap_error(const std::exception& e) {
    // Simplistic mapping – you can refine by checking e.what().
    return HNSW_ERR_INVALID_PARAM;
}

// ------------------------------------------------------------------
// Creation / Destruction
// ------------------------------------------------------------------

extern "C" {

/**
 * @brief Creates a new HNSW index with the specified parameters.
 *
 * @param space           Distance space: HNSW_SPACE_L2 or HNSW_SPACE_IP.
 * @param dim             Dimensionality of vectors (must be >0).
 * @param max_elements    Maximum number of points the index can hold.
 * @param M               Number of bidirectional links per node (except level0). Default 16.
 * @param ef_construction Size of dynamic candidate list during construction. Default 200.
 * @return Opaque handle to the index, or NULL on failure.
 */
HNSW_API HnswHandle hnsw_create(HnswSpace space, int dim, size_t max_elements,
                                 int M, int ef_construction) {
    if (dim <= 0 || max_elements == 0 || M <= 0 || ef_construction <= 0)
        return nullptr;

    hnswlib::SpaceInterface<float>* space_ptr = nullptr;
    try {
        // Instantiate the appropriate distance space
        if (space == HNSW_SPACE_L2) {
            space_ptr = new hnswlib::L2Space(dim);
        } else if (space == HNSW_SPACE_IP) {
            space_ptr = new hnswlib::InnerProductSpace(dim);
        } else {
            return nullptr;
        }

        // Create the HNSW index
        hnswlib::HierarchicalNSW<float>* alg = new hnswlib::HierarchicalNSW<float>(
            space_ptr, max_elements, M, ef_construction);
        HnswInternal* internal = new HnswInternal(space_ptr, alg, dim);
        return static_cast<HnswHandle>(internal);
    } catch (...) {
        if (space_ptr) delete space_ptr;
        return nullptr;
    }
}

/**
 * @brief Destroys an HNSW index and frees all associated memory.
 * @param idx Handle to the index (can be NULL).
 */
HNSW_API void hnsw_destroy(HnswHandle idx) {
    if (!idx) return;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    delete internal;
}

// ---------------------------------------------------------------------
// Insertion and Deletion
// ---------------------------------------------------------------------

/**
 * @brief Internal helper for insertion (with or without replace_deleted flag).
 * @param idx             Handle to index.
 * @param vector          Pointer to float array of length `dim`.
 * @param label           External label for the point (must be unique).
 * @param replace_deleted If true, allows overwriting a deleted entry (if enabled in constructor).
 * @return HNSW_OK on success, or an error code.
 */
static HnswError insert_impl(HnswHandle idx, const float* vector, size_t label, bool replace_deleted) {
    if (!idx || !vector) return HNSW_ERR_INVALID_PARAM;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    try {
        internal->alg->addPoint(static_cast<const void*>(vector), label, replace_deleted);
        return HNSW_OK;
    } catch (const std::exception&) {
        return HNSW_ERR_INVALID_PARAM;
    }
}

/**
 * @brief Inserts a point into the index (does not replace deleted ones).
 * @param idx     Handle to index.
 * @param vector  Pointer to float array.
 * @param label   External label.
 * @return HNSW_OK on success, error otherwise.
 */
HNSW_API HnswError hnsw_insert(HnswHandle idx, const float* vector, size_t label) {
    return insert_impl(idx, vector, label, false);
}

/**
 * @brief Inserts a point, potentially replacing a deleted entry if allowed.
 * @param idx     Handle to index.
 * @param vector  Pointer to float array.
 * @param label   External label.
 * @return HNSW_OK on success, error otherwise.
 */
HNSW_API HnswError hnsw_insert_replace(HnswHandle idx, const float* vector, size_t label) {
    return insert_impl(idx, vector, label, true);
}

/**
 * @brief Marks a point as deleted (logical deletion). Does not remove from graph.
 * @param idx   Handle to index.
 * @param label External label of the point to delete.
 * @return HNSW_OK if found and marked, HNSW_ERR_NOT_FOUND otherwise.
 */
HNSW_API HnswError hnsw_mark_deleted(HnswHandle idx, size_t label) {
    if (!idx) return HNSW_ERR_INVALID_PARAM;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    try {
        internal->alg->markDelete(label);
        return HNSW_OK;
    } catch (...) {
        return HNSW_ERR_NOT_FOUND;
    }
}

// ---------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------

/**
 * @brief Searches for the k nearest neighbors to a query vector.
 *
 * The function temporarily sets the `ef_search` parameter of the index
 * to the provided value, performs the search, and restores the original ef.
 *
 * @param idx           Handle to index.
 * @param query_vector  Pointer to query float array.
 * @param k             Number of neighbors to retrieve.
 * @param ef_search     Size of dynamic candidate list during search (higher = more accurate but slower).
 * @param out_labels    Output array (size at least k) where labels will be stored.
 * @param out_distances Output array (size at least k) where distances will be stored.
 * @return Number of neighbors actually found (may be less than k if index has fewer points),
 *         or negative error code on failure.
 */
HNSW_API int hnsw_search(HnswHandle idx, const float* query_vector, size_t k,
                          int ef_search, size_t* out_labels, float* out_distances) {
    if (!idx || !query_vector || k == 0 || !out_labels || !out_distances)
        return -HNSW_ERR_INVALID_PARAM;

    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    // Cast to HierarchicalNSW (only supported algorithm type in this wrapper)
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    if (!alg) return -HNSW_ERR_INVALID_PARAM;

    // Temporarily change ef_search
    int old_ef = alg->ef_;
    alg->ef_ = ef_search;

    // Perform search
    auto result = alg->searchKnn(query_vector, k);
    size_t returned = result.size();
    if (returned > k) returned = k;

    // The result is a max‑heap (farthest distance on top). We reorder so that
    // output has closest first (distance increasing). This is more intuitive for users.
    std::vector<std::pair<float, hnswlib::labeltype>> ordered;
    ordered.reserve(returned);
    while (!result.empty()) {
        ordered.push_back(result.top());
        result.pop();
    }
    // ordered[0] = farthest, ordered[returned-1] = closest
    for (size_t i = 0; i < returned; ++i) {
        out_labels[i] = ordered[returned - 1 - i].second;
        out_distances[i] = ordered[returned - 1 - i].first;
    }

    // Restore original ef
    alg->ef_ = old_ef;
    return static_cast<int>(returned);
}

/**
 * @brief Sets the ef_search parameter for subsequent searches.
 * @param idx       Handle to index.
 * @param ef_search New ef value.
 */
HNSW_API void hnsw_set_ef(HnswHandle idx, int ef_search) {
    if (!idx) return;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    if (alg) alg->ef_ = ef_search;
}

/**
 * @brief Gets the current ef_search parameter.
 * @param idx Handle to index.
 * @return Current ef value, or -1 on error.
 */
HNSW_API int hnsw_get_ef(HnswHandle idx) {
    if (!idx) return -1;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    return alg ? alg->ef_ : -1;
}

// ---------------------------------------------------------------------
// Index persistence
// ---------------------------------------------------------------------

/**
 * @brief Saves the index to a binary file.
 * @param idx      Handle to index.
 * @param filepath Path to the output file.
 * @return HNSW_OK on success, HNSW_ERR_IO on error.
 */
HNSW_API HnswError hnsw_save(HnswHandle idx, const char* filepath) {
    if (!idx || !filepath) return HNSW_ERR_INVALID_PARAM;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    try {
        internal->alg->saveIndex(filepath);
        return HNSW_OK;
    } catch (...) {
        return HNSW_ERR_IO;
    }
}

/**
 * @brief Loads an index from a binary file previously saved with hnsw_save.
 * @param filepath Path to the index file.
 * @param space    Distance space that matches the saved index (L2 or IP).
 * @param dim      Dimensionality (must match saved index).
 * @return Handle to the loaded index, or NULL on failure.
 */
HNSW_API HnswHandle hnsw_load(const char* filepath, HnswSpace space, int dim) {
    if (!filepath || dim <= 0) return nullptr;
    hnswlib::SpaceInterface<float>* space_ptr = nullptr;
    try {
        if (space == HNSW_SPACE_L2) {
            space_ptr = new hnswlib::L2Space(dim);
        } else if (space == HNSW_SPACE_IP) {
            space_ptr = new hnswlib::InnerProductSpace(dim);
        } else {
            return nullptr;
        }

        // HierarchicalNSW has a constructor that loads from file
        hnswlib::HierarchicalNSW<float>* alg = new hnswlib::HierarchicalNSW<float>(space_ptr, filepath);
        HnswInternal* internal = new HnswInternal(space_ptr, alg, dim);
        return static_cast<HnswHandle>(internal);
    } catch (...) {
        if (space_ptr) delete space_ptr;
        return nullptr;
    }
}

// ---------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------

/**
 * @brief Resizes the index to a new maximum capacity.
 * @param idx              Handle to index.
 * @param new_max_elements New maximum number of elements (must be >= current count).
 * @return HNSW_OK on success, error code otherwise.
 */
HNSW_API HnswError hnsw_resize(HnswHandle idx, size_t new_max_elements) {
    if (!idx) return HNSW_ERR_INVALID_PARAM;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    if (!alg) return HNSW_ERR_INVALID_PARAM;
    try {
        alg->resizeIndex(new_max_elements);
        return HNSW_OK;
    } catch (...) {
        return HNSW_ERR_MEMORY;
    }
}

/**
 * @brief Returns the current number of elements in the index (including deleted).
 * @param idx Handle to index.
 * @return Element count, or 0 if invalid handle.
 */
HNSW_API size_t hnsw_get_current_count(HnswHandle idx) {
    if (!idx) return 0;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    return alg ? alg->getCurrentElementCount() : 0;
}

/**
 * @brief Returns the maximum capacity of the index.
 * @param idx Handle to index.
 * @return Maximum number of elements, or 0 if invalid handle.
 */
HNSW_API size_t hnsw_get_max_elements(HnswHandle idx) {
    if (!idx) return 0;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    return alg ? alg->max_elements_ : 0;
}

/**
 * @brief Returns the dimensionality of vectors stored in the index.
 * @param idx Handle to index.
 * @return Dimension, or 0 if invalid handle.
 */
HNSW_API int hnsw_get_dim(HnswHandle idx) {
    if (!idx) return 0;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    return internal->dim;
}

} // extern "C"
