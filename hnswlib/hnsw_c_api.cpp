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
//   - Filtered search with a C callback.
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
#include "hnswlib.h"
#include <string>
#include <cstring>
#include <new>
#include <vector>

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
    hnsw_label next_label;                    ///< For auto label assignment (not used if user provides)
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
    (void)e; // unused
    return HNSW_ERR_INVALID_PARAM;
}

// ------------------------------------------------------------------
// Functor adapter for C callback (filter)
// ------------------------------------------------------------------

/**
 * @brief Adapter class that wraps a C function pointer and user data
 *        into a hnswlib::BaseFilterFunctor.
 */
class CFunctorFilter : public hnswlib::BaseFilterFunctor {
    hnsw_filter_func callback_;
    void* user_data_;

 public:
    CFunctorFilter(hnsw_filter_func cb, void* ud) : callback_(cb), user_data_(ud) {}

    virtual bool operator()(hnswlib::labeltype id) override {
        // Call the C callback; non-zero return means include the point.
        return callback_(static_cast<hnsw_label>(id), user_data_) != 0;
    }
};

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
HNSW_API HnswHandle hnsw_create(HnswSpace space, int dim, hnsw_size max_elements,
                                 int M, int ef_construction) {
    if (dim <= 0 || max_elements == 0 || M <= 0 || ef_construction <= 0)
        return nullptr;

    hnswlib::SpaceInterface<float>* space_ptr = nullptr;
    try {
        if (space == HNSW_SPACE_L2) {
            space_ptr = new hnswlib::L2Space(dim);
        } else if (space == HNSW_SPACE_IP) {
            space_ptr = new hnswlib::InnerProductSpace(dim);
        } else {
            return nullptr;
        }

        // Convert hnsw_size to size_t (safe on 64-bit; on 32-bit we rely on sizeof(size_t) <= sizeof(hnsw_size))
        hnswlib::HierarchicalNSW<float>* alg = new hnswlib::HierarchicalNSW<float>(
            space_ptr, static_cast<size_t>(max_elements), M, ef_construction);
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
static HnswError insert_impl(HnswHandle idx, const float* vector, hnsw_label label, bool replace_deleted) {
    if (!idx || !vector) return HNSW_ERR_INVALID_PARAM;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    try {
        internal->alg->addPoint(static_cast<const void*>(vector),
                                 static_cast<hnswlib::labeltype>(label),
                                 replace_deleted);
        return HNSW_OK;
    } catch (const std::exception&) {
        return HNSW_ERR_INVALID_PARAM;
    }
}

HNSW_API HnswError hnsw_insert(HnswHandle idx, const float* vector, hnsw_label label) {
    return insert_impl(idx, vector, label, false);
}

HNSW_API HnswError hnsw_insert_replace(HnswHandle idx, const float* vector, hnsw_label label) {
    return insert_impl(idx, vector, label, true);
}

HNSW_API HnswError hnsw_mark_deleted(HnswHandle idx, hnsw_label label) {
    if (!idx) return HNSW_ERR_INVALID_PARAM;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    try {
        internal->alg->markDelete(static_cast<hnswlib::labeltype>(label));
        return HNSW_OK;
    } catch (...) {
        return HNSW_ERR_NOT_FOUND;
    }
}

// ---------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------

/**
 * @brief Searches for the k nearest neighbors to a query vector (no filter).
 *
 * @param idx           Handle to index.
 * @param query_vector  Pointer to query float array.
 * @param k             Number of neighbors to retrieve.
 * @param ef_search     Size of dynamic candidate list during search.
 * @param out_labels    Output array (size at least k) for labels.
 * @param out_distances Output array (size at least k) for distances.
 * @return Number of neighbors actually found, or negative error code.
 */
HNSW_API int hnsw_search(HnswHandle idx, const float* query_vector, hnsw_size k,
                          int ef_search, hnsw_label* out_labels, float* out_distances) {
    if (!idx || !query_vector || k == 0 || !out_labels || !out_distances)
        return -HNSW_ERR_INVALID_PARAM;

    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    if (!alg) return -HNSW_ERR_INVALID_PARAM;

    int old_ef = alg->ef_;
    alg->ef_ = ef_search;

    auto result = alg->searchKnn(query_vector, static_cast<size_t>(k));
    size_t returned = result.size();
    if (returned > k) returned = k;

    // Convert max-heap (farthest first) to closest-first output.
    std::vector<std::pair<float, hnswlib::labeltype>> ordered;
    ordered.reserve(returned);
    while (!result.empty()) {
        ordered.push_back(result.top());
        result.pop();
    }
    for (size_t i = 0; i < returned; ++i) {
        out_labels[i] = static_cast<hnsw_label>(ordered[returned - 1 - i].second);
        out_distances[i] = ordered[returned - 1 - i].first;
    }

    alg->ef_ = old_ef;
    return static_cast<int>(returned);
}

/**
 * @brief Searches for the k nearest neighbors with a user-provided filter.
 *
 * @param idx           Handle to index.
 * @param query_vector  Pointer to query float array.
 * @param k             Number of neighbors to retrieve.
 * @param ef_search     Size of dynamic candidate list during search.
 * @param out_labels    Output array (size at least k) for labels.
 * @param out_distances Output array (size at least k) for distances.
 * @param filter        C callback function that returns non-zero to include a point.
 * @param user_data     Opaque pointer passed to the callback.
 * @return Number of neighbors found, or negative error code.
 */
HNSW_API int hnsw_search_filtered(HnswHandle idx, const float* query_vector, hnsw_size k,
                                   int ef_search, hnsw_label* out_labels, float* out_distances,
                                   hnsw_filter_func filter, void* user_data) {
    if (!idx || !query_vector || k == 0 || !out_labels || !out_distances)
        return -HNSW_ERR_INVALID_PARAM;

    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    if (!alg) return -HNSW_ERR_INVALID_PARAM;

    CFunctorFilter cpp_filter(filter, user_data);

    int old_ef = alg->ef_;
    alg->ef_ = ef_search;

    auto result = alg->searchKnn(query_vector, static_cast<size_t>(k), &cpp_filter);
    size_t returned = result.size();
    if (returned > k) returned = k;

    std::vector<std::pair<float, hnswlib::labeltype>> ordered;
    ordered.reserve(returned);
    while (!result.empty()) {
        ordered.push_back(result.top());
        result.pop();
    }
    for (size_t i = 0; i < returned; ++i) {
        out_labels[i] = static_cast<hnsw_label>(ordered[returned - 1 - i].second);
        out_distances[i] = ordered[returned - 1 - i].first;
    }

    alg->ef_ = old_ef;
    return static_cast<int>(returned);
}

HNSW_API void hnsw_set_ef(HnswHandle idx, int ef_search) {
    if (!idx) return;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    if (alg) alg->ef_ = ef_search;
}

HNSW_API int hnsw_get_ef(HnswHandle idx) {
    if (!idx) return -1;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    return alg ? alg->ef_ : -1;
}

// ---------------------------------------------------------------------
// Index persistence
// ---------------------------------------------------------------------

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

HNSW_API HnswError hnsw_resize(HnswHandle idx, hnsw_size new_max_elements) {
    if (!idx) return HNSW_ERR_INVALID_PARAM;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    if (!alg) return HNSW_ERR_INVALID_PARAM;
    try {
        alg->resizeIndex(static_cast<size_t>(new_max_elements));
        return HNSW_OK;
    } catch (...) {
        return HNSW_ERR_MEMORY;
    }
}

HNSW_API hnsw_size hnsw_get_current_count(HnswHandle idx) {
    if (!idx) return 0;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    return alg ? static_cast<hnsw_size>(alg->getCurrentElementCount()) : 0;
}

HNSW_API hnsw_size hnsw_get_max_elements(HnswHandle idx) {
    if (!idx) return 0;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    return alg ? static_cast<hnsw_size>(alg->max_elements_) : 0;
}

HNSW_API int hnsw_get_dim(HnswHandle idx) {
    if (!idx) return 0;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    return internal->dim;
}

} // extern "C"
