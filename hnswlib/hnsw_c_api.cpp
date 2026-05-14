#include "hnsw_c_api.h"
#include "hnswlib.h"   // include the hnswlib headers
#include <string>
#include <cstring>
#include <new>

// Internal structure holding the actual hnswlib index and its space.
struct HnswInternal {
    hnswlib::SpaceInterface<float>* space;
    hnswlib::AlgorithmInterface<float>* alg;
    int dim;
    hnswlib::labeltype next_label; // for automatic label assignment (not used if user provides)
    bool owns_space;

    HnswInternal(hnswlib::SpaceInterface<float>* sp, hnswlib::AlgorithmInterface<float>* a, int d)
        : space(sp), alg(a), dim(d), next_label(0), owns_space(true) {}
    ~HnswInternal() {
        if (owns_space && space) delete space;
        if (alg) delete alg;
    }
};

// Helper to convert HNSW error to our error codes.
static HnswError wrap_error(const std::exception& e) {
    // Simplistic mapping – you can refine.
    return HNSW_ERR_INVALID_PARAM;
}

// ---------------------------------------------------------------------
// Creation / Destruction
// ---------------------------------------------------------------------
extern "C" {

HNSW_API HnswHandle hnsw_create(HnswSpace space, int dim, size_t max_elements,
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

        hnswlib::HierarchicalNSW<float>* alg = new hnswlib::HierarchicalNSW<float>(
            space_ptr, max_elements, M, ef_construction);
        HnswInternal* internal = new HnswInternal(space_ptr, alg, dim);
        return static_cast<HnswHandle>(internal);
    } catch (...) {
        if (space_ptr) delete space_ptr;
        return nullptr;
    }
}

HNSW_API void hnsw_destroy(HnswHandle idx) {
    if (!idx) return;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    delete internal;
}

// ---------------------------------------------------------------------
// Insertion
// ---------------------------------------------------------------------
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

HNSW_API HnswError hnsw_insert(HnswHandle idx, const float* vector, size_t label) {
    return insert_impl(idx, vector, label, false);
}

HNSW_API HnswError hnsw_insert_replace(HnswHandle idx, const float* vector, size_t label) {
    return insert_impl(idx, vector, label, true);
}

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
HNSW_API int hnsw_search(HnswHandle idx, const float* query_vector, size_t k,
                          int ef_search, size_t* out_labels, float* out_distances) {
    if (!idx || !query_vector || k == 0 || !out_labels || !out_distances)
        return -HNSW_ERR_INVALID_PARAM;

    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    // Set ef_search
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    if (!alg) return -HNSW_ERR_INVALID_PARAM;

    int old_ef = alg->ef_;
    alg->ef_ = ef_search;

    auto result = alg->searchKnn(query_vector, k);
    size_t returned = result.size();
    if (returned > k) returned = k;

    // Fill arrays: result is max-heap, we need to pop from largest distance to smallest?
    // Actually searchKnn returns a max-heap (largest distance on top). We want smaller distance first.
    // We'll pop into a temporary vector.
    std::vector<std::pair<float, hnswlib::labeltype>> ordered;
    ordered.reserve(returned);
    while (!result.empty()) {
        ordered.push_back(result.top());
        result.pop();
    }
    // Now ordered[0] has largest distance, ordered[returned-1] has smallest.
    for (size_t i = 0; i < returned; ++i) {
        out_labels[i] = ordered[returned - 1 - i].second;
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
// Utility
// ---------------------------------------------------------------------
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

HNSW_API size_t hnsw_get_current_count(HnswHandle idx) {
    if (!idx) return 0;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    return alg ? alg->getCurrentElementCount() : 0;
}

HNSW_API size_t hnsw_get_max_elements(HnswHandle idx) {
    if (!idx) return 0;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    hnswlib::HierarchicalNSW<float>* alg = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(internal->alg);
    return alg ? alg->max_elements_ : 0;
}

HNSW_API int hnsw_get_dim(HnswHandle idx) {
    if (!idx) return 0;
    HnswInternal* internal = static_cast<HnswInternal*>(idx);
    return internal->dim;
}

} // extern "C"
