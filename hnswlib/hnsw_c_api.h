#ifndef HNSW_C_API_H
#define HNSW_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
    #ifdef HNSW_C_API_EXPORTS
        #define HNSW_API __declspec(dllexport)
    #else
        #define HNSW_API __declspec(dllimport)
    #endif
#else
    #define HNSW_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to HNSW index
typedef void* HnswHandle;

// Distance space types
typedef enum {
    HNSW_SPACE_L2 = 0,
    HNSW_SPACE_IP = 1   // inner product (cosine if vectors normalized)
} HnswSpace;

// Error codes
typedef enum {
    HNSW_OK = 0,
    HNSW_ERR_INVALID_PARAM = -1,
    HNSW_ERR_MEMORY = -2,
    HNSW_ERR_IO = -3,
    HNSW_ERR_NOT_FOUND = -4
} HnswError;

// Create a new HNSW index.
// space: L2 or IP
// dim: vector dimension
// max_elements: initial capacity (index can grow up to this; may be increased later)
// M: number of bi-directional links (typical 16)
// ef_construction: construction quality/speed trade-off (typical 200)
// Returns handle or NULL on error. Use hnsw_destroy to free.
HNSW_API HnswHandle hnsw_create(HnswSpace space, int dim, size_t max_elements,
                                 int M, int ef_construction);

// Destroy index and free all resources.
HNSW_API void hnsw_destroy(HnswHandle idx);

// Insert one vector with given label.
// label must be >= 0 and unique (if replace_deleted is false and label already exists,
// behavior depends on hnswlib: default throws; we'll implement replace_deleted = false).
// Returns 0 on success, negative error code on failure.
HNSW_API HnswError hnsw_insert(HnswHandle idx, const float* vector, size_t label);

// Insert with replace_deleted = true (allows reusing deleted labels).
HNSW_API HnswError hnsw_insert_replace(HnswHandle idx, const float* vector, size_t label);

// Mark a point as deleted (does not remove from index, but skipped in search).
// Returns 0 on success.
HNSW_API HnswError hnsw_mark_deleted(HnswHandle idx, size_t label);

// Search k nearest neighbors.
// query_vector: float array of dimension dim
// k: number of neighbors to return
// ef_search: search quality/speed trade-off (higher = slower but more accurate)
// out_labels: caller-allocated array of size k (will be filled with labels)
// out_distances: caller-allocated array of size k (will be filled with distances)
// Returns actual number of neighbors found (<= k), or negative error.
HNSW_API int hnsw_search(HnswHandle idx, const float* query_vector, size_t k,
                          int ef_search, size_t* out_labels, float* out_distances);

// Set ef_search parameter for subsequent searches.
HNSW_API void hnsw_set_ef(HnswHandle idx, int ef_search);

// Get current ef_search.
HNSW_API int hnsw_get_ef(HnswHandle idx);

// Save index to file.
HNSW_API HnswError hnsw_save(HnswHandle idx, const char* filepath);

// Load index from file. Space and dim must be known in advance.
// The loaded index will have the same space, dim, max_elements as when saved.
HNSW_API HnswHandle hnsw_load(const char* filepath, HnswSpace space, int dim);

// Resize index to a new maximum number of elements (must be >= current size).
HNSW_API HnswError hnsw_resize(HnswHandle idx, size_t new_max_elements);

// Get current number of elements in index.
HNSW_API size_t hnsw_get_current_count(HnswHandle idx);

// Get maximum elements (capacity).
HNSW_API size_t hnsw_get_max_elements(HnswHandle idx);

// Get dimension.
HNSW_API int hnsw_get_dim(HnswHandle idx);

#ifdef __cplusplus
}
#endif

#endif // HNSW_C_API_H
