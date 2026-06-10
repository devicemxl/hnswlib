// ==================================================================
// hnswlib/hnsw_c_api.h
// ==================================================================
// Documentation for the C API of hnswlib.
// This header defines a plain C interface to the HNSW index,
// allowing integration with languages that support C bindings
// (e.g., Python, C#, Rust, MATLAB, Odin). The implementation is in
// hnsw_c_api.cpp.
//
// The API provides:
//   - Index creation with L2 or Inner Product distance.
//   - Insertion (standard and with replacement of deleted entries).
//   - Logical deletion (mark as deleted without removal).
//   - k‑NN search with configurable ef_search.
//   - Filtered search using a C callback.
//   - Persistence (save/load to/from binary files).
//   - Resizing and status queries.
//
// Thread safety: The underlying C++ index is thread‑safe for
// concurrent searches and concurrent inserts, but mixing inserts
// and searches concurrently is not safe. The caller must manage
// synchronization if needed.
//
// Author:   hnswlib contributors (C API wrapper)
// License:  Apache 2.0
// ==================================================================

#ifndef HNSW_C_API_H
#define HNSW_C_API_H

#include <stdint.h>
#include <stddef.h>

// ------------------------------------------------------------------
// Platform-specific symbol visibility (Windows DLL export/import)
// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
// Fixed-width integer types for ABI stability across platforms
// ------------------------------------------------------------------

/**
 * @brief Type for external labels (identifiers). Always 64-bit.
 */
typedef uint64_t hnsw_label;

/**
 * @brief Type for sizes (number of elements, capacities, etc.). Always 64-bit.
 */
typedef uint64_t hnsw_size;

// ------------------------------------------------------------------
// Opaque handle and basic enums
// ------------------------------------------------------------------

/**
 * @brief Opaque handle to an HNSW index.
 *
 * This handle is returned by hnsw_create() and hnsw_load(), and
 * must be passed to all other functions. It is not meant to be
 * dereferenced directly by C clients.
 */
typedef void* HnswHandle;

/**
 * @brief Supported distance spaces.
 */
typedef enum {
    HNSW_SPACE_L2 = 0,   ///< Squared Euclidean (L2) distance.
    HNSW_SPACE_IP = 1    ///< Inner product distance: 1 - dot(a,b). If vectors are normalized, this is cosine distance.
} HnswSpace;

/**
 * @brief Error codes returned by most functions.
 */
typedef enum {
    HNSW_OK = 0,                    ///< Operation succeeded.
    HNSW_ERR_INVALID_PARAM = -1,    ///< Invalid argument (e.g., null handle, dimension zero).
    HNSW_ERR_MEMORY = -2,           ///< Memory allocation failed.
    HNSW_ERR_IO = -3,               ///< File I/O error (save/load).
    HNSW_ERR_NOT_FOUND = -4         ///< Label not found (mark delete on nonexistent label).
} HnswError;

// ------------------------------------------------------------------
// Filter callback type (for search with filtering)
// ------------------------------------------------------------------

/**
 * @brief Filter callback to exclude points during search.
 *
 * @param label      The label of the candidate point.
 * @param user_data  User-provided pointer (e.g., a set of allowed labels).
 * @return Non-zero to include the point, zero to exclude it.
 *
 * @note The callback is called for each candidate that reaches the top candidate list.
 *       It should be fast and thread-safe if used concurrently.
 */
typedef int (*hnsw_filter_func)(hnsw_label label, void* user_data);

// ------------------------------------------------------------------
// Index lifecycle
// ------------------------------------------------------------------

/**
 * @brief Creates a new empty HNSW index.
 *
 * @param space           Distance space (L2 or IP).
 * @param dim             Dimensionality of vectors (must be > 0).
 * @param max_elements    Initial capacity (maximum number of points the index can hold).
 *                        Can be increased later with hnsw_resize().
 * @param M               Number of bidirectional links per node (except level0).
 *                        Typical values: 16, 32, 64. Higher M increases accuracy and memory.
 * @param ef_construction Size of dynamic candidate list during construction.
 *                        Typical values: 100–200. Higher values improve quality but slow construction.
 * @return Opaque handle to the new index, or NULL on error.
 */
HNSW_API HnswHandle hnsw_create(HnswSpace space, int dim, hnsw_size max_elements,
                                 int M, int ef_construction);

/**
 * @brief Destroys an HNSW index and frees all associated memory.
 * @param idx Handle to the index (can be NULL).
 */
HNSW_API void hnsw_destroy(HnswHandle idx);

// ------------------------------------------------------------------
// Insertion and deletion
// ------------------------------------------------------------------

/**
 * @brief Inserts a point into the index (does not replace deleted ones).
 * @param idx    Handle to index.
 * @param vector Pointer to a float array of length `dim`.
 * @param label  External label.
 * @return HNSW_OK on success, error otherwise.
 */
HNSW_API HnswError hnsw_insert(HnswHandle idx, const float* vector, hnsw_label label);

/**
 * @brief Inserts a point, potentially reusing a slot previously marked as deleted.
 * @param idx    Handle to index.
 * @param vector Pointer to float array.
 * @param label  External label.
 * @return HNSW_OK on success, error otherwise.
 */
HNSW_API HnswError hnsw_insert_replace(HnswHandle idx, const float* vector, hnsw_label label);

/**
 * @brief Logically deletes a point by its label.
 * @param idx   Handle to index.
 * @param label Label of the point to delete.
 * @return HNSW_OK if found and marked, HNSW_ERR_NOT_FOUND otherwise.
 */
HNSW_API HnswError hnsw_mark_deleted(HnswHandle idx, hnsw_label label);

// ------------------------------------------------------------------
// Search
// ------------------------------------------------------------------

/**
 * @brief Finds the k nearest neighbors to a query vector (no filter).
 * @param idx           Handle to index.
 * @param query_vector  Pointer to query float array.
 * @param k             Number of neighbors to retrieve.
 * @param ef_search     Size of dynamic candidate list.
 * @param out_labels    Output array for labels (size at least k).
 * @param out_distances Output array for distances (size at least k).
 * @return Number of neighbors actually found, or negative error code.
 */
HNSW_API int hnsw_search(HnswHandle idx, const float* query_vector, hnsw_size k,
                          int ef_search, hnsw_label* out_labels, float* out_distances);

/**
 * @brief Finds the k nearest neighbors with a filter callback.
 * @param idx           Handle to index.
 * @param query_vector  Pointer to query float array.
 * @param k             Number of neighbors to retrieve.
 * @param ef_search     Size of dynamic candidate list.
 * @param out_labels    Output array for labels.
 * @param out_distances Output array for distances.
 * @param filter        C callback function (return non-zero to include point).
 * @param user_data     Opaque pointer passed to the callback.
 * @return Number of neighbors found, or negative error code.
 */
HNSW_API int hnsw_search_filtered(HnswHandle idx, const float* query_vector, hnsw_size k,
                                   int ef_search, hnsw_label* out_labels, float* out_distances,
                                   hnsw_filter_func filter, void* user_data);

/**
 * @brief Sets the default ef_search parameter for future searches.
 * @param idx       Handle to index.
 * @param ef_search New ef value.
 */
HNSW_API void hnsw_set_ef(HnswHandle idx, int ef_search);

/**
 * @brief Gets the current default ef_search value.
 * @param idx Handle to index.
 * @return Current ef value, or -1 on error.
 */
HNSW_API int hnsw_get_ef(HnswHandle idx);

// ------------------------------------------------------------------
// Persistence
// ------------------------------------------------------------------

/**
 * @brief Saves the index to a binary file.
 * @param idx      Handle to index.
 * @param filepath Path to the output file.
 * @return HNSW_OK on success, HNSW_ERR_IO on error.
 */
HNSW_API HnswError hnsw_save(HnswHandle idx, const char* filepath);

/**
 * @brief Loads an index from a binary file.
 * @param filepath Path to the input file.
 * @param space    Distance space (must match saved index).
 * @param dim      Dimensionality (must match saved index).
 * @return Handle to loaded index, or NULL on failure.
 */
HNSW_API HnswHandle hnsw_load(const char* filepath, HnswSpace space, int dim);

// ------------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------------

/**
 * @brief Changes the maximum capacity of the index.
 * @param idx              Handle to index.
 * @param new_max_elements New maximum number of elements.
 * @return HNSW_OK on success, error code otherwise.
 */
HNSW_API HnswError hnsw_resize(HnswHandle idx, hnsw_size new_max_elements);

/**
 * @brief Returns the current number of elements (including deleted).
 * @param idx Handle to index.
 * @return Element count, or 0 if handle invalid.
 */
HNSW_API hnsw_size hnsw_get_current_count(HnswHandle idx);

/**
 * @brief Returns the maximum capacity of the index.
 * @param idx Handle to index.
 * @return Maximum number of elements, or 0 if handle invalid.
 */
HNSW_API hnsw_size hnsw_get_max_elements(HnswHandle idx);

/**
 * @brief Returns the dimensionality of vectors stored in the index.
 * @param idx Handle to index.
 * @return Dimension, or 0 if handle invalid.
 */
HNSW_API int hnsw_get_dim(HnswHandle idx);

#ifdef __cplusplus
}
#endif

#endif // HNSW_C_API_H
