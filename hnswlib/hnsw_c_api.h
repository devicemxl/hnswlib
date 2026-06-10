// ==================================================================
// hnswlib/hnsw_c_api.h
// ==================================================================
// Documentation for the C API of hnswlib.
// This header defines a plain C interface to the HNSW index,
// allowing integration with languages that support C bindings
// (e.g., Python, C#, Rust, MATLAB). The implementation is in
// hnsw_c_api.cpp.
//
// The API provides:
//   - Index creation with L2 or Inner Product distance.
//   - Insertion (standard and with replacement of deleted entries).
//   - Logical deletion (mark as deleted without removal).
//   - k‑NN search with configurable ef_search.
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
        #define HNSW_API __declspec(dllexport)   // Building the DLL
    #else
        #define HNSW_API __declspec(dllimport)   // Using the DLL
    #endif
#else
    // GCC/Clang: default visibility (exported)
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
// Opaque handle and type definitions
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
 * @return Opaque handle to the new index, or NULL on error (e.g., invalid parameters, allocation failure).
 *
 * @note The index starts empty. Use hnsw_insert() to add points.
 */
HNSW_API HnswHandle hnsw_create(HnswSpace space, int dim, size_t max_elements,
                                 int M, int ef_construction);

/**
 * @brief Destroys an HNSW index and frees all associated memory.
 *
 * @param idx Handle to the index (can be NULL, in which case nothing happens).
 *
 * @note After destruction, the handle is invalid and must not be used again.
 */
HNSW_API void hnsw_destroy(HnswHandle idx);

// ------------------------------------------------------------------
// Insertion and deletion
// ------------------------------------------------------------------

/**
 * @brief Inserts a point into the index.
 *
 * If the label already exists, the behavior depends on whether the index was
 * created with replacement of deleted elements allowed (via the constructor flag).
 * In the default configuration, inserting an existing label overwrites the point.
 * This function does NOT replace deleted entries (i.e., it will not reuse a deleted slot).
 *
 * @param idx    Handle to index.
 * @param vector Pointer to a float array of length `dim` (the dimensionality of the index).
 * @param label  External label for the point (must be unique if not replacing).
 * @return HNSW_OK on success, or an error code (e.g., HNSW_ERR_INVALID_PARAM, HNSW_ERR_MEMORY).
 *
 * @note The label is used to identify points in search results. Labels can be any size_t value.
 * @see hnsw_insert_replace() for version that may reuse deleted slots.
 */
HNSW_API HnswError hnsw_insert(HnswHandle idx, const float* vector, size_t label);

/**
 * @brief Inserts a point, potentially reusing a slot previously marked as deleted.
 *
 * This function behaves like hnsw_insert() but if the index was created with
 * `allow_replace_deleted = true`, it may overwrite a deleted entry instead of
 * increasing the total element count. This is useful for maintaining a fixed-size
 * cache or rolling dataset.
 *
 * @param idx    Handle to index.
 * @param vector Pointer to float array.
 * @param label  External label.
 * @return HNSW_OK on success, or an error code.
 *
 * @note To enable this feature, the index must have been constructed with
 *       `allow_replace_deleted = true`. The default C API constructor (hnsw_create)
 *       does NOT enable it; this function will fail unless you modify the
 *       implementation to set that flag.
 */
HNSW_API HnswError hnsw_insert_replace(HnswHandle idx, const float* vector, size_t label);

/**
 * @brief Logically deletes a point by its label.
 *
 * The point is marked as deleted and will be ignored in subsequent searches.
 * The space it occupies may be reused later if hnsw_insert_replace() is used
 * and the index was created with the appropriate flag.
 *
 * @param idx   Handle to index.
 * @param label Label of the point to delete.
 * @return HNSW_OK if the label was found and marked, HNSW_ERR_NOT_FOUND otherwise.
 *
 * @note This is a logical deletion; the graph structure is not altered immediately,
 *       so memory is not freed. To physically remove points, consider rebuilding the index.
 */
HNSW_API HnswError hnsw_mark_deleted(HnswHandle idx, size_t label);

// ------------------------------------------------------------------
// Search
// ------------------------------------------------------------------

/**
 * @brief Finds the k nearest neighbors to a query vector.
 *
 * The function temporarily sets the index's `ef_search` parameter to the provided value,
 * performs the search, and restores the original `ef_search`. This allows per‑query
 * quality adjustments without affecting concurrent searches (if the index is shared).
 *
 * @param idx           Handle to index.
 * @param query_vector  Pointer to query float array (length `dim`).
 * @param k             Number of neighbors to retrieve.
 * @param ef_search     Size of dynamic candidate list during search. Higher values
 *                      increase accuracy but reduce speed. Typical: 10–200.
 * @param out_labels    Caller‑allocated array of size at least `k`; will be filled with
 *                      the labels of the k nearest points (closest first).
 * @param out_distances Caller‑allocated array of size at least `k`; will be filled with
 *                      the corresponding distances (smallest first).
 * @return The actual number of neighbors found (could be less than `k` if the index
 *         contains fewer points), or a negative error code on failure.
 *
 * @note The output arrays are filled in order of increasing distance (closest first).
 * @note If the index has deleted points, they are excluded from results.
 */
HNSW_API int hnsw_search(HnswHandle idx, const float* query_vector, size_t k,
                          int ef_search, size_t* out_labels, float* out_distances);

/**
 * @brief Sets the default ef_search parameter for future searches.
 *
 * The ef parameter controls the size of the dynamic candidate list during search.
 * Higher values improve recall at the cost of speed.
 *
 * @param idx       Handle to index.
 * @param ef_search New ef value (must be >= k for typical use, but any positive integer).
 *
 * @note This does not affect searches that provide an explicit ef_search argument
 *       to hnsw_search() (the explicit argument overrides it for that call).
 */
HNSW_API void hnsw_set_ef(HnswHandle idx, int ef_search);

/**
 * @brief Gets the current default ef_search value.
 * @param idx Handle to index.
 * @return Current ef value, or -1 if the handle is invalid.
 */
HNSW_API int hnsw_get_ef(HnswHandle idx);

// ------------------------------------------------------------------
// Persistence
// ------------------------------------------------------------------

/**
 * @brief Saves the index to a binary file.
 *
 * The file format is architecture‑dependent (binary dump of internal structures).
 * It includes all metadata, the base layer, and upper layers.
 *
 * @param idx      Handle to index.
 * @param filepath Path to the output file (UTF‑8 on supported platforms).
 * @return HNSW_OK on success, HNSW_ERR_IO on file write error, or another error code.
 *
 * @note The saved index can be reloaded with hnsw_load() using the same space and dim.
 */
HNSW_API HnswError hnsw_save(HnswHandle idx, const char* filepath);

/**
 * @brief Loads an index from a binary file previously saved with hnsw_save().
 *
 * @param filepath Path to the input file.
 * @param space    Distance space that must match the saved index (L2 or IP).
 * @param dim      Dimensionality that must match the saved index.
 * @return Handle to the loaded index, or NULL on failure (e.g., file not found, corrupt, mismatch).
 *
 * @note The loaded index will have the same capacity (max_elements) as when saved.
 * @note The caller is responsible for calling hnsw_destroy() on the returned handle.
 */
HNSW_API HnswHandle hnsw_load(const char* filepath, HnswSpace space, int dim);

// ------------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------------

/**
 * @brief Changes the maximum capacity of the index.
 *
 * The new capacity must be at least the current number of elements.
 * Memory for the base layer and link lists is reallocated.
 *
 * @param idx              Handle to index.
 * @param new_max_elements New maximum number of elements.
 * @return HNSW_OK on success, HNSW_ERR_MEMORY if reallocation fails,
 *         HNSW_ERR_INVALID_PARAM if new_max_elements is too small.
 */
HNSW_API HnswError hnsw_resize(HnswHandle idx, size_t new_max_elements);

/**
 * @brief Returns the current number of elements stored in the index.
 *
 * This count includes points that have been marked as deleted.
 * To get the number of active (non‑deleted) points, subtract hnsw_get_deleted_count()
 * if that function were exposed (not yet in this API).
 *
 * @param idx Handle to index.
 * @return Number of elements (including deleted), or 0 if handle invalid.
 */
HNSW_API size_t hnsw_get_current_count(HnswHandle idx);

/**
 * @brief Returns the maximum capacity (maximum number of elements) of the index.
 * @param idx Handle to index.
 * @return Maximum capacity, or 0 if handle invalid.
 */
HNSW_API size_t hnsw_get_max_elements(HnswHandle idx);

/**
 * @brief Returns the dimensionality of vectors stored in the index.
 * @param idx Handle to index.
 * @return Dimension (positive integer), or 0 if handle invalid.
 */
HNSW_API int hnsw_get_dim(HnswHandle idx);

// (Optional) Additional functions such as hnsw_get_deleted_count() or
// hnsw_unmark_deleted() could be added in future versions.

#ifdef __cplusplus
}
#endif

#endif // HNSW_C_API_H
