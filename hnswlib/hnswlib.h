// ==================================================================
// hnswlib/hnswlib.h
// ==================================================================
// Documentation for the main header of hnswlib.
// This file aggregates core definitions, CPU feature detection,
// base classes, and includes for all major components.
// 
// Purpose:
//   - Provides common infrastructure for the HNSW algorithm and
//     brute‑force search.
//   - Defines interfaces for distance spaces and search algorithms.
//   - Offers runtime detection of SIMD instructions (SSE, AVX, AVX512).
//   - Allows customization of error output stream.
//
// Author:   hnswlib contributors
// License:  Apache 2.0
// ==================================================================

#pragma once

// ------------------------------------------------------------------
// Error stream customization
// ------------------------------------------------------------------
// Allows a user to redirect error messages (e.g., for RcppHNSW or GUI apps)
// by defining HNSWLIB_ERR_OVERRIDE before including this header.
// If not defined, std::cerr is used as default.
// https://github.com/nmslib/hnswlib/pull/508
#ifndef HNSWLIB_ERR_OVERRIDE
  #define HNSWERR std::cerr
#else
  #define HNSWERR HNSWLIB_ERR_OVERRIDE
#endif

// ------------------------------------------------------------------
// SIMD instruction detection (SSE, AVX, AVX512)
// ------------------------------------------------------------------
// This block enables vectorized distance computations when available.
// User can define NO_MANUAL_VECTORIZATION to disable all manual vectorization.
#ifndef NO_MANUAL_VECTORIZATION

// Determine architecture support for SSE/AVX/AVX512 using compiler macros.
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

// ------------------------------------------------------------------
// SIMD detection and helper functions for x86
// ------------------------------------------------------------------
#if defined(USE_AVX) || defined(USE_SSE)

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
// Windows: use __cpuidex and _xgetbv
static void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
    __cpuidex(out, eax, ecx);
}
static __int64 xgetbv(unsigned int x) {
    return _xgetbv(x);
}
#else
#include <x86intrin.h>
#include <cpuid.h>
#include <stdint.h>
// Linux / GCC / Clang: use __cpuid_count and inline assembly for xgetbv
static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}
static uint64_t xgetbv(unsigned int index) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t)edx << 32) | eax;
}
#endif

// Include AVX512 intrinsics if available
#if defined(USE_AVX512)
#include <immintrin.h>
#endif

// Portable alignment macros for stack variables (32-byte for AVX, 64-byte for AVX512)
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

// Adapted from https://github.com/Mysticial/FeatureDetector
// XCR register mask for XSAVE feature (used to check OS support for AVX)
#define _XCR_XFEATURE_ENABLED_MASK  0

/**
 * @brief Checks if the CPU and OS support AVX instructions.
 * @return true if AVX is available, false otherwise.
 */
static bool AVXCapable() {
    int cpuInfo[4];

    // CPU support: query leaf 0 for max basic leaf
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX = false;
    if (nIds >= 0x00000001) {
        cpuid(cpuInfo, 0x00000001, 0);
        HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
    }

    // OS support: check XGETBV for AVX enabled state
    cpuid(cpuInfo, 1, 0);
    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avxSupported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;   // bits 1 and 2 (XMM and YMM state)
    }
    return HW_AVX && avxSupported;
}

/**
 * @brief Checks if the CPU and OS support AVX512 instructions.
 * @return true if AVX512F (foundation) is available, false otherwise.
 */
static bool AVX512Capable() {
    if (!AVXCapable()) return false;

    int cpuInfo[4];

    // CPU support: leaf 0x00000007, subleaf 0, bit 16 indicates AVX512F
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX512F = false;
    if (nIds >= 0x00000007) {
        cpuid(cpuInfo, 0x00000007, 0);
        HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
    }

    // OS support: check XGETBV for AVX512 state (bits 1,2,5,6,7)
    cpuid(cpuInfo, 1, 0);
    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avx512Supported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;  // bits for opmask, ZMM0-15, ZMM16-31
    }
    return HW_AVX512F && avx512Supported;
}
#endif  // defined(USE_AVX) || defined(USE_SSE)

// ------------------------------------------------------------------
// Standard library includes used throughout hnswlib
// ------------------------------------------------------------------
#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

namespace hnswlib {

// ------------------------------------------------------------------
// Type definitions
// ------------------------------------------------------------------
/**
 * @brief Type used for point labels (external identifiers).
 * Typically size_t, but can be changed if needed.
 */
typedef size_t labeltype;

// ------------------------------------------------------------------
// BaseFilterFunctor : interface for custom filtering during search
// ------------------------------------------------------------------
/**
 * @brief Abstract functor for filtering points by label during search.
 * 
 * Users can derive from this class and override operator() to return
 * true for labels that should be considered in the result set.
 * Example: filter points that belong to a specific category.
 */
class BaseFilterFunctor {
 public:
    /**
     * @brief Determines whether a point with given label should be included.
     * @param id The label of the point.
     * @return true to include the point, false to exclude it.
     */
    virtual bool operator()(hnswlib::labeltype id) { return true; }
    virtual ~BaseFilterFunctor() {};
};

// ------------------------------------------------------------------
// BaseSearchStopCondition : advanced interface for early termination
// ------------------------------------------------------------------
/**
 * @brief Abstract class to define custom stopping conditions for search.
 * 
 * This interface allows fine‑grained control over search termination,
 * result pruning, and dynamic result set management. It is used by
 * the HNSW algorithm's internal search routine.
 * 
 * @tparam dist_t Type of distance values (float/double).
 */
template<typename dist_t>
class BaseSearchStopCondition {
 public:
    /**
     * @brief Called when a candidate point is added to the result set.
     * @param label      Label of the added point.
     * @param datapoint  Pointer to the vector data (optional).
     * @param dist       Distance from query to the point.
     */
    virtual void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    /**
     * @brief Called when a point is removed from the result set (e.g., pruned).
     * @param label      Label of the removed point.
     * @param datapoint  Pointer to the vector data.
     * @param dist       Distance of the removed point.
     */
    virtual void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    /**
     * @brief Determines whether the overall search should stop.
     * @param candidate_dist Distance of the current candidate being evaluated.
     * @param lowerBound     Current worst distance among results (or a threshold).
     * @return true to stop search, false to continue.
     */
    virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;

    /**
     * @brief Decides whether a candidate point should be considered for insertion.
     * @param candidate_dist Distance of the candidate.
     * @param lowerBound     Current worst distance among results.
     * @return true if the candidate could be added, false to skip it.
     */
    virtual bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) = 0;

    /**
     * @brief Indicates if the result set should be trimmed (kept at size k).
     * @return true if extra results should be removed.
     */
    virtual bool should_remove_extra() = 0;

    /**
     * @brief Post‑processes the candidate list before finalizing results.
     * @param candidates Reference to a vector of (distance, label) pairs.
     */
    virtual void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) = 0;

    virtual ~BaseSearchStopCondition() {}
};

// ------------------------------------------------------------------
// Helper comparator for max‑heap (priority_queue) over distance pairs
// ------------------------------------------------------------------
/**
 * @brief Functor for max‑heap ordering of (distance, label) pairs.
 * 
 * The default std::priority_queue is a max‑heap. For nearest neighbor search
 * we often need the smallest distances. This comparator inverts the order
 * so that the top of the heap contains the largest distance among the kept
 * results (i.e., the worst distance). This is used internally.
 * 
 * @tparam T A std::pair<dist_t, labeltype> or similar.
 */
template <typename T>
class pairGreater {
 public:
    bool operator()(const T& p1, const T& p2) {
        return p1.first > p2.first;
    }
};

// ------------------------------------------------------------------
// Binary I/O helpers for POD (Plain Old Data) types
// ------------------------------------------------------------------
/**
 * @brief Writes a POD variable to a binary output stream.
 * @tparam T A POD type (e.g., size_t, float).
 * @param out   Output stream (std::ofstream).
 * @param podRef Reference to the variable to write.
 */
template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

/**
 * @brief Reads a POD variable from a binary input stream.
 * @tparam T A POD type.
 * @param in    Input stream (std::ifstream).
 * @param podRef Reference where the read value will be stored.
 */
template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

// ------------------------------------------------------------------
// Distance function type definition
// ------------------------------------------------------------------
/**
 * @brief Pointer to a distance function.
 * 
 * The function takes three arguments: two vectors (a and b) and a context
 * parameter (e.g., dimension or scaling factor). Returns the distance as MTYPE.
 * 
 * @tparam MTYPE Usually float or double.
 * @param a First vector.
 * @param b Second vector.
 * @param param Extra parameter (e.g., dimension).
 * @return The computed distance.
 */
template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

// ------------------------------------------------------------------
// SpaceInterface : abstract distance space
// ------------------------------------------------------------------
/**
 * @brief Abstract interface for a distance space (metric or non‑metric).
 * 
 * Any concrete space (L2, Inner Product, Cosine, etc.) must provide:
 *   - The size in bytes of a single vector (data_size).
 *   - A distance function compatible with hnswlib.
 *   - Optional parameters for the distance function.
 * 
 * @tparam MTYPE Type used for distance values (float/double).
 */
template<typename MTYPE>
class SpaceInterface {
 public:
    /**
     * @return Size in bytes of one vector in this space.
     */
    virtual size_t get_data_size() = 0;

    /**
     * @return Pointer to the distance function for this space.
     */
    virtual DISTFUNC<MTYPE> get_dist_func() = 0;

    /**
     * @return Extra parameter (e.g., dimension) passed to the distance function.
     */
    virtual void *get_dist_func_param() = 0;

    virtual ~SpaceInterface() {}
};

// ------------------------------------------------------------------
// AlgorithmInterface : common operations for any index (HNSW, brute‑force)
// ------------------------------------------------------------------
/**
 * @brief Abstract interface that all search algorithms must implement.
 * 
 * @tparam dist_t Type of distance values (float/double).
 */
template<typename dist_t>
class AlgorithmInterface {
 public:
    /**
     * @brief Adds a point to the index.
     * @param datapoint      Pointer to vector data.
     * @param label          External identifier (must be unique).
     * @param replace_deleted If true, allows overwriting a deleted entry.
     */
    virtual void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) = 0;

    /**
     * @brief Returns k nearest neighbors as a max‑heap (farthest-first order).
     * 
     * The returned priority_queue contains pairs (distance, label). The top element
     * has the largest distance among the k results (the worst one). This format
     * is useful for incremental pruning in the HNSW search.
     * 
     * @param query_data     Pointer to query vector.
     * @param k              Number of neighbors to retrieve.
     * @param isIdAllowed    Optional filter (if non‑null, only points satisfying
     *                       the functor are considered).
     * @return Max‑heap of (distance, label) pairs.
     */
    virtual std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void*, size_t, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    /**
     * @brief Returns k nearest neighbors in the order of closest first.
     * 
     * This is a convenience method that calls searchKnn and then reverses the
     * order. The default implementation is provided in the .h file.
     * 
     * @param query_data     Pointer to query vector.
     * @param k              Number of neighbors.
     * @param isIdAllowed    Optional filter.
     * @return Vector of (distance, label) sorted from smallest distance to largest.
     */
    virtual std::vector<std::pair<dist_t, labeltype>>
        searchKnnCloserFirst(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const;

    /**
     * @brief Saves the index to a binary file.
     * @param location File path.
     */
    virtual void saveIndex(const std::string &location) = 0;

    virtual ~AlgorithmInterface(){
    }
};

// ------------------------------------------------------------------
// Default implementation of searchKnnCloserFirst
// ------------------------------------------------------------------
/**
 * @brief Converts the max‑heap result from searchKnn into a vector sorted
 *        by increasing distance (closest first).
 * 
 * Complexity: O(k log k) due to heap pop operations.
 */
template<typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k,
                                                 BaseFilterFunctor* isIdAllowed) const {
    std::vector<std::pair<dist_t, labeltype>> result;

    // searchKnn returns a max‑heap (farthest first)
    auto ret = searchKnn(query_data, k, isIdAllowed);
    {
        size_t sz = ret.size();
        result.resize(sz);
        while (!ret.empty()) {
            result[--sz] = ret.top();   // fill from the end
            ret.pop();
        }
    }

    return result;
}

}  // namespace hnswlib

// ------------------------------------------------------------------
// Include concrete implementations
// ------------------------------------------------------------------
// The following files provide:
//   - space_l2.h      : Euclidean (L2) distance space
//   - space_ip.h      : Inner product (IP) space (with optional normalization)
//   - stop_condition.h: Default search stopping condition (standard HNSW)
//   - bruteforce.h    : Exact brute‑force search class
//   - hnswalg.h       : HierarchicalNSW index implementation
#include "space_l2.h"
#include "space_ip.h"
#include "stop_condition.h"
#include "bruteforce.h"
#include "hnswalg.h"
