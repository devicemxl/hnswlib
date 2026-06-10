// ==================================================================
// hnswlib/space_l2.h
// ==================================================================
// Documentation for the L2 (Euclidean) space.
// This file implements squared L2 distance (sum of squared differences)
// for floating‑point and unsigned byte vectors. It includes scalar,
// SSE, AVX, and AVX512 implementations with runtime dispatch based
// on CPU capabilities and dimension alignment.
//
// Note: The distance is squared Euclidean (no square root), which
//       preserves ranking and is faster to compute.
//
// Author:   hnswlib contributors
// License:  Apache 2.0
// ==================================================================

#pragma once
#include "hnswlib.h"

namespace hnswlib {

// ------------------------------------------------------------------
// Scalar (non‑SIMD) L2 squared distance for float vectors
// ------------------------------------------------------------------

/**
 * @brief Computes squared Euclidean distance between two float vectors.
 * @param pVect1v First vector.
 * @param pVect2v Second vector.
 * @param qty_ptr Pointer to dimension count (size_t).
 * @return Squared L2 distance: Σ (a_i - b_i)^2
 */
static float
L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

// ------------------------------------------------------------------
// SIMD implementations (AVX512, AVX, SSE)
// ------------------------------------------------------------------

#if defined(USE_AVX512)

/**
 * @brief AVX512 implementation: processes 16 floats per iteration.
 *        Uses 512‑bit registers and reduction at the end.
 */
static float
L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;          // number of 16‑float blocks

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);  // commented in original, but uses mul+add
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}
#endif

#if defined(USE_AVX)

/**
 * @brief AVX implementation: processes 16 floats per iteration (two 8‑float chunks).
 */
static float
L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#endif

#if defined(USE_SSE)

/**
 * @brief SSE implementation: processes 16 floats per iteration (four 4‑float chunks).
 */
static float
L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
#endif

// ------------------------------------------------------------------
// SIMD function pointers and residual handlers for float L2
// ------------------------------------------------------------------

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
// Global pointer to the best available 16‑float SIMD implementation
static DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

/**
 * @brief Handles dimensions that are not a multiple of 16.
 *        Processes complete 16‑float blocks with SIMD, then scalar for the rest.
 */
static float
L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;   // largest multiple of 16 <= qty
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}
#endif

#if defined(USE_SSE)

/**
 * @brief SSE implementation that processes 4 floats at a time (for dimensions multiple of 4).
 */
static float
L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty4 = qty >> 2;   // number of 4‑float blocks

    const float *pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

/**
 * @brief Handles dimensions not multiple of 4 but >4.
 *        Uses SIMD for 4‑float blocks and scalar for residuals.
 */
static float
L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif

// ------------------------------------------------------------------
// L2Space for float vectors
// ------------------------------------------------------------------

/**
 * @brief L2 (squared Euclidean) distance space for float vectors.
 *
 * This class implements SpaceInterface<float> and automatically selects
 * the fastest distance function based on:
 *   - CPU capabilities (AVX512, AVX, SSE) detected at runtime.
 *   - The dimensionality (alignment to 16, 4, or using residual fallback).
 */
class L2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;   ///< Selected distance function
    size_t data_size_;              ///< Bytes per vector (dim * sizeof(float))
    size_t dim_;                    ///< Dimensionality

 public:
    /**
     * @brief Constructs an L2 space with given dimension.
     * @param dim Number of dimensions (must be >0).
     */
    L2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;   // fallback scalar
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    // Runtime dispatch for best instruction set
    #if defined(USE_AVX512)
        if (AVX512Capable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        else if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #endif

        // Choose function based on dimension alignment
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
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2Space() {}
};

// ------------------------------------------------------------------
// L2 distance for unsigned byte vectors (integer)
// ------------------------------------------------------------------

/**
 * @brief Squared L2 distance for byte vectors, processing 4 bytes per iteration.
 *        Used when dimension is a multiple of 4.
 * @param pVect1 First byte vector.
 * @param pVect2 Second byte vector.
 * @param qty_ptr Pointer to dimension count (in bytes).
 * @return Squared distance as integer.
 */
static int
L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    int res = 0;
    unsigned char *a = (unsigned char *) pVect1;
    unsigned char *b = (unsigned char *) pVect2;

    qty = qty >> 2;   // number of 4‑byte blocks
    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

/**
 * @brief Scalar squared L2 distance for byte vectors (one byte at a time).
 * @param pVect1 First byte vector.
 * @param pVect2 Second byte vector.
 * @param qty_ptr Pointer to dimension count.
 * @return Squared distance as integer.
 */
static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    int res = 0;
    unsigned char* a = (unsigned char*)pVect1;
    unsigned char* b = (unsigned char*)pVect2;

    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

// ------------------------------------------------------------------
// L2SpaceI for unsigned byte vectors
// ------------------------------------------------------------------

/**
 * @brief L2 (squared Euclidean) distance space for unsigned byte vectors.
 *
 * This class is used when data are stored as bytes (e.g., image pixels).
 * It computes integer squared distances, which can be faster and memory‑efficient.
 * The distance function automatically uses 4‑byte unrolling if dimension is
 * a multiple of 4, otherwise falls back to scalar byte‑by‑byte.
 */
class L2SpaceI : public SpaceInterface<int> {
    DISTFUNC<int> fstdistfunc_;   ///< Selected distance function (int return)
    size_t data_size_;            ///< Bytes per vector (dim * sizeof(unsigned char))
    size_t dim_;                  ///< Dimensionality

 public:
    /**
     * @brief Constructs an integer L2 space for byte vectors.
     * @param dim Number of dimensions (each dimension is one unsigned char).
     */
    L2SpaceI(size_t dim) {
        if (dim % 4 == 0) {
            fstdistfunc_ = L2SqrI4x;      // faster, process 4 bytes at a time
        } else {
            fstdistfunc_ = L2SqrI;        // scalar fallback
        }
        dim_ = dim;
        data_size_ = dim * sizeof(unsigned char);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<int> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2SpaceI() {}
};

}  // namespace hnswlib
