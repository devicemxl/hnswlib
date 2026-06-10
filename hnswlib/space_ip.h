// ==================================================================
// hnswlib/space_ip.h
// ==================================================================
// Documentation for the InnerProduct space.
// This file defines distance functions for inner product (dot product)
// similarity, converted to a distance (1 - inner product) for use in
// hnswlib. It includes scalar and SIMD-optimized implementations
// (SSE, AVX, AVX512) with automatic dispatch based on CPU capabilities
// and vector dimension alignment.
//
// Note: The inner product is a similarity measure (higher = more similar).
//       To convert it into a distance suitable for nearest neighbor search
//       (where smaller is better), we return 1 - inner product. This yields
//       0 for identical vectors when they are unit vectors? Actually,
//       inner product can be >1, so distance may be negative; but HNSW
//       works with any metric where smaller is better. Cosine similarity
//       normalization is not applied here; that's in space_cosine.h.
//
// Author:   hnswlib contributors
// License:  Apache 2.0
// ==================================================================

#pragma once
#include "hnswlib.h"

namespace hnswlib {

// ------------------------------------------------------------------
// Scalar (non‑SIMD) implementations
// ------------------------------------------------------------------

/**
 * @brief Computes the inner product (dot product) of two float vectors.
 * @param pVect1 First vector.
 * @param pVect2 Second vector.
 * @param qty_ptr Pointer to a size_t containing the number of dimensions.
 * @return The inner product value.
 */
static float
InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
    }
    return res;
}

/**
 * @brief Converts inner product similarity to a distance (1 - inner product).
 * @param pVect1 First vector.
 * @param pVect2 Second vector.
 * @param qty_ptr Pointer to dimension count.
 * @return Distance value (1 - dot product).
 */
static float
InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}

// ------------------------------------------------------------------
// SIMD implementations (AVX, SSE, AVX512)
// ------------------------------------------------------------------
// The following functions process data in blocks of 16, 8, or 4 floats,
// taking advantage of vector registers. They fall back to scalar
// for residual dimensions.

#if defined(USE_AVX)

/**
 * @brief AVX implementation processing 4 floats at a time (i.e., 4-wide AVX,
 *        but actually using 256-bit registers, 8 floats). The name "SIMD4Ext"
 *        is legacy: it processes 8 floats per iteration, but the alignment
 *        is for 4 floats? Actually it uses __m256 (8 floats) and loops over
 *        8‑float chunks. The "4Ext" likely refers to handling dimensions
 *        multiple of 4.
 */
static float
InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m256 sum256 = _mm256_set1_ps(0);

    // Process 16 floats per iteration (two 8‑float AVX loads)
    while (pVect1 < pEnd1) {
        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    // Combine the two 128‑bit halves of the 256‑bit register
    __m128 v1, v2;
    __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    // Process remaining 4‑float blocks
    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    return sum;
}

static float
InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif // USE_AVX

#if defined(USE_SSE)

/**
 * @brief SSE implementation processing 4 floats at a time (128‑bit registers).
 *        Works for dimensions multiple of 4.
 */
static float
InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    // Process 16 floats per iteration (four SSE loads)
    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    // Process remaining 4‑float blocks
    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif // USE_SSE

#if defined(USE_AVX512)

/**
 * @brief AVX512 implementation processing 16 floats at a time (512‑bit registers).
 *        Uses fused multiply-add for efficiency.
 */
static float
InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    size_t loop = qty16 / 4;
    
    // Process 4 blocks of 16 floats (64 floats) per iteration
    while (loop--) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v3 = _mm512_loadu_ps(pVect1);
        __m512 v4 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v5 = _mm512_loadu_ps(pVect1);
        __m512 v6 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v7 = _mm512_loadu_ps(pVect1);
        __m512 v8 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
        sum512 = _mm512_fmadd_ps(v3, v4, sum512);
        sum512 = _mm512_fmadd_ps(v5, v6, sum512);
        sum512 = _mm512_fmadd_ps(v7, v8, sum512);
    }

    // Process remaining full 16‑float blocks
    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;
        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
    }

    float sum = _mm512_reduce_add_ps(sum512);
    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
}

#endif // USE_AVX512

#if defined(USE_AVX)

/**
 * @brief AVX implementation processing 16 floats at a time (two 8‑float chunks per iteration).
 *        This version is used when dimension is a multiple of 16.
 */
static float
InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif // USE_AVX

#if defined(USE_SSE)

/**
 * @brief SSE implementation processing 16 floats at a time (four SSE loads).
 *        Used when dimension is multiple of 16.
 */
static float
InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif // USE_SSE

// ------------------------------------------------------------------
// SIMD function pointers and residual handlers
// ------------------------------------------------------------------

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
// Global function pointers that will be set at runtime to the best available implementation.
static DISTFUNC<float> InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
static DISTFUNC<float> InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;

/**
 * @brief Handles dimensions that are not a multiple of 16 but >16.
 *        Computes full 16‑float blocks with SIMD, then processes the remaining
 *        dimensions (less than 16) with scalar code.
 */
static float
InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;   // largest multiple of 16 <= qty
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

/**
 * @brief Handles dimensions that are not a multiple of 4 but >4.
 *        Computes 4‑float blocks with SIMD, then processes residuals with scalar.
 */
static float
InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;   // largest multiple of 4 <= qty

    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}
#endif

// ------------------------------------------------------------------
// InnerProductSpace class
// ------------------------------------------------------------------

/**
 * @brief SpaceInterface implementation for inner product distance.
 *
 * This space computes distance as `1 - dot(a,b)`. It automatically selects
 * the fastest SIMD implementation based on:
 *   - CPU capabilities (AVX512, AVX, SSE) detected at runtime.
 *   - The dimensionality (alignment to 16, 4, or using residuals).
 *
 * The distance is not a true metric because it can be negative and does not
 * satisfy triangle inequality in general, but HNSW still works effectively
 * for many applications (e.g., recommendation systems).
 */
class InnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;  ///< Selected distance function (with SIMD if available)
    size_t data_size_;             ///< Size of one vector in bytes (dim * sizeof(float))
    size_t dim_;                   ///< Dimensionality

 public:
    /**
     * @brief Constructs an inner product space with given dimensionality.
     * @param dim Number of dimensions (must be >0).
     */
    InnerProductSpace(size_t dim) {
        fstdistfunc_ = InnerProductDistance;   // fallback scalar
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    // Runtime dispatch for best available instruction set
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

        // Choose the best function based on dimension alignment
        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    /**
     * @return Size in bytes of one vector.
     */
    size_t get_data_size() {
        return data_size_;
    }

    /**
     * @return Pointer to the distance function selected for this space.
     */
    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    /**
     * @return Pointer to the dimension (used as extra parameter for distance functions).
     */
    void *get_dist_func_param() {
        return &dim_;
    }

    ~InnerProductSpace() {}
};

}  // namespace hnswlib
