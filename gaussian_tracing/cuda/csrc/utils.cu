#include "cuda_fp16.h"

__device__ __inline__ void atomicAdd4(float4 *address, float4 val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
    atomicAdd(&address->z, val.z);
    atomicAdd(&address->w, val.w);
}

__device__ __inline__ void atomicAdd3(float3 *address, float3 val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
    atomicAdd(&address->z, val.z);
}

__device__ int lcg_hash(int x) { return (1103515245 * x + 12345); }

__forceinline__ __device__ void
getRect(const float2 p, int max_radius, uint2 &rect_min, uint2 &rect_max) {
    rect_min = {
        (unsigned int)max((int)0, (int)((p.x - max_radius) / 16)),
        (unsigned int)max((int)0, (int)((p.y - max_radius) / 16))};
    rect_max = {
        (unsigned int)max((int)0, (int)((p.x + max_radius + 16 - 1) / 16)),
        (unsigned int)max((int)0, (int)((p.y + max_radius + 16 - 1) / 16))};
}

__forceinline__ __device__ float ndc2Pix(float v, int S) {
    return ((v + 1.0) * S - 1.0) * 0.5;
}

template <typename T> __device__ void fill_array(T *arr, uint32_t size, T val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

__device__ unsigned int packFloats(const float &distance, const float &alpha) {
    return __float_as_uint(distance);
}

__device__ float unpackDistance(const float &packed) {
    return __uint_as_float(packed);
}

__device__ float unpackAlpha(const float &packed) {
    return __uint_as_float(packed);
}
// #endif

#ifdef SORT_BY_COUNTING
__device__ unsigned int packId(const uint32_t gaussian_id, uint32_t count) {
    // store the last 4 bits of count into the 4 most significant bits of
    // gaussian_id
    return (gaussian_id << 4) | (count & 0xF);
}

__device__ uint32_t unpackId(const uint32_t &packed) { return packed >> 4; }

__device__ uint32_t unpackCount(const uint32_t &packed) { return packed & 0xF; }
#else
__device__ unsigned int packId(const uint32_t gaussian_id, uint32_t count) {
    // store the last 4 bits of count into the 4 most significant bits of
    // gaussian_id
    return gaussian_id;
}

__device__ uint32_t unpackId(const uint32_t &packed) { return packed; }

__device__ uint32_t unpackCount(const uint32_t &packed) { return 0; }
#endif
