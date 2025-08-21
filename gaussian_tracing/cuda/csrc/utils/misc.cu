#pragma once

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

template <typename T> __device__ void fill_array(T *arr, uint32_t size, T val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}
