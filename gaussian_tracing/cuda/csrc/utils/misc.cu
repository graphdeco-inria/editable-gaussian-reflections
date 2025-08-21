#pragma once

#include "cuda_fp16.h"

__device__ __inline__ void atomicAddX(float4 *address, float4 val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
    atomicAdd(&address->z, val.z);
    atomicAdd(&address->w, val.w);
}

__device__ __inline__ void atomicAddX(float3 *address, float3 val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
    atomicAdd(&address->z, val.z);
}

__device__ __inline__ void atomicAddX(float2 *address, float2 val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
}

__device__ __inline__ void atomicAddX(float *address, float val) {
    atomicAdd(address, val);
}

__device__ void fill_array(float *arr, uint32_t size, float val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

__device__ void fill_array(float2 *arr, uint32_t size, float2 val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

__device__ void fill_array(float3 *arr, uint32_t size, float3 val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

__device__ void fill_array(float4 *arr, uint32_t size, float4 val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

template <typename T> __device__ void fill_array(T *arr, uint32_t size, T val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

__device__ float dot(float a, float b) {
    return a * b; // * convenience function for generated code
}

__device__ float3 sign(float3 v) {
    return make_float3(
        copysignf(1.0f, v.x), copysignf(1.0f, v.y), copysignf(1.0f, v.z));
}

__device__ float sign(float v) { return copysignf(1.0f, v); }