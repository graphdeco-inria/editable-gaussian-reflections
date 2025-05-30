#ifndef TONEMAP_CUH
#define TONEMAP_CUH

#include <cmath>
#include <cuda_runtime.h>

constexpr float tonemapping_gamma = 1.3;

__device__ float tonemap(float x) {
    return powf(
        ((x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)),
        tonemapping_gamma);
}

__device__ float tonemap_grad(float x) {
    float num = x * (6.2 * x + 0.5);
    float denum = x * (6.2 * x + 1.7) + 0.06;
    float dl_dnum = (6.2 * x + 0.5) + 6.2 * x;
    float dl_ddenum = ((6.2 * x + 1.7) + 6.2 * x) / -powf(denum, 2.0f);
    return tonemapping_gamma * (dl_dnum / denum + num * dl_ddenum) *
           powf(num / denum, tonemapping_gamma - 1.0f);
}

__device__ float3 tonemap(float3 v) {
    return make_float3(tonemap(v.x), tonemap(v.y), tonemap(v.z));
}

__device__ float3 tonemap_grad(float3 v) {
    return make_float3(tonemap_grad(v.x), tonemap_grad(v.y), tonemap_grad(v.z));
}

#endif