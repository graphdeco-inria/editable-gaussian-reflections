#include "vec_math.h"

__device__ float sigmoid_act(float x) { return 1.0f / (1.0f + exp(-x)); }
__device__ float3 sigmoid_act(float3 x) {
    return {sigmoid_act(x.x), sigmoid_act(x.y), sigmoid_act(x.z)};
}

__device__ float backward_sigmoid_act(float dL_dy, float y) {
    return dL_dy * y * (1.0f - y);
}
__device__ float3 backward_sigmoid_act(float3 dL_dy, float3 y) {
    return {
        backward_sigmoid_act(dL_dy.x, y.x),
        backward_sigmoid_act(dL_dy.y, y.y),
        backward_sigmoid_act(dL_dy.z, y.z)};
}

__device__ float softplus_act(float x) { return log(1.0f + exp(x)); }
__device__ float3 softplus_act(float3 x) {
    return {softplus_act(x.x), softplus_act(x.y), softplus_act(x.z)};
}

__device__ float abs_act(float x) { return abs(x); }
__device__ float backward_abs_act(float dL_dy, float x) {
    return dL_dy * (x < 0.0f ? -1.0f : 1.0f);
}

__device__ float backward_softplus_act(float dL_dy, float x, float y) {
    return dL_dy / (1.0f + exp(-x));
}
__device__ float3 backward_softplus_act(float3 dL_dy, float3 x, float3 y) {
    return {
        backward_softplus_act(dL_dy.x, x.x, y.x),
        backward_softplus_act(dL_dy.y, x.y, y.y),
        backward_softplus_act(dL_dy.z, x.z, y.z)};
}

// RELU

__device__ float relu_act(float x) { return max(0.0f, x); }

__device__ float backward_relu_act(float dL_dy, float y) {
    return dL_dy * (y >= 0.0f ? 1.0f : 0.0f);
}

__device__ float3 relu_act(float3 x) {
    return {relu_act(x.x), relu_act(x.y), relu_act(x.z)};
}

__device__ float3 backward_relu_act(float3 dL_dy, float3 y) {
    return {
        backward_relu_act(dL_dy.x, y.x),
        backward_relu_act(dL_dy.y, y.y),
        backward_relu_act(dL_dy.z, y.z)};
}

// CLIPPED RELU

__device__ float clipped_relu_act(float x) { return min(max(0.0f, x), 1.0f); }

__device__ float backward_clipped_relu_act(float dL_dy, float y) {
    return dL_dy * (y >= 0.0f && y <= 1.0f ? 1.0f : 0.0f);
}

__device__ float3 clipped_relu_act(float3 x) {
    return {
        clipped_relu_act(x.x), clipped_relu_act(x.y), clipped_relu_act(x.z)};
}

__device__ float3 backward_clipped_relu_act(float3 dL_dy, float3 y) {
    return {
        backward_clipped_relu_act(dL_dy.x, y.x),
        backward_clipped_relu_act(dL_dy.y, y.y),
        backward_clipped_relu_act(dL_dy.z, y.z)};
}

// EXP

__device__ float exp_act(float x) { return exp(x); }
__device__ float3 exp_act(float3 x) {
    return {exp_act(x.x), exp_act(x.y), exp_act(x.z)};
}

__device__ float backward_exp_act(float dL_dy, float y) { return dL_dy * y; }
__device__ float3 backward_exp_act(float3 dL_dy, float3 y) {
    return {
        backward_exp_act(dL_dy.x, y.x),
        backward_exp_act(dL_dy.y, y.y),
        backward_exp_act(dL_dy.z, y.z)};
}

__device__ float4 normalize_act(float4 x) {
    float norm = length(x);
    return {x.x / norm, x.y / norm, x.z / norm, x.w / norm};
}
__device__ float4 backward_normalize_act(float4 dL_dy, float4 x, float4 y) {
    return dot(dL_dy, x) * -x / powf(length(x), 3) + dL_dy / length(x);
}

// this doesn't change much
// __device__ float3 __intrinsic_read(const float3* ptr) {
//     float3 result;
//     result.x = __ldg(&ptr->x);
//     result.y = __ldg(&ptr->y);
//     result.z = __ldg(&ptr->z);
//     return result;
// }
// __device__ float __intrinsic_read(const float* ptr) {
//     return __ldg(ptr);
// }

#define READ_RGB(gaussian_id) params.gaussian_rgb[gaussian_id]
#define READ_OPACITY(gaussian_id)                                              \
    sigmoid_act(params.gaussian_opacity[gaussian_id])
#define READ_ROTATION(gaussian_id)                                             \
    normalize_act(params.gaussian_rotations[gaussian_id])
#define READ_SCALE(gaussian_id) exp_act(params.gaussian_scales[gaussian_id])
#define READ_MEAN(gaussian_id) params.gaussian_means[gaussian_id]

#define READ_LOD_MEAN(gaussian_id) params.gaussian_lod_mean[gaussian_id]
#define READ_LOD_SCALE(gaussian_id)                                            \
    exp_act(params.gaussian_lod_scale[gaussian_id])

#define READ_ROUGHNESS(gaussian_id)                                            \
    clipped_relu_act(params.gaussian_roughness[gaussian_id])
#define READ_F0(gaussian_id) clipped_relu_act(params.gaussian_f0[gaussian_id])
