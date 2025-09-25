#pragma once

// * Helpers that auto-apply the correct activation for gaussian parameters.
// Kept in a separate file to avoid circular dependencies.

#include "../params.h"
#include "activations.cu"

// Read helpers
__device__ inline auto read_opacity(const Params &params, int gaussian_id) {
    return sigmoid_act(params.gaussians.opacity[gaussian_id]);
}
__device__ inline auto read_scale(const Params &params, int gaussian_id) {
    return exp_act(params.gaussians.scale[gaussian_id]);
}
__device__ inline auto read_mean(const Params &params, int gaussian_id) {
    return identity_act(params.gaussians.mean[gaussian_id]);
}
__device__ inline auto read_rotation(const Params &params, int gaussian_id) {
    return normalize_act(params.gaussians.rotation[gaussian_id]);
}
__device__ inline auto read_rgb(const Params &params, int gaussian_id) {
    return relu_act(params.gaussians.rgb[gaussian_id]);
}
__device__ inline auto read_normal(const Params &params, int gaussian_id) {
    return identity_act(params.gaussians.normal[gaussian_id]);
}
__device__ inline auto read_f0(const Params &params, int gaussian_id) {
    return clipped_relu_act(params.gaussians.f0[gaussian_id]);
}
__device__ inline auto read_roughness(const Params &params, int gaussian_id) {
    return clipped_relu_act(params.gaussians.roughness[gaussian_id]);
}

// Backprop helpers
__device__ inline auto backward_act_for_opacity(auto dL_dvalue, auto value) {
    return backward_sigmoid_act(dL_dvalue, value);
}
__device__ inline auto backward_act_for_scale(auto dL_dvalue, auto value) {
    return backward_exp_act(dL_dvalue, value);
}
__device__ inline auto backward_act_for_mean(auto dL_dvalue, auto value) {
    return backward_identity_act(dL_dvalue, value);
}
__device__ inline auto backward_act_for_rotation(auto dL_dvalue, auto value) {
    return backward_normalize_act(dL_dvalue, value);
}
__device__ inline auto backward_act_for_rgb(auto dL_dvalue, auto value) {
    return backward_relu_act(dL_dvalue, value);
}
__device__ inline auto backward_act_for_normal(auto dL_dvalue, auto value) {
    return backward_identity_act(dL_dvalue, value);
}
__device__ inline auto backward_act_for_f0(auto dL_dvalue, auto value) {
    return backward_clipped_relu_act(dL_dvalue, value);
}
__device__ inline auto backward_act_for_roughness(auto dL_dvalue, auto value) {
    return backward_clipped_relu_act(dL_dvalue, value);
}
__device__ inline auto backward_act_for_depth(auto dL_dvalue, auto value) {
    return backward_identity_act(dL_dvalue, value);
}
