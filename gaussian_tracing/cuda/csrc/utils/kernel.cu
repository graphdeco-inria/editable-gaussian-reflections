#pragma once

__device__ __host__ float
compute_scaling_factor(float opacity, float alpha_threshold, float exp_power) {
    float k = 2.0f * exp_power;
    return opacity <= alpha_threshold ? 0.0 : powf(k * log(opacity / alpha_threshold), 1.0f / k);
}

__device__ float eval_gaussian(float3 local_hit, float exp_power) {
    float k = 2.0f * exp_power;
    float d = dot(local_hit, local_hit);
    return exp(-powf(d, exp_power) / k);
}

__device__ float compute_alpha(float guassval, float opacity, float alpha_threshold) {
    return MAX_ALPHA * guassval * opacity;
}
