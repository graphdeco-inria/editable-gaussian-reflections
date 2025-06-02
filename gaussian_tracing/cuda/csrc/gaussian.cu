

__device__ __host__ float
compute_scaling_factor(float opacity, float alpha_threshold, float exp_power) {
#if USE_EPANECHNIKOV_KERNEL == true
    return 1.0f;
#endif
#if DYN_CLAMPING == true
#if EXP_POWER == 1
    return max(
        MIN_SCALING_FACTOR,
        opacity <= alpha_threshold
            ? 0.0
            : sqrtf(2.0f * log(opacity / alpha_threshold)));
#else
    float k = 2.0f * exp_power;
    return max(
        MIN_SCALING_FACTOR,
        opacity <= alpha_threshold
            ? 0.0
            : powf(k * log(opacity / alpha_threshold), 1.0f / k));
    // return max(MIN_SCALING_FACTOR, opacity <= alpha_threshold ? 0.0 : powf(2
    // * log(opacity / alpha_threshold), 1.0f / 2^k));
#endif
#else
    return 1.0; // note: this clamps very agressively, just used for debugging
#endif
}
// exp(-x^2/2) = y
// x = +- sqrt(2 * log(y))
// exp(-x^k/k) = y
// x = +- (k * log(y))^(1/k)

#if USE_EPANECHNIKOV_KERNEL == true
__device__ float eval_gaussian(float3 local_hit, float exp_power) {
    return 1.0 - powf(dot(local_hit, local_hit), exp_power);
}
#else
__device__ float eval_gaussian(float3 local_hit, float exp_power) {
    float k = 2.0f * exp_power;
#if SQUARE_KERNEL == true
    float d = powf(fabs(local_hit.x), k) + powf(fabs(local_hit.y), k) +
              powf(fabs(local_hit.z), k);
    return exp(-d / k);
#else
    float d = dot(local_hit, local_hit);
    return exp(-powf(d, exp_power) / k);
#endif
}
#endif

__device__ float
compute_alpha(float guassval, float opacity, float alpha_threshold) {
#if ALLOW_OPACITY_ABOVE_1 == true
    float alpha = min(MAX_ALPHA, guassval * opacity);
#else
    float alpha = MAX_ALPHA * guassval * opacity;
#endif
#if ALPHA_RESCALE == true
    alpha = (alpha - alpha_threshold) / (1 - alpha_threshold);
#endif
#if ALPHA_SMOOTHING == true
    if (alpha < ALPHA_SMOOTHING_THRESHOLD) {
        alpha = (alpha - alpha_threshold) / (1 - alpha_threshold);
    }
#endif
    return alpha;
}

#if USE_LEVEL_OF_DETAIL == true
__device__ float compute_alpha_windowing(
    float blur_level, float gaussian_lod_mean, float guassian_lod_scale) {
    return max(
        0.0f,
        1.0 - powf(
                  abs(blur_level - gaussian_lod_mean) / guassian_lod_scale,
                  LOD_KERNEL_EXPONENT));
}
#endif
