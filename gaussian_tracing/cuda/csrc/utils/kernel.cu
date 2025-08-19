#include "vec_math.h"

__device__ __host__ float
compute_scaling_factor(float opacity, float alpha_threshold, float exp_power) {
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
}
// exp(-x^2/2) = y
// x = +- sqrt(2 * log(y))
// exp(-x^k/k) = y
// x = +- (k * log(y))^(1/k)

__device__ float eval_gaussian(float3 local_hit, float exp_power) {
    float k = 2.0f * exp_power;
    float d = dot(local_hit, local_hit);
    return exp(-powf(d, exp_power) / k);
}

__device__ float
compute_alpha(float guassval, float opacity, float alpha_threshold) {
    float alpha = MAX_ALPHA * guassval * opacity;
    return alpha;
}
