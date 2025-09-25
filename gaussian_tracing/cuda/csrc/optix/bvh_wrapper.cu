#include <optix.h>
#include <optix_stubs.h>

#include "../core/all.h"
#include "../flags.h"
#include "../params.h"
#include "../utils/common.h"

__device__ void create_transform_matrix(
    const float4 rotation, const float3 scaling, const float3 position, float4 (&matrix)[3]) {
    float3 s = scaling;
    float r = rotation.x;
    float x = rotation.y;
    float y = rotation.z;
    float z = rotation.w;

    // * Normalize quaternion
    float norm = sqrtf(r * r + x * x + y * y + z * z);
    r /= norm;
    x /= norm;
    y /= norm;
    z /= norm;

    // * Populate matrix
    matrix[0] = make_float4(
        s.x * (1.f - 2.f * (y * y + z * z)),
        s.y * (2.f * (x * y - r * z)),
        s.z * (2.f * (x * z + r * y)),
        position.x);
    matrix[1] = make_float4(
        s.x * (2.f * (x * y + r * z)),
        s.y * (1.f - 2.f * (x * x + z * z)),
        s.z * (2.f * (y * z - r * x)),
        position.y);
    matrix[2] = make_float4(
        s.x * (2.f * (x * z - r * y)),
        s.y * (2.f * (y * z + r * x)),
        s.z * (1.f - 2.f * (x * x + y * y)),
        position.z);
}

__global__ void _populateBVH(
    OptixInstance *instances,
    OptixTraversableHandle gasHandle,
    int gaussian_count,
    Gaussians gaussians, // * Must copy here
    float alpha_threshold,
    float exp_power,
    float global_scale_factor) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= gaussians.count)
        return;

    instances[i].traversableHandle = gasHandle;
    instances[i].sbtOffset = 0;
    instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;

    float opacity = sigmoid_act(gaussians.opacity[i]);
    float scaling_factor = compute_scaling_factor(opacity, alpha_threshold, exp_power);

    float3 sizes = exp_act(gaussians.scale[i]);
    sizes = sizes * scaling_factor * global_scale_factor;

    instances[i].visibilityMask =
        scaling_factor > 0.0f && (sizes.x > 0.0f || sizes.y > 0.0f || sizes.z > 0.0f);

    create_transform_matrix(
        gaussians.rotation[i],
        sizes,
        gaussians.mean[i],
        reinterpret_cast<float4(&)[3]>(instances[i].transform));
}

void populateBVH(
    OptixInstance *instances,
    OptixTraversableHandle gasHandle,
    const Gaussians &gaussians,
    float alpha_threshold,
    float exp_power,
    float global_scale_factor) {
    _populateBVH<<<(gaussians.count + 31) / 32, 32>>>(
        instances,
        gasHandle,
        gaussians.count,
        gaussians,
        alpha_threshold,
        exp_power,
        global_scale_factor);
}
