#include <optix.h>
#include <optix_stubs.h>

#include "../flags.h"
#include "../params.h"
#include "../utils/activations.cu"
#include "../utils/kernel.cu"

__device__ void create_transform_matrix(
    const float4 rotation,
    const float3 scaling,
    const float3 position,
    float4 (&matrix)[3]) {
    float3 s = scaling;
    float r = rotation.x;
    float x = rotation.y;
    float y = rotation.z;
    float z = rotation.w;

#if ACTIVATION_IN_CUDA == true
    // normalize quaternion
    float norm = sqrtf(r * r + x * x + y * y + z * z);
    r /= norm;
    x /= norm;
    y /= norm;
    z /= norm;
#endif

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
    int num_gaussians,
    float3 *camera_position_world,
    float3 *scales,
    float4 *rotations,
    float3 *means,
    float *opacities,
    float *lod_means,
    float *lod_scales,
    bool *mask,
    float global_scaling_factor,
    float alpha_threshold,
#if OPTIMIZE_EXP_POWER == true
    float *exp_power
#else
    float exp_power
#endif
) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < num_gaussians) {
        instances[i].traversableHandle = gasHandle;
        // instances[i].instanceId = i; // this does not get used in practice
        instances[i].sbtOffset = 0;
        instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;

        float opacity = opacities[i];
#if ACTIVATION_IN_CUDA == true
        opacity = sigmoid_act(opacity);
#endif

#if COMPATIBILITY_MODE == true
        float3 sizes = scales[i];
#if ACTIVATION_IN_CUDA == true
        sizes = exp_act(sizes);
#endif
        sizes = sizes * 3.0;
        if (ANTIALIASING > 0.0f) {
            sizes += ANTIALIASING * 2.0f / 2048.0f; // hacky approximation
        }
        instances[i].visibilityMask = 255;
#else
        float3 sizes = scales[i];
#if ACTIVATION_IN_CUDA == true
        sizes = exp_act(sizes);
#endif
// ????????? shouldn't clamping be after antialiasing?
#if OPTIMIZE_EXP_POWER == true
        float p = exp_power[i];
#else
        float p = exp_power;
#endif
        float scaling_factor =
            compute_scaling_factor(opacity, alpha_threshold, p);
        sizes = sizes * scaling_factor * global_scaling_factor;
        instances[i].visibilityMask =
            scaling_factor >
            0.0f; // & mask[i]; // todo & product of scales also > 0

#if IGNORE_0_VOLUME_GAUSSIANS == true
        if (sizes.x < IGNORE_0_VOLUME_MINSIZE ||
            sizes.y < IGNORE_0_VOLUME_MINSIZE ||
            sizes.z < IGNORE_0_VOLUME_MINSIZE) {
            instances[i].visibilityMask = 0;
        }
#endif

        sizes *= BB_SHRINKAGE;
        sizes = sizes +
                ANTIALIASING; //! should be before the scaling no? & also taken
                              //! into consideration with the dynamic clamping
#ifdef BBOX_PADDING
        sizes = sizes + BBOX_PADDING;
#endif
#ifdef MIN_BBOX_SIZE
        sizes = make_float3(
            fmaxf(sizes.x, MIN_BBOX_SIZE),
            fmaxf(sizes.y, MIN_BBOX_SIZE),
            fmaxf(sizes.z, MIN_BBOX_SIZE));
#endif
#endif

        create_transform_matrix(
            rotations[i],
            sizes,
            means[i],
            reinterpret_cast<float4(&)[3]>(instances[i].transform));
    }
}

void populateBVH(
    OptixInstance *instances,
    OptixTraversableHandle gasHandle,
    int num_gaussians,
    float3 *camera_position_world,
    float3 *scales,
    float4 *rotations,
    float3 *means,
    float *opacities,
    float *lod_means,
    float *lod_scales,
    bool *mask,
    float global_scaling_factor,
    float alpha_threshold,
#if OPTIMIZE_EXP_POWER == true
    float *exp_power
#else
    float exp_power
#endif
) {
    _populateBVH<<<(num_gaussians + 31) / 32, 32>>>(
        instances,
        gasHandle,
        num_gaussians,
        camera_position_world,
        scales,
        rotations,
        means,
        opacities,
        lod_means,
        lod_scales,
        mask,
        global_scaling_factor,
        alpha_threshold,
        exp_power);
}

__global__ void _populateTensor(
    float *mesh_cage_verts,
    int num_verts_per_gaussian,
    int *mesh_cage_faces,
    int num_faces_per_gaussian,
    float *output_verts,
    int *output_faces,
    int num_gaussians,
    float3 *camera_position_world,
    float3 *scales,
    float4 *rotations,
    float3 *means,
    float *opacities,
    float *lod_means,
    float *lod_scales,
    bool *mask,
    float global_scaling_factor,
    float alpha_threshold,
#if OPTIMIZE_EXP_POWER == true
    float *exp_power
#else
    float exp_power
#endif
) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    float4 transform[3];
    if (i < num_gaussians) {
        float opacity = opacities[i];
#if ACTIVATION_IN_CUDA == true
        opacity = sigmoid_act(opacity);
#endif

        float3 sizes = scales[i];
#if ACTIVATION_IN_CUDA == true
        sizes = exp_act(sizes);
#endif
// ????????? shouldn't clamping be after antialiasing?
#if OPTIMIZE_EXP_POWER == true
        float p = exp_power[i];
#else
        float p = exp_power;
#endif

        float scaling_factor =
            compute_scaling_factor(opacity, alpha_threshold, p);
        sizes = sizes * scaling_factor * global_scaling_factor; //!!!!!!!!!!!!
        sizes *= BB_SHRINKAGE;
        sizes = sizes +
                ANTIALIASING; //! should be before the scaling no? & also taken
                              //! into consideration with the dynamic clamping
#ifdef BBOX_PADDING
        sizes = sizes + BBOX_PADDING;
#endif

        create_transform_matrix(rotations[i], sizes, means[i], transform);
        for (int j = 0; j < num_verts_per_gaussian; j++) {
            float4 vert = make_float4(
                mesh_cage_verts[j * 3],
                mesh_cage_verts[j * 3 + 1],
                mesh_cage_verts[j * 3 + 2],
                1.0f);
            output_verts[j * 3 + 0 + 3 * num_verts_per_gaussian * i] =
                dot(vert, transform[0]);
            output_verts[j * 3 + 1 + 3 * num_verts_per_gaussian * i] =
                dot(vert, transform[1]);
            output_verts[j * 3 + 2 + 3 * num_verts_per_gaussian * i] =
                dot(vert, transform[2]);
        }
#if TRI_SOUP_FOR_MESHES == false
        for (int j = 0; j < num_faces_per_gaussian; j++) {
            output_faces[j * 3 + 0 + 3 * num_faces_per_gaussian * i] =
                mesh_cage_faces[j * 3 + 0] + num_verts_per_gaussian * i;
            output_faces[j * 3 + 1 + 3 * num_faces_per_gaussian * i] =
                mesh_cage_faces[j * 3 + 1] + num_verts_per_gaussian * i;
            output_faces[j * 3 + 2 + 3 * num_faces_per_gaussian * i] =
                mesh_cage_faces[j * 3 + 2] + num_verts_per_gaussian * i;
        }
#endif
    }
}

void transformVerts(
    float *mesh_cage_verts,
    int num_verts_per_gaussian,
    int *mesh_cage_faces,
    int num_faces_per_gaussian,
    float *output_verts,
    int *output_faces,
    int num_gaussians,
    float3 *camera_position_world,
    float3 *scales,
    float4 *rotations,
    float3 *means,
    float *opacities,
    float *lod_means,
    float *lod_scales,
    bool *mask,
    float global_scaling_factor,
    float alpha_threshold,
#if OPTIMIZE_EXP_POWER == true
    float *exp_power
#else
    float exp_power
#endif
) {
    _populateTensor<<<(num_gaussians + 31) / 32, 32>>>(
        mesh_cage_verts,
        num_verts_per_gaussian,
        mesh_cage_faces,
        num_faces_per_gaussian,
        output_verts,
        output_faces,
        num_gaussians,
        camera_position_world,
        scales,
        rotations,
        means,
        opacities,
        lod_means,
        lod_scales,
        mask,
        global_scaling_factor,
        alpha_threshold,
        exp_power);
}
