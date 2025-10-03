#pragma once

#ifdef __CUDACC__
#include "../utils/random.h"
#include "../utils/vec_math.h"
#endif

struct Camera {
    const float3 *origin;
    const float *vertical_fov_radians;
    const float3 *rotation_c2w; // * stores 3 rows for each 3x3 matrix
    const float3 *rotation_w2c; // * stores 3 rows for each 3x3 matrix
    const float *znear;
    const float *zfar;

#ifdef __CUDACC__
    __device__ float3
    compute_primary_ray_direction(const bool jitter, const uint3 idx, const uint3 dim, unsigned int &seed) {
        // * Get camera intrinsics
        float view_size = tan(*vertical_fov_radians / 2);
        float aspect_ratio = float(dim.x) / float(dim.y);

        // * Compute sub-pixel jitter
        float2 idxf = make_float2(idx.x, idx.y);
        if (jitter) {
            const float2 jitter_offset = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
            idxf += jitter_offset;
        }

        // * Convert to NDC
        float y = view_size * (1.0f - 2.0f * (idxf.y + 1.0f / 2.0f) / (float(dim.y)));
        float x = aspect_ratio * view_size * (2.0f * (idxf.x + 1.0f / 2.0f) / (float(dim.x)) - 1.0f);

        // * Rotate to world and normalize (n.b. multiplies by *transposed* w2c)
        return normalize(rotation_w2c[0] * x + rotation_w2c[1] * y - rotation_w2c[2]);
    }
#endif
};

#ifndef __CUDACC__
#include "../headers/torch.h"

struct CameraDataHolder : torch::CustomClassHolder {
    Tensor origin = torch::zeros({3}, CUDA_FLOAT32);
    Tensor vertical_fov_radians = torch::zeros({1}, CUDA_FLOAT32);
    Tensor rotation_c2w = torch::zeros({3, 3}, CUDA_FLOAT32);
    Tensor rotation_w2c = torch::zeros({3, 3}, CUDA_FLOAT32);
    Tensor znear = torch::zeros({1}, CUDA_FLOAT32);
    Tensor zfar = torch::zeros({1}, CUDA_FLOAT32);

    Camera reify() {
        return Camera{
            .origin = reinterpret_cast<float3 *>(origin.data_ptr()),
            .vertical_fov_radians = reinterpret_cast<float *>(vertical_fov_radians.data_ptr()),
            .rotation_c2w = reinterpret_cast<float3 *>(rotation_c2w.data_ptr()),
            .rotation_w2c = reinterpret_cast<float3 *>(rotation_w2c.data_ptr()),
            .znear = reinterpret_cast<float *>(znear.data_ptr()),
            .zfar = reinterpret_cast<float *>(zfar.data_ptr()),
        };
    }

    void set_pose(const Tensor &c2w_origin, const Tensor &c2w_rotation) {
        TORCH_CHECK(c2w_rotation.sizes() == torch::IntArrayRef({3, 3}), "c2w_rotation must be 3x3");
        TORCH_CHECK(c2w_origin.sizes() == torch::IntArrayRef({3}), "c2w_origin must be 3");
        rotation_c2w.copy_(c2w_rotation);
        origin.copy_(c2w_origin);
        rotation_w2c.copy_(c2w_rotation.transpose(0, 1));
    }

    static void bind(torch::Library &m) {
        m.class_<CameraDataHolder>("CameraDataHolder")
            .def("set_pose", &CameraDataHolder::set_pose)
            .def_readonly("vertical_fov_radians", &CameraDataHolder::vertical_fov_radians)
            .def_readonly("znear", &CameraDataHolder::znear)
            .def_readonly("zfar", &CameraDataHolder::zfar);
    }
};
#endif
