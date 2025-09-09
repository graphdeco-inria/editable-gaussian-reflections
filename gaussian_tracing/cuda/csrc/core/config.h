#pragma once

#include "../buffer_definition.h"
#include "../flags.h"

struct Config {
    const float *exp_power;
    const float *alpha_threshold;
    const float *transmittance_threshold;
    const float *global_scale_factor;
#define DECLARE_LOSS_PARAM(name, k) const float *loss_weight_##name;
    ALL_TARGET_BUFFERS(DECLARE_LOSS_PARAM);
    const float *eps_forward_normalization;
    const float *eps_scale_grad;
    const float *eps_ray_surface_offset;
    const float *eps_min_roughness;
    const float *reflection_invalid_normal_threshold;
    const float *backfacing_invalid_normal_threshold;
    const float *backfacing_max_dist;
};

#ifndef __CUDACC__
#include "../headers/torch.h"

struct ConfigDataHolder : torch::CustomClassHolder {
    Tensor exp_power = torch::tensor({3}, CUDA_FLOAT32);
    Tensor alpha_threshold = torch::tensor({0.005}, CUDA_FLOAT32);
    Tensor transmittance_threshold = torch::tensor({0.01}, CUDA_FLOAT32);
    Tensor global_scale_factor = torch::ones({1}, CUDA_FLOAT32);
#define DECLARE_LOSS_BUFFER(name, k)                                           \
    Tensor loss_weight_##name = torch::tensor({1.0}, CUDA_FLOAT32);
    ALL_TARGET_BUFFERS(DECLARE_LOSS_BUFFER)
    Tensor eps_forward_normalization = torch::tensor({1e-12}, CUDA_FLOAT32);
    Tensor eps_scale_grad = torch::tensor({1e-12f}, CUDA_FLOAT32);
    Tensor eps_ray_surface_offset = torch::tensor({0.01f}, CUDA_FLOAT32);
    Tensor eps_min_roughness = torch::tensor({0.01f}, CUDA_FLOAT32);
    Tensor reflection_invalid_normal_threshold =
        torch::tensor({0.7f}, CUDA_FLOAT32);
    Tensor backfacing_invalid_normal_threshold =
        torch::tensor({0.9f}, CUDA_FLOAT32);
    Tensor backfacing_max_dist = torch::tensor({0.1f}, CUDA_FLOAT32);

    Config reify() {
        return Config{
            .exp_power = reinterpret_cast<float *>(exp_power.data_ptr()),
            .alpha_threshold =
                reinterpret_cast<float *>(alpha_threshold.data_ptr()),
            .transmittance_threshold =
                reinterpret_cast<float *>(transmittance_threshold.data_ptr()),
            .global_scale_factor =
                reinterpret_cast<float *>(global_scale_factor.data_ptr()),
#define REIFY_LOSS_BUFFER(name, k)                                             \
    .loss_weight_##name =                                                      \
        reinterpret_cast<float *>(loss_weight_##name.data_ptr()),
            ALL_TARGET_BUFFERS(REIFY_LOSS_BUFFER).eps_forward_normalization =
                reinterpret_cast<float *>(eps_forward_normalization.data_ptr()),
            .eps_scale_grad =
                reinterpret_cast<float *>(eps_scale_grad.data_ptr()),
            .eps_ray_surface_offset =
                reinterpret_cast<float *>(eps_ray_surface_offset.data_ptr()),
            .eps_min_roughness =
                reinterpret_cast<float *>(eps_min_roughness.data_ptr()),
            .reflection_invalid_normal_threshold = reinterpret_cast<float *>(
                reflection_invalid_normal_threshold.data_ptr()),
            .backfacing_invalid_normal_threshold = reinterpret_cast<float *>(
                backfacing_invalid_normal_threshold.data_ptr()),
            .backfacing_max_dist =
                reinterpret_cast<float *>(backfacing_max_dist.data_ptr())};
    }

    static void bind(torch::Library &m) {
        m.class_<ConfigDataHolder>("ConfigDataHolder")
            .def_readonly("exp_power", &ConfigDataHolder::exp_power)
            .def_readonly("alpha_threshold", &ConfigDataHolder::alpha_threshold)
            .def_readonly(
                "transmittance_threshold",
                &ConfigDataHolder::transmittance_threshold)
            .def_readonly(
                "global_scale_factor", &ConfigDataHolder::global_scale_factor)
#define BIND_LOSS_BUFFER(name, k)                                              \
    .def_readonly("loss_weight_" #name, &ConfigDataHolder::loss_weight_##name)
                ALL_TARGET_BUFFERS(BIND_LOSS_BUFFER)
            .def_readonly(
                "eps_forward_normalization",
                &ConfigDataHolder::eps_forward_normalization)
            .def_readonly("eps_scale_grad", &ConfigDataHolder::eps_scale_grad)
            .def_readonly(
                "eps_ray_surface_offset",
                &ConfigDataHolder::eps_ray_surface_offset)
            .def_readonly(
                "eps_min_roughness", &ConfigDataHolder::eps_min_roughness)
            .def_readonly(
                "reflection_invalid_normal_threshold",
                &ConfigDataHolder::reflection_invalid_normal_threshold)
            .def_readonly(
                "backfacing_invalid_normal_threshold",
                &ConfigDataHolder::backfacing_invalid_normal_threshold)
            .def_readonly(
                "backfacing_max_dist", &ConfigDataHolder::backfacing_max_dist);
    }
};
#endif
