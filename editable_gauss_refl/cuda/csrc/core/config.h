#pragma once

#include "../flags.h"

struct Config {
    const float *exp_power;
    const float *alpha_threshold;
    const float *transmittance_threshold;
    const bool *accumulate_samples;
    const bool *jitter_primary_rays;
    const int *num_bounces;
    const float *global_scale_factor;
    const float *loss_weight_diffuse;
    const float *loss_weight_specular;
    const float *loss_weight_depth;
    const float *loss_weight_normal;
    const float *loss_weight_f0;
    const float *loss_weight_roughness;
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
    Tensor accumulate_samples = torch::tensor({false}, CUDA_BOOL);
    Tensor jitter_primary_rays = torch::tensor({true}, CUDA_BOOL);
    Tensor num_bounces = torch::tensor({2}, CUDA_INT32);
    Tensor global_scale_factor = torch::ones({1}, CUDA_FLOAT32);
    Tensor loss_weight_diffuse = torch::tensor({1.0}, CUDA_FLOAT32);
    Tensor loss_weight_specular = torch::tensor({1.0}, CUDA_FLOAT32);
    Tensor loss_weight_depth = torch::tensor({1.0}, CUDA_FLOAT32);
    Tensor loss_weight_normal = torch::tensor({1.0}, CUDA_FLOAT32);
    Tensor loss_weight_f0 = torch::tensor({1.0}, CUDA_FLOAT32);
    Tensor loss_weight_roughness = torch::tensor({1.0}, CUDA_FLOAT32);
    Tensor eps_forward_normalization = torch::tensor({1e-12}, CUDA_FLOAT32);
    Tensor eps_scale_grad = torch::tensor({1e-12f}, CUDA_FLOAT32);
    Tensor eps_ray_surface_offset = torch::tensor({0.01f}, CUDA_FLOAT32);
    Tensor eps_min_roughness = torch::tensor({0.01f}, CUDA_FLOAT32);
    Tensor reflection_invalid_normal_threshold = torch::tensor({0.7f}, CUDA_FLOAT32);
    Tensor backfacing_invalid_normal_threshold = torch::tensor({0.9f}, CUDA_FLOAT32);
    Tensor backfacing_max_dist = torch::tensor({0.1f}, CUDA_FLOAT32);

    Config reify() {
        return Config{
            .exp_power = reinterpret_cast<float *>(exp_power.data_ptr()),
            .alpha_threshold = reinterpret_cast<float *>(alpha_threshold.data_ptr()),
            .transmittance_threshold = reinterpret_cast<float *>(transmittance_threshold.data_ptr()),
            .accumulate_samples = reinterpret_cast<bool *>(accumulate_samples.data_ptr()),
            .jitter_primary_rays = reinterpret_cast<bool *>(jitter_primary_rays.data_ptr()),
            .num_bounces = reinterpret_cast<int *>(num_bounces.data_ptr()),
            .global_scale_factor = reinterpret_cast<float *>(global_scale_factor.data_ptr()),
            .loss_weight_diffuse = reinterpret_cast<float *>(loss_weight_diffuse.data_ptr()),
            .loss_weight_specular = reinterpret_cast<float *>(loss_weight_specular.data_ptr()),
            .loss_weight_depth = reinterpret_cast<float *>(loss_weight_depth.data_ptr()),
            .loss_weight_normal = reinterpret_cast<float *>(loss_weight_normal.data_ptr()),
            .loss_weight_f0 = reinterpret_cast<float *>(loss_weight_f0.data_ptr()),
            .loss_weight_roughness = reinterpret_cast<float *>(loss_weight_roughness.data_ptr()),
            .eps_forward_normalization = reinterpret_cast<float *>(eps_forward_normalization.data_ptr()),
            .eps_scale_grad = reinterpret_cast<float *>(eps_scale_grad.data_ptr()),
            .eps_ray_surface_offset = reinterpret_cast<float *>(eps_ray_surface_offset.data_ptr()),
            .eps_min_roughness = reinterpret_cast<float *>(eps_min_roughness.data_ptr()),
            .reflection_invalid_normal_threshold =
                reinterpret_cast<float *>(reflection_invalid_normal_threshold.data_ptr()),
            .backfacing_invalid_normal_threshold =
                reinterpret_cast<float *>(backfacing_invalid_normal_threshold.data_ptr()),
            .backfacing_max_dist = reinterpret_cast<float *>(backfacing_max_dist.data_ptr())};
    }

    static void bind(torch::Library &m) {
        m.class_<ConfigDataHolder>("ConfigDataHolder")
            .def_readonly("exp_power", &ConfigDataHolder::exp_power)
            .def_readonly("alpha_threshold", &ConfigDataHolder::alpha_threshold)
            .def_readonly("transmittance_threshold", &ConfigDataHolder::transmittance_threshold)
            .def_readonly("accumulate_samples", &ConfigDataHolder::accumulate_samples)
            .def_readonly("jitter_primary_rays", &ConfigDataHolder::jitter_primary_rays)
            .def_readonly("num_bounces", &ConfigDataHolder::num_bounces)
            .def_readonly("global_scale_factor", &ConfigDataHolder::global_scale_factor)
            .def_readonly("loss_weight_diffuse", &ConfigDataHolder::loss_weight_diffuse)
            .def_readonly("loss_weight_specular", &ConfigDataHolder::loss_weight_specular)
            .def_readonly("loss_weight_depth", &ConfigDataHolder::loss_weight_depth)
            .def_readonly("loss_weight_normal", &ConfigDataHolder::loss_weight_normal)
            .def_readonly("loss_weight_f0", &ConfigDataHolder::loss_weight_f0)
            .def_readonly("loss_weight_roughness", &ConfigDataHolder::loss_weight_roughness)
            .def_readonly("eps_forward_normalization", &ConfigDataHolder::eps_forward_normalization)
            .def_readonly("eps_scale_grad", &ConfigDataHolder::eps_scale_grad)
            .def_readonly("eps_ray_surface_offset", &ConfigDataHolder::eps_ray_surface_offset)
            .def_readonly("eps_min_roughness", &ConfigDataHolder::eps_min_roughness)
            .def_readonly("reflection_invalid_normal_threshold", &ConfigDataHolder::reflection_invalid_normal_threshold)
            .def_readonly("backfacing_invalid_normal_threshold", &ConfigDataHolder::backfacing_invalid_normal_threshold)
            .def_readonly("backfacing_max_dist", &ConfigDataHolder::backfacing_max_dist);
    }
};
#endif
