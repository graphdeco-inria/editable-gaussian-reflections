#pragma once

struct Gaussians {
    uint32_t count;

    const float3 *rgb;
    const float3 *normal;
    const float3 *f0;
    const float *roughness;
    const float *opacity;
    const float3 *scale;
    const float3 *mean;
    const float4 *rotation;

    float3 *__restrict__ dL_drgb;
    float3 *__restrict__ dL_dnormal;
    float3 *__restrict__ dL_df0;
    float *__restrict__ dL_droughness;
    float *__restrict__ dL_dopacity;
    float3 *__restrict__ dL_dscale;
    float3 *__restrict__ dL_dmean;
    float4 *__restrict__ dL_drotation;

    float *__restrict__ total_weight;
};

#ifndef __CUDACC__
#include "../headers/torch.h"

struct GaussianDataHolder : torch::CustomClassHolder {
    uint32_t count = 1;

    Tensor rgb = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor normal = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor f0 = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor roughness = torch::zeros({1, 1}, CUDA_FLOAT32);
    Tensor opacity = torch::zeros({1, 1}, CUDA_FLOAT32);
    Tensor scale = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor mean = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor rotation = torch::zeros({1, 4}, CUDA_FLOAT32);

    Tensor dL_drgb = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor dL_dnormal = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor dL_df0 = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor dL_droughness = torch::zeros({1, 1}, CUDA_FLOAT32);
    Tensor dL_dopacity = torch::zeros({1, 1}, CUDA_FLOAT32);
    Tensor dL_dscale = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor dL_dmean = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor dL_drotation = torch::zeros({1, 4}, CUDA_FLOAT32);

    Tensor total_weight = torch::zeros({1, 1}, CUDA_FLOAT32);

    GaussianDataHolder() {
        rgb.mutable_grad() = dL_drgb;
        normal.mutable_grad() = dL_dnormal;
        f0.mutable_grad() = dL_df0;
        roughness.mutable_grad() = dL_droughness;
        opacity.mutable_grad() = dL_dopacity;
        scale.mutable_grad() = dL_dscale;
        mean.mutable_grad() = dL_dmean;
        rotation.mutable_grad() = dL_drotation;
    }

    void resize(int64_t num_new_gaussians) {
        torch::NoGradGuard no_grad;
        count = num_new_gaussians;
        rgb.resize_({num_new_gaussians, 3});
        normal.resize_({num_new_gaussians, 3});
        f0.resize_({num_new_gaussians, 3});
        roughness.resize_({num_new_gaussians, 1});
        opacity.resize_({num_new_gaussians, 1});
        scale.resize_({num_new_gaussians, 3});
        mean.resize_({num_new_gaussians, 3});
        rotation.resize_({num_new_gaussians, 4});

        dL_drgb.resize_({num_new_gaussians, 3});
        dL_dnormal.resize_({num_new_gaussians, 3});
        dL_df0.resize_({num_new_gaussians, 3});
        dL_droughness.resize_({num_new_gaussians, 1});
        dL_dopacity.resize_({num_new_gaussians, 1});
        dL_dscale.resize_({num_new_gaussians, 3});
        dL_dmean.resize_({num_new_gaussians, 3});
        dL_drotation.resize_({num_new_gaussians, 4});

        total_weight.resize_({num_new_gaussians, 1});
    }

    Gaussians reify() {
        Gaussians gaussians;
        gaussians.count = count;
        gaussians.rgb = reinterpret_cast<float3 *>(rgb.data_ptr());
        gaussians.normal = reinterpret_cast<float3 *>(normal.data_ptr());
        gaussians.f0 = reinterpret_cast<float3 *>(f0.data_ptr());
        gaussians.roughness = reinterpret_cast<float *>(roughness.data_ptr());
        gaussians.opacity = reinterpret_cast<float *>(opacity.data_ptr());
        gaussians.scale = reinterpret_cast<float3 *>(scale.data_ptr());
        gaussians.mean = reinterpret_cast<float3 *>(mean.data_ptr());
        gaussians.rotation = reinterpret_cast<float4 *>(rotation.data_ptr());

        gaussians.dL_drgb = reinterpret_cast<float3 *>(dL_drgb.data_ptr());
        gaussians.dL_dnormal = reinterpret_cast<float3 *>(dL_dnormal.data_ptr());
        gaussians.dL_df0 = reinterpret_cast<float3 *>(dL_df0.data_ptr());
        gaussians.dL_droughness = reinterpret_cast<float *>(dL_droughness.data_ptr());
        gaussians.dL_dopacity = reinterpret_cast<float *>(dL_dopacity.data_ptr());
        gaussians.dL_dscale = reinterpret_cast<float3 *>(dL_dscale.data_ptr());
        gaussians.dL_dmean = reinterpret_cast<float3 *>(dL_dmean.data_ptr());
        gaussians.dL_drotation = reinterpret_cast<float4 *>(dL_drotation.data_ptr());

        gaussians.total_weight = reinterpret_cast<float *>(total_weight.data_ptr());
        return gaussians;
    }

    static void bind(torch::Library &m) {
        m.class_<GaussianDataHolder>("GaussianDataHolder")
            .def_readonly("rgb", &GaussianDataHolder::rgb)
            .def_readonly("normal", &GaussianDataHolder::normal)
            .def_readonly("f0", &GaussianDataHolder::f0)
            .def_readonly("roughness", &GaussianDataHolder::roughness)
            .def_readonly("opacity", &GaussianDataHolder::opacity)
            .def_readonly("scale", &GaussianDataHolder::scale)
            .def_readonly("mean", &GaussianDataHolder::mean)
            .def_readonly("rotation", &GaussianDataHolder::rotation)

            .def_readonly("dL_drgb", &GaussianDataHolder::dL_drgb)
            .def_readonly("dL_dnormal", &GaussianDataHolder::dL_dnormal)
            .def_readonly("dL_df0", &GaussianDataHolder::dL_df0)
            .def_readonly("dL_droughness", &GaussianDataHolder::dL_droughness)
            .def_readonly("dL_dopacity", &GaussianDataHolder::dL_dopacity)
            .def_readonly("dL_dscale", &GaussianDataHolder::dL_dscale)
            .def_readonly("dL_dmean", &GaussianDataHolder::dL_dmean)
            .def_readonly("dL_drotation", &GaussianDataHolder::dL_drotation)

            .def_readonly("total_weight", &GaussianDataHolder::total_weight);
    }
};
#endif