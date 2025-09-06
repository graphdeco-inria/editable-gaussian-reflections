
#pragma once

#include "../flags.h"

struct Config {
    const float *exp_power;
    const float *alpha_threshold;
    const float *transmittance_threshold;
    const float *global_scale_factor;
};

#ifndef __CUDACC__
#include "../headers/torch.h"

struct ConfigDataHolder : torch::CustomClassHolder {
    Tensor exp_power = torch::tensor({3}, CUDA_FLOAT32);
    Tensor alpha_threshold = torch::tensor({0.005}, CUDA_FLOAT32);
    Tensor transmittance_threshold = torch::tensor({0.01}, CUDA_FLOAT32);
    Tensor global_scale_factor = torch::ones({1}, CUDA_FLOAT32);

    Config reify() {
        return Config{
            .exp_power = reinterpret_cast<float *>(exp_power.data_ptr()),
            .alpha_threshold =
                reinterpret_cast<float *>(alpha_threshold.data_ptr()),
            .transmittance_threshold =
                reinterpret_cast<float *>(transmittance_threshold.data_ptr()),
            .global_scale_factor =
                reinterpret_cast<float *>(global_scale_factor.data_ptr())};
    }

    static void bind(torch::Library &m) {
        m.class_<ConfigDataHolder>("ConfigDataHolder")
            .def_readonly("exp_power", &ConfigDataHolder::exp_power)
            .def_readonly("alpha_threshold", &ConfigDataHolder::alpha_threshold)
            .def_readonly(
                "transmittance_threshold",
                &ConfigDataHolder::transmittance_threshold)
            .def_readonly(
                "global_scale_factor", &ConfigDataHolder::global_scale_factor);
    }
};
#endif
