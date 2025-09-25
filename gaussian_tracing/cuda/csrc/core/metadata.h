#pragma once

struct Metadata {
    bool *__restrict__ grads_enabled;
    uint32_t *__restrict__ total_num_calls;
    uint32_t *__restrict__ random_seeds;
};

#ifndef __CUDACC__
#include "../headers/torch.h"

struct MetaDataHolder : torch::CustomClassHolder {
    Tensor grads_enabled = torch::ones({1}, CUDA_BOOL);
    Tensor total_num_calls = torch::zeros({1}, CUDA_INT32);
    Tensor random_seeds;

    MetaDataHolder(uint32_t image_width, uint32_t image_height) {
        random_seeds = torch::randint(0, 1000000000, {image_height, image_width, 1}, CUDA_INT32);
    }

    Metadata reify() {
        return Metadata{
            .grads_enabled = reinterpret_cast<bool *>(grads_enabled.data_ptr()),
            .total_num_calls = reinterpret_cast<uint32_t *>(total_num_calls.data_ptr()),
            .random_seeds = reinterpret_cast<uint32_t *>(random_seeds.data_ptr())};
    }

    void update() {
        grads_enabled.fill_(torch::autograd::GradMode::is_enabled());
        total_num_calls += 1;
    }

    static void bind(torch::Library &m) {
        m.class_<MetaDataHolder>("MetaDataHolder")
            .def_readonly("grads_enabled", &MetaDataHolder::grads_enabled)
            .def_readonly("total_num_calls", &MetaDataHolder::total_num_calls)
            .def_readonly("random_seeds", &MetaDataHolder::random_seeds);
    }
};

#endif