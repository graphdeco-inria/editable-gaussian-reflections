#pragma once

struct Stats {
    int *__restrict__ num_accumulated_per_pixel;
    int *__restrict__ num_traversed_per_pixel;
};

#ifndef __CUDACC__
#include "../headers/torch.h"

struct StatsDataHolder : torch::CustomClassHolder {
    Tensor num_accumulated_per_pixel;
    Tensor num_traversed_per_pixel;

    StatsDataHolder(uint32_t image_width, uint32_t image_height) {
        num_accumulated_per_pixel = torch::zeros({image_height, image_width}, CUDA_INT32);
        num_traversed_per_pixel = torch::zeros({image_height, image_width}, CUDA_INT32);
    }

    Stats reify() {
        return Stats{
            .num_accumulated_per_pixel = reinterpret_cast<int *>(num_accumulated_per_pixel.data_ptr()),
            .num_traversed_per_pixel = reinterpret_cast<int *>(num_traversed_per_pixel.data_ptr())};
    }

    void reset() {
        num_accumulated_per_pixel.zero_();
        num_traversed_per_pixel.zero_();
    }

    static void bind(torch::Library &m) {
        m.class_<StatsDataHolder>("StatsDataHolder")
            .def_readonly("num_accumulated_per_pixel", &StatsDataHolder::num_accumulated_per_pixel)
            .def_readonly("num_traversed_per_pixel", &StatsDataHolder::num_traversed_per_pixel);
    }
};

#endif