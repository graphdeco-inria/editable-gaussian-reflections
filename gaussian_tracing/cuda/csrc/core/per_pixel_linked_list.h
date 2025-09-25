#pragma once

struct PerPixelLinkedList {
    static constexpr uint32_t NULL_PTR = 2 << 29;

    uint32_t *head_per_pixel; // * For each pixel point to the first entry
    uint32_t *total_hits;
    // * Entry data (all pixels in a flat array)
    uint32_t *gaussian_ids;
    float *distances;
    float *alphas;
    float *transmittances;
    float3 *local_hits;
    float *gaussvals;
    // * Entry metadata
    uint32_t *previous_entries; // * Backward pointer to the last entry

#ifdef __CUDACC__
    __device__ void insert(
        bool grads_enabled,
        uint32_t pixel_id,
        uint32_t gaussian_id,
        float distance,
        float3 local_hit,
        float gaussval,
        float alpha,
        float transmittance = -1.0 // * just pass in a dummy value if not needed
                                   // (forward), simpler than having 2 classes
    ) {
        int hit_idx = atomicAdd(total_hits, 1);
        gaussian_ids[hit_idx] = gaussian_id;
        distances[hit_idx] = distance;
        alphas[hit_idx] = alpha;
        if (grads_enabled) {
            gaussvals[hit_idx] = gaussval;
            local_hits[hit_idx] = local_hit;
            if (transmittance != -1.0) {
                transmittances[hit_idx] = transmittance;
            }
        }
        previous_entries[hit_idx] = head_per_pixel[pixel_id];
        head_per_pixel[pixel_id] = hit_idx;
    }

    __device__ void reset(uint32_t pixel_id) {
        head_per_pixel[pixel_id] = NULL_PTR; // * Reset the head pointer for this pixel
    }

    // * Boilerplate iteration logic below
    struct PixelIterator {
        const PerPixelLinkedList *parent;
        uint32_t hit_idx;

        __device__ uint32_t operator*() const { return hit_idx; }

        __device__ PixelIterator &operator++() {
            hit_idx = parent->previous_entries[hit_idx];
            return *this;
        }

        __device__ bool operator!=(const PixelIterator &other) const {
            return hit_idx != other.hit_idx;
        }
    };

    struct PixelView {
        const PerPixelLinkedList *parent;
        uint32_t pixel_id;

        __device__ PixelIterator begin() const {
            return PixelIterator{parent, parent->head_per_pixel[pixel_id]};
        }

        __device__ PixelIterator end() const {
            return PixelIterator{parent, PerPixelLinkedList::NULL_PTR};
        }
    };

    __device__ PixelView pixel_view(uint32_t pixel_id) const { return PixelView{this, pixel_id}; }
#endif
};

#ifndef __CUDACC__
#include "../headers/torch.h"

struct PPLLDataHolder : torch::CustomClassHolder {
    Tensor head_per_pixel;
    Tensor total_hits = torch::zeros({1}, CUDA_INT32) - 1;
    // * Entry data
    Tensor gaussian_ids;
    Tensor distances;
    Tensor gaussvals;
    Tensor alphas;
    Tensor local_hits;
    Tensor transmittances;
    // * Entry metadata
    Tensor previous_entries;

    PPLLDataHolder(uint32_t image_width, uint32_t image_height, uint32_t size) {
        head_per_pixel = torch::zeros({image_height, image_width}, CUDA_INT32);
        head_per_pixel.fill_((int)PerPixelLinkedList::NULL_PTR);
        gaussian_ids = torch::zeros({size}, CUDA_INT32);
        distances = torch::zeros({size}, CUDA_FLOAT32);
        gaussvals = torch::zeros({size}, CUDA_FLOAT32);
        alphas = torch::zeros({size}, CUDA_FLOAT32);
        local_hits = torch::zeros({size, 3}, CUDA_FLOAT32);
        transmittances = torch::zeros({size}, CUDA_FLOAT32);
        previous_entries = torch::zeros({size}, CUDA_INT32);
    }

    PerPixelLinkedList reify() {
        return PerPixelLinkedList{
            .head_per_pixel = reinterpret_cast<uint32_t *>(head_per_pixel.data_ptr()),
            .total_hits = reinterpret_cast<uint32_t *>(total_hits.data_ptr()),
            .gaussian_ids = reinterpret_cast<uint32_t *>(gaussian_ids.data_ptr()),
            .distances = reinterpret_cast<float *>(distances.data_ptr()),
            .alphas = reinterpret_cast<float *>(alphas.data_ptr()),
            .transmittances = reinterpret_cast<float *>(transmittances.data_ptr()),
            .local_hits = reinterpret_cast<float3 *>(local_hits.data_ptr()),
            .gaussvals = reinterpret_cast<float *>(gaussvals.data_ptr()),
            .previous_entries = reinterpret_cast<uint32_t *>(previous_entries.data_ptr())};
    }

    void reset() {
        total_hits.fill_(0);
        head_per_pixel.fill_((int)PerPixelLinkedList::NULL_PTR);
    }

    static void bind(torch::Library &m) {
        m.class_<PPLLDataHolder>("PPLLDataHolder")
            .def_readonly("head_per_pixel", &PPLLDataHolder::head_per_pixel)
            .def_readonly("total_hits", &PPLLDataHolder::total_hits)
            .def_readonly("gaussian_ids", &PPLLDataHolder::gaussian_ids)
            .def_readonly("distances", &PPLLDataHolder::distances)
            .def_readonly("alphas", &PPLLDataHolder::alphas)
            .def_readonly("transmittances", &PPLLDataHolder::transmittances)
            .def_readonly("local_hits", &PPLLDataHolder::local_hits)
            .def_readonly("gaussvals", &PPLLDataHolder::gaussvals)
            .def_readonly("previous_entries", &PPLLDataHolder::previous_entries)

            .def_static("NULL_PTR", []() { return (int64_t)PerPixelLinkedList::NULL_PTR; });
    }
};
#endif