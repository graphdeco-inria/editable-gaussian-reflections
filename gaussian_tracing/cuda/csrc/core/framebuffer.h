#pragma once

#include "../flags.h"
#ifdef __CUDACC__
#include "../utils/common.h"
#endif

struct Pixel {
    uint32_t id;

    float3 output_rgb[MAX_BOUNCES + 1];
    float3 remaining_rgb[MAX_BOUNCES + 1];
    float output_depth[MAX_BOUNCES + 1];
    float remaining_depth[MAX_BOUNCES + 1];
    float3 output_normal[MAX_BOUNCES + 1];
    float3 remaining_normal[MAX_BOUNCES + 1];
    float3 output_f0[MAX_BOUNCES + 1];
    float3 remaining_f0[MAX_BOUNCES + 1];
    float output_roughness[MAX_BOUNCES + 1];
    float remaining_roughness[MAX_BOUNCES + 1];
    float output_transmittance[MAX_BOUNCES + 1];
    float remaining_transmittance[MAX_BOUNCES + 1];
    float output_total_transmittance[MAX_BOUNCES + 1];
    float remaining_total_transmittance[MAX_BOUNCES + 1];
    float3 output_brdf[MAX_BOUNCES + 1];
    float3 remaining_brdf[MAX_BOUNCES + 1];
    float3 output_ray_origin[MAX_BOUNCES + 1];
    float3 remaining_ray_origin[MAX_BOUNCES + 1];
    float3 output_ray_direction[MAX_BOUNCES + 1];
    float3 remaining_ray_direction[MAX_BOUNCES + 1];
    float3 output_throughput[MAX_BOUNCES + 1];
    float3 output_final;

    float3 accumulated_rgb[MAX_BOUNCES + 1];
    float accumulated_transmittance[MAX_BOUNCES + 1];
    float accumulated_total_transmittance[MAX_BOUNCES + 1];
    float accumulated_depth[MAX_BOUNCES + 1];
    float3 accumulated_normal[MAX_BOUNCES + 1];
    float3 accumulated_f0[MAX_BOUNCES + 1];
    float accumulated_roughness[MAX_BOUNCES + 1];
    int accumulated_sample_count;

    float3 target_diffuse;
    float3 target_glossy;
    float target_depth;
    float3 target_normal;
    float3 target_f0;
    float target_roughness;

#ifdef __CUDACC__
    __device__ Pixel(uint32_t id_) : id(id_) {
        fill_array(output_rgb, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(remaining_rgb, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(output_depth, MAX_BOUNCES + 1, 0.0f);
        fill_array(remaining_depth, MAX_BOUNCES + 1, 0.0f);
        fill_array(output_normal, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(remaining_normal, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(output_f0, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(remaining_f0, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(output_roughness, MAX_BOUNCES + 1, 0.0f);
        fill_array(remaining_roughness, MAX_BOUNCES + 1, 0.0f);
        fill_array(output_transmittance, MAX_BOUNCES + 1, 1.0f);
        fill_array(remaining_transmittance, MAX_BOUNCES + 1, 1.0f);
        fill_array(output_total_transmittance, MAX_BOUNCES + 1, 1.0f);
        fill_array(remaining_total_transmittance, MAX_BOUNCES + 1, 1.0f);
        fill_array(output_brdf, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(remaining_brdf, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(output_ray_origin, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(remaining_ray_origin, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(output_ray_direction, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(remaining_ray_direction, MAX_BOUNCES + 1, make_float3(0.0f, 0.0f, 0.0f));
        fill_array(output_throughput, MAX_BOUNCES + 1, make_float3(1.0f, 1.0f, 1.0f));
        output_final = make_float3(0.0f, 0.0f, 0.0f);

        target_diffuse = make_float3(0.0f, 0.0f, 0.0f);
        target_glossy = make_float3(0.0f, 0.0f, 0.0f);
        target_depth = 0.0f;
        target_normal = make_float3(0.0f, 0.0f, 0.0f);
        target_f0 = make_float3(0.0f, 0.0f, 0.0f);
        target_roughness = 0.0f;
    }
#endif
};

struct Framebuffer {
    float3 *__restrict__ output_rgb;
    float *__restrict__ output_depth;
    float3 *__restrict__ output_normal;
    float3 *__restrict__ output_f0;
    float *__restrict__ output_roughness;
    float *__restrict__ output_transmittance;
    float *__restrict__ output_total_transmittance;
    float3 *__restrict__ output_brdf;
    float3 *__restrict__ output_ray_origin;
    float3 *__restrict__ output_ray_direction;
    float3 *__restrict__ output_final;
    float3 *__restrict__ output_denoised;

    float3 *__restrict__ accumulated_rgb;
    float *__restrict__ accumulated_transmittance;
    float *__restrict__ accumulated_total_transmittance;
    float *__restrict__ accumulated_depth;
    float3 *__restrict__ accumulated_normal;
    float3 *__restrict__ accumulated_f0;
    float *__restrict__ accumulated_roughness;
    int *accumulated_sample_count;

    const float3 *__restrict__ target_diffuse;
    const float3 *__restrict__ target_glossy;
    const float *__restrict__ target_depth;
    const float3 *__restrict__ target_normal;
    const float3 *__restrict__ target_f0;
    const float *__restrict__ target_roughness;

    uint32_t num_pixels;

#ifdef __CUDACC__
    __device__ void update_accumulators(Pixel &pixel) {
        int accum_count = *accumulated_sample_count;
        pixel.output_final = make_float3(0.0f, 0.0f, 0.0f);
        for (int step = 0; step < MAX_BOUNCES + 1; step++) {
            accumulated_rgb[pixel.id + num_pixels * step] += pixel.output_rgb[step];
            accumulated_transmittance[pixel.id + num_pixels * step] += pixel.output_transmittance[step];
            accumulated_total_transmittance[pixel.id + num_pixels * step] += pixel.output_total_transmittance[step];
            accumulated_depth[pixel.id + num_pixels * step] += pixel.output_depth[step];
            accumulated_normal[pixel.id + num_pixels * step] += pixel.output_normal[step];
            accumulated_f0[pixel.id + num_pixels * step] += pixel.output_f0[step];
            accumulated_roughness[pixel.id + num_pixels * step] += pixel.output_roughness[step];
            
            pixel.output_rgb[step] = accumulated_rgb[pixel.id + num_pixels * step] / float(accum_count + 1);
            pixel.output_transmittance[step] = accumulated_transmittance[pixel.id + num_pixels * step] / float(accum_count + 1);
            pixel.output_total_transmittance[step] = accumulated_total_transmittance[pixel.id + num_pixels * step] / float(accum_count + 1);
            pixel.output_depth[step] = accumulated_depth[pixel.id + num_pixels * step] / float(accum_count + 1);
            pixel.output_normal[step] = accumulated_normal[pixel.id + num_pixels * step] / float(accum_count + 1);
            pixel.output_f0[step] = accumulated_f0[pixel.id + num_pixels * step] / float(accum_count + 1);
            pixel.output_roughness[step] = accumulated_roughness[pixel.id + num_pixels * step] / float(accum_count + 1);
            
            pixel.output_final += pixel.output_rgb[step];
        }
    }

    __device__ void write_outputs(const Pixel &pixel) {
        for (int step = 0; step < MAX_BOUNCES + 1; step++) {
            output_rgb[pixel.id + num_pixels * step] = pixel.output_rgb[step];
            output_depth[pixel.id + num_pixels * step] = pixel.output_depth[step];
            output_normal[pixel.id + num_pixels * step] = pixel.output_normal[step];
            output_f0[pixel.id + num_pixels * step] = pixel.output_f0[step];
            output_roughness[pixel.id + num_pixels * step] = pixel.output_roughness[step];
            output_transmittance[pixel.id + num_pixels * step] = pixel.output_transmittance[step];
            output_total_transmittance[pixel.id + num_pixels * step] = pixel.output_total_transmittance[step];
            output_brdf[pixel.id + num_pixels * step] = pixel.output_brdf[step];
            output_ray_origin[pixel.id + num_pixels * step] = pixel.output_ray_origin[step];
            output_ray_direction[pixel.id + num_pixels * step] = pixel.output_ray_direction[step];
        }
        output_final[pixel.id] = pixel.output_final;
    }

    __device__ void fetch_targets(Pixel &pixel) {
        pixel.target_diffuse = target_diffuse[pixel.id];
        pixel.target_glossy = target_glossy[pixel.id];
        pixel.target_depth = target_depth[pixel.id];
        pixel.target_normal = target_normal[pixel.id];
        pixel.target_f0 = target_f0[pixel.id];
        pixel.target_roughness = target_roughness[pixel.id];
    }
#endif
};

#ifndef __CUDACC__
#include "../headers/torch.h"

struct FramebufferDataHolder : torch::CustomClassHolder {
    Tensor output_rgb;
    Tensor output_depth;
    Tensor output_normal;
    Tensor output_f0;
    Tensor output_roughness;
    Tensor output_transmittance;
    Tensor output_total_transmittance;
    Tensor output_brdf;
    Tensor output_ray_origin;
    Tensor output_ray_direction;
    Tensor output_final;
    Tensor output_denoised;

    Tensor accumulated_rgb;
    Tensor accumulated_transmittance;
    Tensor accumulated_total_transmittance;
    Tensor accumulated_depth;
    Tensor accumulated_normal;
    Tensor accumulated_f0;
    Tensor accumulated_roughness;
    Tensor accumulated_sample_count;

    Tensor target_diffuse;
    Tensor target_glossy;
    Tensor target_depth;
    Tensor target_normal;
    Tensor target_f0;
    Tensor target_roughness;

    FramebufferDataHolder(uint32_t image_width, uint32_t image_height) {
        output_rgb = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 3}, CUDA_FLOAT32);
        output_depth = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 1}, CUDA_FLOAT32);
        output_normal = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 3}, CUDA_FLOAT32);
        output_f0 = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 3}, CUDA_FLOAT32);
        output_roughness = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 1}, CUDA_FLOAT32);
        output_transmittance = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 1}, CUDA_FLOAT32);
        output_total_transmittance = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 1}, CUDA_FLOAT32);
        output_brdf = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 3}, CUDA_FLOAT32);
        output_ray_origin = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 3}, CUDA_FLOAT32);
        output_ray_direction = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 3}, CUDA_FLOAT32);
        output_final = torch::zeros({1, image_height, image_width, 3}, CUDA_FLOAT32);
        output_denoised = torch::zeros({1, image_height, image_width, 3}, CUDA_FLOAT32);

        accumulated_rgb = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 3}, CUDA_FLOAT32);
        accumulated_transmittance = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 1}, CUDA_FLOAT32);
        accumulated_total_transmittance = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 1}, CUDA_FLOAT32);
        accumulated_depth = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 1}, CUDA_FLOAT32);
        accumulated_normal = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 3}, CUDA_FLOAT32);
        accumulated_f0 = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 3}, CUDA_FLOAT32);
        accumulated_roughness = torch::zeros({MAX_BOUNCES + 1, image_height, image_width, 1}, CUDA_FLOAT32);
        accumulated_sample_count = torch::zeros({1}, CUDA_INT32);

        target_diffuse = torch::zeros({image_height, image_width, 3}, CUDA_FLOAT32);
        target_glossy = torch::zeros({image_height, image_width, 3}, CUDA_FLOAT32);
        target_depth = torch::zeros({image_height, image_width, 1}, CUDA_FLOAT32);
        target_normal = torch::zeros({image_height, image_width, 3}, CUDA_FLOAT32);
        target_f0 = torch::zeros({image_height, image_width, 3}, CUDA_FLOAT32);
        target_roughness = torch::zeros({image_height, image_width, 1}, CUDA_FLOAT32);
    }

    Framebuffer reify() {
        return Framebuffer{
            .output_rgb = reinterpret_cast<float3 *>(output_rgb.data_ptr()),
            .output_depth = reinterpret_cast<float *>(output_depth.data_ptr()),
            .output_normal = reinterpret_cast<float3 *>(output_normal.data_ptr()),
            .output_f0 = reinterpret_cast<float3 *>(output_f0.data_ptr()),
            .output_roughness = reinterpret_cast<float *>(output_roughness.data_ptr()),
            .output_transmittance = reinterpret_cast<float *>(output_transmittance.data_ptr()),
            .output_total_transmittance = reinterpret_cast<float *>(output_total_transmittance.data_ptr()),
            .output_brdf = reinterpret_cast<float3 *>(output_brdf.data_ptr()),
            .output_ray_origin = reinterpret_cast<float3 *>(output_ray_origin.data_ptr()),
            .output_ray_direction = reinterpret_cast<float3 *>(output_ray_direction.data_ptr()),
            .output_final = reinterpret_cast<float3 *>(output_final.data_ptr()),
            .output_denoised = reinterpret_cast<float3 *>(output_denoised.data_ptr()),
            .accumulated_rgb = reinterpret_cast<float3 *>(accumulated_rgb.data_ptr()),
            .accumulated_transmittance = reinterpret_cast<float *>(accumulated_transmittance.data_ptr()),
            .accumulated_total_transmittance = reinterpret_cast<float *>(accumulated_total_transmittance.data_ptr()),
            .accumulated_depth = reinterpret_cast<float *>(accumulated_depth.data_ptr()),
            .accumulated_normal = reinterpret_cast<float3 *>(accumulated_normal.data_ptr()),
            .accumulated_f0 = reinterpret_cast<float3 *>(accumulated_f0.data_ptr()),
            .accumulated_roughness = reinterpret_cast<float *>(accumulated_roughness.data_ptr()),
            .accumulated_sample_count = reinterpret_cast<int *>(accumulated_sample_count.data_ptr()),
            .target_diffuse = reinterpret_cast<float3 *>(target_diffuse.data_ptr()),
            .target_glossy = reinterpret_cast<float3 *>(target_glossy.data_ptr()),
            .target_depth = reinterpret_cast<float *>(target_depth.data_ptr()),
            .target_normal = reinterpret_cast<float3 *>(target_normal.data_ptr()),
            .target_f0 = reinterpret_cast<float3 *>(target_f0.data_ptr()),
            .target_roughness = reinterpret_cast<float *>(target_roughness.data_ptr()),
            .num_pixels = static_cast<uint32_t>(output_final.size(1) * output_final.size(2)),
        };
    }

    void reset_accumulators() {
        accumulated_rgb.zero_();
        accumulated_transmittance.zero_();
        accumulated_total_transmittance.zero_();
        accumulated_depth.zero_();
        accumulated_normal.zero_();
        accumulated_f0.zero_();
        accumulated_roughness.zero_();
        accumulated_sample_count.zero_();
    }

    static void bind(torch::Library &m) {
        m.class_<FramebufferDataHolder>("Framebuffer")
            .def_readonly("output_rgb", &FramebufferDataHolder::output_rgb)
            .def_readonly("output_depth", &FramebufferDataHolder::output_depth)
            .def_readonly("output_normal", &FramebufferDataHolder::output_normal)
            .def_readonly("output_f0", &FramebufferDataHolder::output_f0)
            .def_readonly("output_roughness", &FramebufferDataHolder::output_roughness)
            .def_readonly("output_transmittance", &FramebufferDataHolder::output_transmittance)
            .def_readonly("output_total_transmittance", &FramebufferDataHolder::output_total_transmittance)
            .def_readonly("output_brdf", &FramebufferDataHolder::output_brdf)
            .def_readonly("output_ray_origin", &FramebufferDataHolder::output_ray_origin)
            .def_readonly("output_ray_direction", &FramebufferDataHolder::output_ray_direction)
            .def_readonly("output_final", &FramebufferDataHolder::output_final)
            .def_readonly("output_denoised", &FramebufferDataHolder::output_denoised)
            .def_readonly("accumulated_rgb", &FramebufferDataHolder::accumulated_rgb)
            .def_readonly("accumulated_transmittance", &FramebufferDataHolder::accumulated_transmittance)
            .def_readonly("accumulated_total_transmittance", &FramebufferDataHolder::accumulated_total_transmittance)
            .def_readonly("accumulated_depth", &FramebufferDataHolder::accumulated_depth)
            .def_readonly("accumulated_normal", &FramebufferDataHolder::accumulated_normal)
            .def_readonly("accumulated_f0", &FramebufferDataHolder::accumulated_f0)
            .def_readonly("accumulated_roughness", &FramebufferDataHolder::accumulated_roughness)
            .def_readonly("accumulated_sample_count", &FramebufferDataHolder::accumulated_sample_count)
            .def_readonly("target_diffuse", &FramebufferDataHolder::target_diffuse)
            .def_readonly("target_glossy", &FramebufferDataHolder::target_glossy)
            .def_readonly("target_depth", &FramebufferDataHolder::target_depth)
            .def_readonly("target_normal", &FramebufferDataHolder::target_normal)
            .def_readonly("target_f0", &FramebufferDataHolder::target_f0)
            .def_readonly("target_roughness", &FramebufferDataHolder::target_roughness);
    }
};

#endif
