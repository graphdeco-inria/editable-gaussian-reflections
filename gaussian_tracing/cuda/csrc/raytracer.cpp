#include <cstddef>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <string>
#include <tuple>

#include "headers/torch.h"
#include "utils/exception.h"

#include "core/all.h"

#include "optix/bvh_wrapper.h"
#include "optix/denoiser_wrapper.h"
#include "optix/pipeline_wrapper.h"

#include "params.h"

struct Raytracer : torch::CustomClassHolder {
    int width;
    int height;

    Params params_on_host;
    CUdeviceptr params_on_device;

    // * Intrusive points are required to expose data to Python
    c10::intrusive_ptr<CameraDataHolder> camera_data;
    c10::intrusive_ptr<ConfigDataHolder> config_data;
    c10::intrusive_ptr<FramebufferDataHolder> framebuffer_data;
    c10::intrusive_ptr<GaussianDataHolder> gaussian_data;
    c10::intrusive_ptr<MetaDataHolder> meta_data;
    c10::intrusive_ptr<StatsDataHolder> stats_data;
    c10::intrusive_ptr<PPLLDataHolder> ppll_forward_data;
    c10::intrusive_ptr<PPLLDataHolder> ppll_backward_data;

    std::unique_ptr<PipelineWrapper> pipeline_wrapper;
    std::unique_ptr<BVHWrapper> bvh_wrapper;
    std::unique_ptr<DenoiserWrapper> denoiser_wrapper;

    Raytracer(
        int64_t width_, int64_t height_, int64_t num_gaussians, int64_t forward_ppl_size, int64_t backward_ppl_size)
        : width(width_), height(height_), camera_data(c10::make_intrusive<CameraDataHolder>()),
          config_data(c10::make_intrusive<ConfigDataHolder>()),
          framebuffer_data(c10::make_intrusive<FramebufferDataHolder>(width, height)),
          gaussian_data(c10::make_intrusive<GaussianDataHolder>()),
          meta_data(c10::make_intrusive<MetaDataHolder>(width, height)),
          stats_data(c10::make_intrusive<StatsDataHolder>(width, height)),
          ppll_forward_data(c10::make_intrusive<PPLLDataHolder>(width, height, forward_ppl_size)),
          ppll_backward_data(c10::make_intrusive<PPLLDataHolder>(width, height, backward_ppl_size)) {
        if (num_gaussians > 0) {
            gaussian_data->resize(num_gaussians);
        }

        params_on_host.image_width = width;
        params_on_host.image_height = height;
        params_on_host.camera = camera_data->reify();
        params_on_host.config = config_data->reify();
        params_on_host.framebuffer = framebuffer_data->reify();
        params_on_host.gaussians = gaussian_data->reify();
        params_on_host.ppll_forward = ppll_forward_data->reify();
        params_on_host.ppll_backward = ppll_backward_data->reify();
        params_on_host.metadata = meta_data->reify();
        params_on_host.stats = stats_data->reify();

        pipeline_wrapper = std::make_unique<PipelineWrapper>();
        bvh_wrapper = std::make_unique<BVHWrapper>(pipeline_wrapper->context, *config_data, params_on_host);
        denoiser_wrapper = std::make_unique<DenoiserWrapper>(pipeline_wrapper->context, params_on_host);

        // * Transfer params to device
        params_on_host.bvh_handle = bvh_wrapper->tlas_handle;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&params_on_device), sizeof(Params)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(params_on_device), &params_on_host, sizeof(Params), cudaMemcpyHostToDevice));
    }

    void raytrace() {
        meta_data->update();
        stats_data->reset();

        ppll_forward_data->reset();
        ppll_backward_data->reset();

        assert(params_on_device != 0);
        pipeline_wrapper->launch(width, height, params_on_device);

        if (config_data->accumulate_samples.item<bool>()) {
            framebuffer_data->accumulated_sample_count += 1;
        }
    }

    void denoise() { denoiser_wrapper->run(); }

    void reset_accumulators() { framebuffer_data->reset_accumulators(); }

    void update_bvh() { bvh_wrapper->update(); }

    void rebuild_bvh() {
        bvh_wrapper->rebuild();
        params_on_host.bvh_handle = bvh_wrapper->tlas_handle;
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(params_on_device + offsetof(Params, bvh_handle)),
            &params_on_host.bvh_handle,
            sizeof(params_on_host.bvh_handle),
            cudaMemcpyHostToDevice));
    }

    void resize(int64_t new_num_gaussians) {
        gaussian_data->resize(new_num_gaussians);
        params_on_host.gaussians = gaussian_data->reify();
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(params_on_device + offsetof(Params, gaussians)),
            &params_on_host.gaussians,
            sizeof(Gaussians),
            cudaMemcpyHostToDevice));
    }

    static void bind(torch::Library &m) {
        m.class_<Raytracer>("Raytracer")
            .def(torch::init<int64_t, int64_t, int64_t, int64_t, int64_t>())
            .def("raytrace", &Raytracer::raytrace)
            .def("denoise", &Raytracer::denoise)
            .def("reset_accumulators", &Raytracer::reset_accumulators)
            .def("update_bvh", &Raytracer::update_bvh)
            .def("rebuild_bvh", &Raytracer::rebuild_bvh)
            .def("resize", &Raytracer::resize)
            .def("get_camera", [](const c10::intrusive_ptr<Raytracer> &self) { return self->camera_data; })
            .def("get_config", [](const c10::intrusive_ptr<Raytracer> &self) { return self->config_data; })
            .def("get_framebuffer", [](const c10::intrusive_ptr<Raytracer> &self) { return self->framebuffer_data; })
            .def("get_gaussians", [](const c10::intrusive_ptr<Raytracer> &self) { return self->gaussian_data; })
            .def("get_metadata", [](const c10::intrusive_ptr<Raytracer> &self) { return self->meta_data; })
            .def("get_stats", [](const c10::intrusive_ptr<Raytracer> &self) { return self->stats_data; })
            .def(
                "get_ppll_forward_data",
                [](const c10::intrusive_ptr<Raytracer> &self) { return self->ppll_forward_data; })
            .def(
                "get_ppll_backward_data",
                [](const c10::intrusive_ptr<Raytracer> &self) { return self->ppll_backward_data; })

            // * Expose constants for access in Python
            .def_static("MAX_BOUNCES", []() { return (int64_t)MAX_BOUNCES; })
            .def_static("MAX_ALPHA", []() { return (double)MAX_ALPHA; })
            .def_static("ROUGHNESS_DOWNWEIGHT_GRAD", []() { return (bool)ROUGHNESS_DOWNWEIGHT_GRAD; })
            .def_static("ROUGHNESS_DOWNWEIGHT_GRAD_POWER", []() { return (double)ROUGHNESS_DOWNWEIGHT_GRAD_POWER; })

            // * Expose all buffer names for easy looping in Python
            .def(
                "describe_output_buffers",
                [](const c10::intrusive_ptr<Raytracer> &self) {
                    std::vector<std::tuple<std::string, int64_t>> tmp;
                    tmp.emplace_back(std::make_tuple("output_rgb", 3));
                    tmp.emplace_back(std::make_tuple("output_depth", 1));
                    tmp.emplace_back(std::make_tuple("output_normal", 3));
                    tmp.emplace_back(std::make_tuple("output_f0", 3));
                    tmp.emplace_back(std::make_tuple("output_roughness", 1));
                    tmp.emplace_back(std::make_tuple("output_transmittance", 1));
                    tmp.emplace_back(std::make_tuple("output_total_transmittance", 1));
                    tmp.emplace_back(std::make_tuple("output_brdf", 3));
                    tmp.emplace_back(std::make_tuple("output_ray_origin", 3));
                    tmp.emplace_back(std::make_tuple("output_ray_direction", 3));
                    tmp.emplace_back(std::make_tuple("output_final", 3));
                    return tmp;
                })
            .def(
                "describe_accumulation_buffers",
                [](const c10::intrusive_ptr<Raytracer> &self) {
                    std::vector<std::tuple<std::string, int64_t>> tmp;
                    tmp.emplace_back(std::make_tuple("accumulated_rgb", 3));
                    tmp.emplace_back(std::make_tuple("accumulated_transmittance", 1));
                    tmp.emplace_back(std::make_tuple("accumulated_total_transmittance", 1));
                    tmp.emplace_back(std::make_tuple("accumulated_depth", 1));
                    tmp.emplace_back(std::make_tuple("accumulated_normal", 3));
                    tmp.emplace_back(std::make_tuple("accumulated_f0", 3));
                    tmp.emplace_back(std::make_tuple("accumulated_roughness", 1));
                    return tmp;
                })
            .def(
                "describe_target_buffers",
                [](const c10::intrusive_ptr<Raytracer> &self) {
                    std::vector<std::tuple<std::string, int64_t>> tmp;
                    tmp.emplace_back(std::make_tuple("target_depth", 1));
                    tmp.emplace_back(std::make_tuple("target_normal", 3));
                    tmp.emplace_back(std::make_tuple("target_f0", 3));
                    tmp.emplace_back(std::make_tuple("target_roughness", 1));
                    return tmp;
                })

            // * Expose gaussian attributes as well
            .def("describe_gaussian_attributes", [](const c10::intrusive_ptr<Raytracer> &self) {
                std::vector<std::tuple<std::string, int64_t>> tmp;
                tmp.emplace_back(std::make_tuple("gaussian_rgb", 3));
                tmp.emplace_back(std::make_tuple("gaussian_normal", 3));
                tmp.emplace_back(std::make_tuple("gaussian_f0", 3));
                tmp.emplace_back(std::make_tuple("gaussian_roughness", 1));
                tmp.emplace_back(std::make_tuple("gaussian_opacity", 1));
                tmp.emplace_back(std::make_tuple("gaussian_scale", 3));
                tmp.emplace_back(std::make_tuple("gaussian_mean", 3));
                tmp.emplace_back(std::make_tuple("gaussian_rotation", 4));
                return tmp;
            });
    }
};

TORCH_LIBRARY(raytracer, m) {
    CameraDataHolder::bind(m);
    ConfigDataHolder::bind(m);
    FramebufferDataHolder::bind(m);
    GaussianDataHolder::bind(m);
    MetaDataHolder::bind(m);
    PPLLDataHolder::bind(m);
    StatsDataHolder::bind(m);

    Raytracer::bind(m);
}