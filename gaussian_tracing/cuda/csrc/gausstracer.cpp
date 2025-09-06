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

float GetEnvironmentVariableOrDefault(
    const std::string &variable_name, float default_value) {
    const char *value = getenv(variable_name.c_str());
    if (value == 0) {
        return default_value;
    }
    return std::stof(value);
}

struct Raytracer : torch::CustomClassHolder {
    float *m_output_transmittances;

    int m_width;
    int m_height;

    // * Intrusive points are required to expose data to Python
    c10::intrusive_ptr<ConfigDataHolder> config_data;

    Params m_h_params;
    CUdeviceptr m_d_params;

    Tensor m_iteration;

    Tensor m_grads_enabled =
        torch::ones({1}, torch::dtype(torch::kBool).device(torch::kCUDA));
    Tensor m_total_hits =
        torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // PPLL storage for forward pass
    Tensor m_all_gaussian_ids = torch::zeros(
        {PPLL_STORAGE_SIZE}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    Tensor m_all_distances = torch::zeros(
        {PPLL_STORAGE_SIZE},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_all_alphas = torch::zeros(
        {PPLL_STORAGE_SIZE * TILE_SIZE * TILE_SIZE},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_all_gaussvals = torch::zeros(
        {PPLL_STORAGE_SIZE * TILE_SIZE * TILE_SIZE},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_all_local_hits = torch::zeros(
        {PPLL_STORAGE_SIZE * TILE_SIZE * TILE_SIZE, 3},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_all_prev_hits = torch::zeros(
        {PPLL_STORAGE_SIZE}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    Tensor m_prev_hit_per_pixel;

    // PPLL storage for backward pass
    Tensor m_total_hits_for_backprop =
        torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    Tensor m_all_gaussian_ids_for_backprop = torch::zeros(
        {PPLL_STORAGE_SIZE_BACKWARD},
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    Tensor m_all_prev_hits_for_backprop = torch::zeros(
        {PPLL_STORAGE_SIZE_BACKWARD},
        torch::dtype(torch::kInt32).device(torch::kCUDA));

    Tensor m_all_alphas_for_backprop = torch::zeros(
        {PPLL_STORAGE_SIZE_BACKWARD * TILE_SIZE * TILE_SIZE},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_all_local_hits_for_backprop = torch::zeros(
        {PPLL_STORAGE_SIZE_BACKWARD * TILE_SIZE * TILE_SIZE, 3},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_all_distances_for_backprop = torch::zeros(
        {PPLL_STORAGE_SIZE_BACKWARD * TILE_SIZE * TILE_SIZE},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_all_Ts_for_backprop = torch::zeros(
        {PPLL_STORAGE_SIZE_BACKWARD * TILE_SIZE * TILE_SIZE},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_all_gaussvals_for_backprop = torch::zeros(
        {PPLL_STORAGE_SIZE_BACKWARD * TILE_SIZE * TILE_SIZE},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_prev_hit_per_pixel_for_backprop;

    // Gaussian attributes
    Tensor m_gaussian_rgb;
    Tensor m_dL_drgb;
    Tensor m_gaussian_opacity;
    Tensor m_dL_dopacity;
    Tensor m_gaussian_scales;
    Tensor m_dL_dscales;
    Tensor m_gaussian_rotations;
    Tensor m_dL_drotations;
    Tensor m_gaussian_means;
    Tensor m_dL_dmeans;
    // New attributes
    Tensor m_gaussian_exp_power;
    Tensor m_dL_dexp_powers;
    Tensor m_gaussian_lod_mean;
    Tensor m_dL_dgaussian_lod_mean;
    Tensor m_gaussian_lod_scale;
    Tensor m_dL_dgaussian_lod_scale;
    // Attached attributes
    Tensor m_gaussian_position;
    Tensor m_dL_dgaussian_position;
    Tensor m_gaussian_normal;
    Tensor m_dL_dgaussian_normal;
    Tensor m_gaussian_f0;
    Tensor m_dL_dgaussian_f0;
    Tensor m_gaussian_roughness;
    Tensor m_dL_dgaussian_roughness;

    // Camera buffers
    Tensor m_vertical_fov_radians =
        torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_camera_rotation_c2w = torch::zeros(
        {3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_camera_rotation_w2c = torch::zeros(
        {3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_camera_position_world =
        torch::zeros({3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_camera_znear =
        torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_camera_zfar =
        torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    Tensor m_max_lod_size =
        torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor m_init_blur_sigma =
        torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Output buffers
    Tensor m_output_rgb;
    Tensor m_output_t;
    Tensor m_output_incident_radiance;
    Tensor m_output_position;
    Tensor m_output_depth;
    Tensor m_output_normal;
    Tensor m_output_f0;
    Tensor m_output_roughness;
    Tensor m_output_distortion;
    Tensor m_output_lod_mean;
    Tensor m_output_lod_scale;
    Tensor m_output_ray_lod;
    Tensor m_output_brdf;
    Tensor m_output_diffuse_irradiance;
    Tensor m_output_glossy_irradiance;

    // Sample accumulation
    Tensor m_accumulated_rgb;
    Tensor m_accumulated_normal;
    Tensor m_accumulated_depth;
    Tensor m_accumulated_f0;
    Tensor m_accumulated_roughness;
    Tensor m_accumulated_sample_count;

    // Target buffers
    Tensor m_target_rgb;
    Tensor m_target_diffuse;
    Tensor m_target_glossy;
    Tensor m_target_position;
    Tensor m_target_depth;
    Tensor m_target_normal;
    Tensor m_target_f0;
    Tensor m_target_roughness;
    Tensor m_target_brdf;
    Tensor m_target_diffuse_irradiance;
    Tensor m_target_glossy_irradiance;
    Tensor m_loss_tensor;

    // BRDF computation buffers (for visual debugging)
    Tensor m_output_ray_origin;
    Tensor m_output_ray_direction;
    Tensor m_output_lut_values;
    Tensor m_output_n_dot_v;
    Tensor m_output_effective_reflection_position;
    Tensor m_output_effective_reflection_normal;
    Tensor m_output_effective_F0;
    Tensor m_output_effective_roughness;
    Tensor m_output_effective_normal;

    // Other buffers
    Tensor m_random_seeds;
    Tensor m_t_maxes;
    Tensor m_t_mins;

    Tensor m_gaussian_total_weight;
    Tensor m_densification_gradient_diffuse;
    Tensor m_densification_gradient_glossy;

    Tensor m_lut;
    Tensor m_cheap_approx =
        torch::zeros({1}, torch::dtype(torch::kBool).device(torch::kCUDA));

    Tensor m_num_hits_per_pixel;
    Tensor m_num_traversed_per_pixel;

    std::unique_ptr<PipelineWrapper> pipeline_wrapper;
    std::unique_ptr<BVHWrapper> bvh_wrapper;
    std::unique_ptr<DenoiserWrapper> denoiser_wrapper;

    Raytracer(
        int64_t image_width, int64_t image_height, int64_t num_gaussians) {
        if (num_gaussians <= 0) {
            num_gaussians =
                1; // default to 1 gaussian to avoid degenerate tensor shapes
        }

        m_width = image_width;
        m_height = image_height;

        config_data = c10::make_intrusive<ConfigDataHolder>();

        // Inititalize gaussian parameters
        m_gaussian_rgb = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_drgb = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_gaussian_opacity = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dopacity = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_gaussian_scales = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dscales = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_gaussian_rotations = torch::zeros(
            {num_gaussians, 4},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_drotations = torch::zeros(
            {num_gaussians, 4},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_gaussian_means = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dmeans = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_gaussian_total_weight = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_densification_gradient_diffuse = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_densification_gradient_glossy = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        //
        m_gaussian_lod_mean = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dgaussian_lod_mean = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_gaussian_lod_scale = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dgaussian_lod_scale = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        //
        m_gaussian_position = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dgaussian_position = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_gaussian_normal = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dgaussian_normal = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_gaussian_f0 = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dgaussian_f0 = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_gaussian_roughness = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dgaussian_roughness = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));

        // Create output and target buffers
        m_output_rgb = torch::zeros(
            {MAX_BOUNCES + 2, image_height, image_width, 3},
            torch::dtype(torch::kFloat32)
                .device(
                    torch::kCUDA)); // last pass is the sum of all light bounces
        m_accumulated_rgb = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_accumulated_normal = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_accumulated_depth = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_accumulated_f0 = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_accumulated_roughness = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));

        m_accumulated_sample_count =
            torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        m_target_rgb = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_t = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 2},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_target_diffuse = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_target_glossy = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_ray_origin = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_ray_direction = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_position = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_target_position = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_depth = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_target_depth = torch::zeros(
            {image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_normal = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_target_normal = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_f0 = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_target_f0 = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_roughness = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_target_roughness = torch::zeros(
            {image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_brdf = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_incident_radiance = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));

        m_iteration =
            torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        m_random_seeds = torch::randint(
            0,
            1000000000,
            {image_height, image_width, 1},
            torch::dtype(torch::kInt32).device(torch::kCUDA));

        m_t_maxes = torch::zeros(
            {m_height, m_width},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_t_maxes.fill_(100000.0);

        m_t_mins = torch::zeros(
            {m_height, m_width},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_t_mins.fill_(0.0);

        m_num_hits_per_pixel = torch::zeros(
            {m_height, m_width},
            torch::dtype(torch::kInt32).device(torch::kCUDA));
        m_num_traversed_per_pixel = torch::zeros(
            {m_height, m_width},
            torch::dtype(torch::kInt32).device(torch::kCUDA));
        m_prev_hit_per_pixel = torch::zeros(
            {m_height, m_width},
            torch::dtype(torch::kInt32).device(torch::kCUDA));
        m_prev_hit_per_pixel.fill_(999999999);
        m_prev_hit_per_pixel_for_backprop = torch::zeros(
            {m_height, m_width},
            torch::dtype(torch::kInt32).device(torch::kCUDA));
        m_prev_hit_per_pixel_for_backprop.fill_(999999999);

        { // Create params object
            m_h_params.image_width = m_width;
            m_h_params.image_height = m_height;

            // Render settings
            m_h_params.denoise = reinterpret_cast<bool *>(m_denoise.data_ptr());
            m_h_params.num_bounces =
                reinterpret_cast<int *>(m_num_bounces.data_ptr());

            // Configuration
            m_h_params.config = config_data->reify();
            set_losses(false);

            // Camera buffers
            m_h_params.vertical_fov_radians =
                reinterpret_cast<float *>(m_vertical_fov_radians.data_ptr());
            m_h_params.camera_znear =
                reinterpret_cast<float *>(m_camera_znear.data_ptr());
            m_h_params.camera_zfar =
                reinterpret_cast<float *>(m_camera_zfar.data_ptr());
            m_h_params.max_lod_size =
                reinterpret_cast<float *>(m_max_lod_size.data_ptr());

            m_h_params.init_blur_sigma =
                reinterpret_cast<float *>(m_init_blur_sigma.data_ptr());
            m_h_params.camera_rotation_c2w =
                reinterpret_cast<float3 *>(m_camera_rotation_c2w.data_ptr());
            m_h_params.camera_rotation_w2c =
                reinterpret_cast<float3 *>(m_camera_rotation_w2c.data_ptr());
            m_h_params.camera_position_world =
                reinterpret_cast<float3 *>(m_camera_position_world.data_ptr());

            m_h_params.all_gaussian_ids =
                reinterpret_cast<uint32_t *>(m_all_gaussian_ids.data_ptr());
            m_h_params.all_distances =
                reinterpret_cast<float *>(m_all_distances.data_ptr());
            // m_h_params.all_half_chord_lengths =
            // reinterpret_cast<float*>(m_all_half_chord_lengths.data_ptr());

            m_h_params.all_alphas =
                reinterpret_cast<float *>(m_all_alphas.data_ptr());
            m_h_params.all_gaussvals =
                reinterpret_cast<float *>(m_all_gaussvals.data_ptr());
            m_h_params.all_local_hits =
                reinterpret_cast<float3 *>(m_all_local_hits.data_ptr());

            m_h_params.all_prev_hits =
                reinterpret_cast<uint32_t *>(m_all_prev_hits.data_ptr());
            m_h_params.prev_hit_per_pixel =
                reinterpret_cast<uint32_t *>(m_prev_hit_per_pixel.data_ptr());
            m_h_params.total_hits =
                reinterpret_cast<uint32_t *>(m_total_hits.data_ptr());

            m_h_params.all_gaussian_ids_for_backprop =
                reinterpret_cast<uint32_t *>(
                    m_all_gaussian_ids_for_backprop.data_ptr());
            m_h_params.all_alphas_for_backprop =
                reinterpret_cast<float *>(m_all_alphas_for_backprop.data_ptr());
            m_h_params.all_local_hits_for_backprop = reinterpret_cast<float3 *>(
                m_all_local_hits_for_backprop.data_ptr());
            m_h_params.all_Ts_for_backprop =
                reinterpret_cast<float *>(m_all_Ts_for_backprop.data_ptr());
            m_h_params.all_distances_for_backprop = reinterpret_cast<float *>(
                m_all_distances_for_backprop.data_ptr());
            m_h_params.all_gaussvals_for_backprop = reinterpret_cast<float *>(
                m_all_gaussvals_for_backprop.data_ptr());
            m_h_params.all_prev_hits_for_backprop =
                reinterpret_cast<uint32_t *>(
                    m_all_prev_hits_for_backprop.data_ptr());
            m_h_params.prev_hit_per_pixel_for_backprop =
                reinterpret_cast<uint32_t *>(
                    m_prev_hit_per_pixel_for_backprop.data_ptr());
            m_h_params.total_hits_for_backprop = reinterpret_cast<uint32_t *>(
                m_total_hits_for_backprop.data_ptr());

            m_gaussian_rgb.mutable_grad() = m_dL_drgb;
            m_gaussian_opacity.mutable_grad() = m_dL_dopacity;
            m_gaussian_scales.mutable_grad() = m_dL_dscales;
            m_gaussian_rotations.mutable_grad() = m_dL_drotations;
            m_gaussian_means.mutable_grad() = m_dL_dmeans;
            m_gaussian_lod_mean.mutable_grad() = m_dL_dgaussian_lod_mean;
            m_gaussian_lod_scale.mutable_grad() = m_dL_dgaussian_lod_scale;
            m_gaussian_position.mutable_grad() = m_dL_dgaussian_position;
            m_gaussian_normal.mutable_grad() = m_dL_dgaussian_normal;
            m_gaussian_f0.mutable_grad() = m_dL_dgaussian_f0;
            m_gaussian_roughness.mutable_grad() = m_dL_dgaussian_roughness;

            // Gaussian parameters
            m_h_params.gaussian_rgb =
                reinterpret_cast<float3 *>(m_gaussian_rgb.data_ptr());
            m_h_params.dL_drgb =
                reinterpret_cast<float3 *>(m_dL_drgb.data_ptr());
            m_h_params.gaussian_opacity =
                reinterpret_cast<float *>(m_gaussian_opacity.data_ptr());
            m_h_params.dL_dopacity =
                reinterpret_cast<float *>(m_dL_dopacity.data_ptr());
            m_h_params.gaussian_rotations =
                reinterpret_cast<float4 *>(m_gaussian_rotations.data_ptr());
            m_h_params.dL_drotations =
                reinterpret_cast<float4 *>(m_dL_drotations.data_ptr());
            m_h_params.gaussian_scales =
                reinterpret_cast<float3 *>(m_gaussian_scales.data_ptr());
            m_h_params.dL_dscales =
                reinterpret_cast<float3 *>(m_dL_dscales.data_ptr());
            m_h_params.gaussian_means =
                reinterpret_cast<float3 *>(m_gaussian_means.data_ptr());
            m_h_params.dL_dmeans =
                reinterpret_cast<float3 *>(m_dL_dmeans.data_ptr());
            m_h_params.gaussian_lod_mean =
                reinterpret_cast<float *>(m_gaussian_lod_mean.data_ptr());
            m_h_params.dL_dgaussian_lod_mean =
                reinterpret_cast<float *>(m_dL_dgaussian_lod_mean.data_ptr());
            m_h_params.gaussian_lod_scale =
                reinterpret_cast<float *>(m_gaussian_lod_scale.data_ptr());
            m_h_params.dL_dgaussian_lod_scale =
                reinterpret_cast<float *>(m_dL_dgaussian_lod_scale.data_ptr());
            m_h_params.gaussian_position =
                reinterpret_cast<float3 *>(m_gaussian_position.data_ptr());
            m_h_params.dL_dgaussian_position =
                reinterpret_cast<float3 *>(m_dL_dgaussian_position.data_ptr());
            m_h_params.gaussian_normal =
                reinterpret_cast<float3 *>(m_gaussian_normal.data_ptr());
            m_h_params.dL_dgaussian_normal =
                reinterpret_cast<float3 *>(m_dL_dgaussian_normal.data_ptr());
            m_h_params.gaussian_f0 =
                reinterpret_cast<float3 *>(m_gaussian_f0.data_ptr());
            m_h_params.dL_dgaussian_f0 =
                reinterpret_cast<float3 *>(m_dL_dgaussian_f0.data_ptr());
            m_h_params.gaussian_roughness =
                reinterpret_cast<float *>(m_gaussian_roughness.data_ptr());
            m_h_params.dL_dgaussian_roughness =
                reinterpret_cast<float *>(m_dL_dgaussian_roughness.data_ptr());

            m_h_params.gaussian_total_weight =
                reinterpret_cast<float *>(m_gaussian_total_weight.data_ptr());
            m_h_params.densification_gradient_diffuse =
                reinterpret_cast<float3 *>(
                    m_densification_gradient_diffuse.data_ptr());
            m_h_params.densification_gradient_glossy =
                reinterpret_cast<float3 *>(
                    m_densification_gradient_glossy.data_ptr());

            // Output and target buffers
            m_h_params.output_rgb =
                reinterpret_cast<float3 *>(m_output_rgb.data_ptr());
            m_h_params.accumulated_rgb =
                reinterpret_cast<float3 *>(m_accumulated_rgb.data_ptr());
            m_h_params.accumulated_normal =
                reinterpret_cast<float3 *>(m_accumulated_normal.data_ptr());
            m_h_params.accumulated_depth =
                reinterpret_cast<float *>(m_accumulated_depth.data_ptr());
            m_h_params.accumulated_f0 =
                reinterpret_cast<float3 *>(m_accumulated_f0.data_ptr());
            m_h_params.accumulated_roughness =
                reinterpret_cast<float *>(m_accumulated_roughness.data_ptr());

            m_h_params.accumulated_sample_count =
                reinterpret_cast<int *>(m_accumulated_sample_count.data_ptr());
            m_h_params.target_rgb =
                reinterpret_cast<float3 *>(m_target_rgb.data_ptr());
            m_h_params.output_t =
                reinterpret_cast<float2 *>(m_output_t.data_ptr());
            m_h_params.target_diffuse =
                reinterpret_cast<float3 *>(m_target_diffuse.data_ptr());
            m_h_params.target_glossy =
                reinterpret_cast<float3 *>(m_target_glossy.data_ptr());
            m_h_params.output_ray_origin =
                reinterpret_cast<float3 *>(m_output_ray_origin.data_ptr());
            m_h_params.output_ray_direction =
                reinterpret_cast<float3 *>(m_output_ray_direction.data_ptr());
            m_h_params.output_incident_radiance = reinterpret_cast<float3 *>(
                m_output_incident_radiance.data_ptr());
            m_h_params.output_position =
                reinterpret_cast<float3 *>(m_output_position.data_ptr());
            m_h_params.target_position =
                reinterpret_cast<float3 *>(m_target_position.data_ptr());
            m_h_params.output_depth =
                reinterpret_cast<float *>(m_output_depth.data_ptr());
            m_h_params.target_depth =
                reinterpret_cast<float *>(m_target_depth.data_ptr());
            m_h_params.output_normal =
                reinterpret_cast<float3 *>(m_output_normal.data_ptr());
            m_h_params.target_normal =
                reinterpret_cast<float3 *>(m_target_normal.data_ptr());
            m_h_params.output_f0 =
                reinterpret_cast<float3 *>(m_output_f0.data_ptr());
            m_h_params.target_f0 =
                reinterpret_cast<float3 *>(m_target_f0.data_ptr());
            m_h_params.output_roughness =
                reinterpret_cast<float *>(m_output_roughness.data_ptr());
            m_h_params.target_roughness =
                reinterpret_cast<float *>(m_target_roughness.data_ptr());
            m_h_params.output_brdf =
                reinterpret_cast<float3 *>(m_output_brdf.data_ptr());

            // Other buffers
            m_h_params.t_maxes =
                reinterpret_cast<float *>(m_t_maxes.data_ptr());
            m_h_params.t_mins = reinterpret_cast<float *>(m_t_mins.data_ptr());
            m_h_params.random_seeds =
                reinterpret_cast<uint32_t *>(m_random_seeds.data_ptr());
            m_h_params.iteration =
                reinterpret_cast<int *>(m_iteration.data_ptr());
            m_h_params.grads_enabled =
                reinterpret_cast<bool *>(m_grads_enabled.data_ptr());
            m_h_params.cheap_approx =
                reinterpret_cast<bool *>(m_cheap_approx.data_ptr());
            m_h_params.num_hits_per_pixel =
                reinterpret_cast<int *>(m_num_hits_per_pixel.data_ptr());
            m_h_params.num_traversed_per_pixel =
                reinterpret_cast<int *>(m_num_traversed_per_pixel.data_ptr());
        }

        pipeline_wrapper = std::make_unique<PipelineWrapper>();
        bvh_wrapper = std::make_unique<BVHWrapper>(
            pipeline_wrapper->context,
            m_gaussian_means,
            m_gaussian_scales,
            m_gaussian_rotations,
            m_gaussian_opacity,
            *config_data);
        denoiser_wrapper = std::make_unique<DenoiserWrapper>(
            pipeline_wrapper->context,
            m_h_params,
            m_output_rgb,
            m_output_normal);
        cudaDeviceSynchronize();

        // Set TLAS and copy
        m_h_params.handle = bvh_wrapper->tlas_handle;
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&m_d_params), sizeof(Params)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_d_params),
            &m_h_params,
            sizeof(Params),
            cudaMemcpyHostToDevice));
    }

    void set_losses(bool updateParams = true) {
        m_h_params.diffuse_loss_weight =
            GetEnvironmentVariableOrDefault("DIFFUSE_LOSS_WEIGHT", 1.0f);
        m_h_params.glossy_loss_weight =
            GetEnvironmentVariableOrDefault("GLOSSY_LOSS_WEIGHT", 1.0f);
        m_h_params.normal_loss_weight =
            GetEnvironmentVariableOrDefault("NORMAL_LOSS_WEIGHT", 1.0f);
        m_h_params.position_loss_weight =
            GetEnvironmentVariableOrDefault("POSITION_LOSS_WEIGHT", 1.0f);
        m_h_params.f0_loss_weight =
            GetEnvironmentVariableOrDefault("F0_LOSS_WEIGHT", 1.0f);
        m_h_params.roughness_loss_weight =
            GetEnvironmentVariableOrDefault("ROUGHNESS_LOSS_WEIGHT", 1.0f);

        printf("diffuse_loss_weight: %f\n", m_h_params.diffuse_loss_weight);
        printf("glossy_loss_weight: %f\n", m_h_params.glossy_loss_weight);
        printf("normal_loss_weight: %f\n", m_h_params.normal_loss_weight);
        printf("position_loss_weight: %f\n", m_h_params.position_loss_weight);
        printf("f0_loss_weight: %f\n", m_h_params.f0_loss_weight);
        printf("roughness_loss_weight: %f\n", m_h_params.roughness_loss_weight);

        if (updateParams) {
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(m_d_params),
                &m_h_params,
                sizeof(Params),
                cudaMemcpyHostToDevice));
        }
    }

    void resize(int64_t num_new_gaussians) {
        printf("Resizing to %ld\n", num_new_gaussians);
        m_gaussian_rgb.resize_({num_new_gaussians, 3});
        m_dL_drgb.resize_({num_new_gaussians, 3});
        m_h_params.gaussian_rgb =
            reinterpret_cast<float3 *>(m_gaussian_rgb.data_ptr());
        m_h_params.dL_drgb = reinterpret_cast<float3 *>(m_dL_drgb.data_ptr());
        //
        m_gaussian_opacity.resize_({num_new_gaussians, 1});
        m_dL_dopacity.resize_({num_new_gaussians, 1});
        m_h_params.gaussian_opacity =
            reinterpret_cast<float *>(m_gaussian_opacity.data_ptr());
        m_h_params.dL_dopacity =
            reinterpret_cast<float *>(m_dL_dopacity.data_ptr());
        //
        m_gaussian_rotations.resize_({num_new_gaussians, 4});
        m_dL_drotations.resize_({num_new_gaussians, 4});
        m_h_params.gaussian_rotations =
            reinterpret_cast<float4 *>(m_gaussian_rotations.data_ptr());
        m_h_params.dL_drotations =
            reinterpret_cast<float4 *>(m_dL_drotations.data_ptr());
        //
        m_gaussian_scales.resize_({num_new_gaussians, 3});
        m_dL_dscales.resize_({num_new_gaussians, 3});
        m_h_params.gaussian_scales =
            reinterpret_cast<float3 *>(m_gaussian_scales.data_ptr());
        m_h_params.dL_dscales =
            reinterpret_cast<float3 *>(m_dL_dscales.data_ptr());
        //
        m_gaussian_means.resize_({num_new_gaussians, 3});
        m_dL_dmeans.resize_({num_new_gaussians, 3});
        m_h_params.gaussian_means =
            reinterpret_cast<float3 *>(m_gaussian_means.data_ptr());
        m_h_params.dL_dmeans =
            reinterpret_cast<float3 *>(m_dL_dmeans.data_ptr());
        //
        m_gaussian_total_weight.resize_({num_new_gaussians, 1});
        m_h_params.gaussian_total_weight =
            reinterpret_cast<float *>(m_gaussian_total_weight.data_ptr());
        m_densification_gradient_diffuse.resize_({num_new_gaussians, 3});
        m_h_params.densification_gradient_diffuse = reinterpret_cast<float3 *>(
            m_densification_gradient_diffuse.data_ptr());
        m_densification_gradient_glossy.resize_({num_new_gaussians, 3});
        m_h_params.densification_gradient_glossy = reinterpret_cast<float3 *>(
            m_densification_gradient_glossy.data_ptr());
        m_gaussian_lod_mean.resize_({num_new_gaussians, 1});
        m_dL_dgaussian_lod_mean.resize_({num_new_gaussians, 1});
        m_h_params.gaussian_lod_mean =
            reinterpret_cast<float *>(m_gaussian_lod_mean.data_ptr());
        m_h_params.dL_dgaussian_lod_mean =
            reinterpret_cast<float *>(m_dL_dgaussian_lod_mean.data_ptr());
        m_gaussian_lod_scale.resize_({num_new_gaussians, 1});
        m_dL_dgaussian_lod_scale.resize_({num_new_gaussians, 1});
        m_h_params.gaussian_lod_scale =
            reinterpret_cast<float *>(m_gaussian_lod_scale.data_ptr());
        m_h_params.dL_dgaussian_lod_scale =
            reinterpret_cast<float *>(m_dL_dgaussian_lod_scale.data_ptr());
        //
        m_gaussian_position.resize_({num_new_gaussians, 3});
        m_dL_dgaussian_position.resize_({num_new_gaussians, 3});
        m_h_params.gaussian_position =
            reinterpret_cast<float3 *>(m_gaussian_position.data_ptr());
        m_h_params.dL_dgaussian_position =
            reinterpret_cast<float3 *>(m_dL_dgaussian_position.data_ptr());
        m_gaussian_normal.resize_({num_new_gaussians, 3});
        m_dL_dgaussian_normal.resize_({num_new_gaussians, 3});
        m_h_params.gaussian_normal =
            reinterpret_cast<float3 *>(m_gaussian_normal.data_ptr());
        m_h_params.dL_dgaussian_normal =
            reinterpret_cast<float3 *>(m_dL_dgaussian_normal.data_ptr());
        m_gaussian_f0.resize_({num_new_gaussians, 3});
        m_dL_dgaussian_f0.resize_({num_new_gaussians, 3});
        m_h_params.gaussian_f0 =
            reinterpret_cast<float3 *>(m_gaussian_f0.data_ptr());
        m_h_params.dL_dgaussian_f0 =
            reinterpret_cast<float3 *>(m_dL_dgaussian_f0.data_ptr());
        m_gaussian_roughness.resize_({num_new_gaussians, 1});
        m_dL_dgaussian_roughness.resize_({num_new_gaussians, 1});
        m_h_params.gaussian_roughness =
            reinterpret_cast<float *>(m_gaussian_roughness.data_ptr());
        m_h_params.dL_dgaussian_roughness =
            reinterpret_cast<float *>(m_dL_dgaussian_roughness.data_ptr());

        cudaDeviceSynchronize();
    }

    void rebuild_bvh() {
        bvh_wrapper->rebuild(
            m_gaussian_means,
            m_gaussian_scales,
            m_gaussian_rotations,
            m_gaussian_opacity);
        m_h_params.handle = bvh_wrapper->tlas_handle;

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_d_params),
            &m_h_params,
            sizeof(Params),
            cudaMemcpyHostToDevice));
    }

    void update_bvh() {
        bvh_wrapper->update(
            m_gaussian_means,
            m_gaussian_scales,
            m_gaussian_rotations,
            m_gaussian_opacity);
    }

    Tensor m_denoise =
        torch::tensor({false}, torch::dtype(torch::kBool).device(torch::kCUDA));
    Tensor m_accumulate =
        torch::tensor({false}, torch::dtype(torch::kBool).device(torch::kCUDA));
    Tensor m_denoise_glossy =
        torch::tensor({false}, torch::dtype(torch::kBool).device(torch::kCUDA));
    Tensor m_num_bounces = torch::tensor(
        {MAX_BOUNCES}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    void raytrace() {
        bool grads_enabled = torch::GradMode::is_enabled();
        m_grads_enabled.fill_(grads_enabled);

        m_num_hits_per_pixel.zero_();
        m_num_traversed_per_pixel.zero_();
        m_prev_hit_per_pixel.fill_(999999999);
        m_total_hits.fill_(0);
        m_prev_hit_per_pixel_for_backprop.fill_(999999999);
        m_total_hits_for_backprop.fill_(0);

        assert(m_d_params != 0);
        pipeline_wrapper->launch(m_width, m_height, m_d_params);

        // used to seed the random noise, need to increment every sample
        m_iteration += 1;

        if (!torch::GradMode::is_enabled() && m_denoise.item<bool>()) {
            denoiser_wrapper->run();
        }

        if (!m_accumulate.item<bool>()) {
            m_accumulated_rgb.zero_();
            m_accumulated_normal.zero_();
            m_accumulated_depth.zero_();
            m_accumulated_roughness.zero_();
            m_accumulated_f0.zero_();
            m_accumulated_sample_count.zero_();
        } else {
            m_accumulated_sample_count[0] += 1;
        }
    }

    void set_camera(
        const torch::Tensor
            &camera_rotation_c2w, // 3x3 matrix, blender convention (not colmap)
        const torch::Tensor &camera_position_world, // 3 vector
        double vertical_fov_radians,
        double znear,
        double zfar,
        double max_lod_size) {
        m_camera_rotation_c2w.copy_(camera_rotation_c2w);
        m_camera_position_world.copy_(camera_position_world);
        // set the camera_rotation_w2c to the transpose
        m_camera_rotation_w2c.copy_(camera_rotation_c2w.transpose(0, 1));
        m_vertical_fov_radians.fill_((float)vertical_fov_radians);
        m_camera_znear.fill_((float)znear);
        m_camera_zfar.fill_((float)zfar);
        m_max_lod_size.fill_((float)max_lod_size);
    }

    static void bind(torch::Library &m) {
        m.class_<Raytracer>("Raytracer")
            .def(torch::init<int64_t, int64_t, int64_t>())
            .def("rebuild_bvh", &Raytracer::rebuild_bvh)
            .def("update_bvh", &Raytracer::update_bvh)
            .def("set_losses", &Raytracer::set_losses)
            .def("raytrace", &Raytracer::raytrace)
            .def("set_camera", &Raytracer::set_camera)
            .def("resize", &Raytracer::resize)
            // Render settings
            .def_readonly("denoise", &Raytracer::m_denoise)
            .def_readonly("accumulate", &Raytracer::m_accumulate)
            .def_readonly("num_bounces", &Raytracer::m_num_bounces)
            // Camera params
            .def_readonly(
                "camera_rotation_c2w", &Raytracer::m_camera_rotation_c2w)
            .def_readonly(
                "camera_rotation_w2c", &Raytracer::m_camera_rotation_w2c)
            .def_readonly(
                "camera_position_world", &Raytracer::m_camera_position_world)
            .def_readonly(
                "vertical_fov_radians", &Raytracer::m_vertical_fov_radians)
            .def_readonly("init_blur_sigma", &Raytracer::m_init_blur_sigma)
            // Gaussian params
            .def_readonly("gaussian_rgb", &Raytracer::m_gaussian_rgb)
            .def_readonly("dL_drgb", &Raytracer::m_dL_drgb)
            .def_readonly("gaussian_opacity", &Raytracer::m_gaussian_opacity)
            .def_readonly("dL_dopacity", &Raytracer::m_dL_dopacity)
            .def_readonly("gaussian_scales", &Raytracer::m_gaussian_scales)
            .def_readonly("dL_dscales", &Raytracer::m_dL_dscales)
            .def_readonly(
                "gaussian_rotations", &Raytracer::m_gaussian_rotations)
            .def_readonly("dL_drotations", &Raytracer::m_dL_drotations)
            .def_readonly("gaussian_means", &Raytracer::m_gaussian_means)
            .def_readonly("dL_dmeans", &Raytracer::m_dL_dmeans)
            .def_readonly(
                "gaussian_exp_power", &Raytracer::m_gaussian_exp_power)
            .def_readonly("dL_dexp_powers", &Raytracer::m_dL_dexp_powers)
            .def_readonly("gaussian_lod_mean", &Raytracer::m_gaussian_lod_mean)
            .def_readonly(
                "dL_dgaussian_lod_mean", &Raytracer::m_dL_dgaussian_lod_mean)
            .def_readonly(
                "gaussian_lod_scale", &Raytracer::m_gaussian_lod_scale)
            .def_readonly(
                "dL_dgaussian_lod_scale", &Raytracer::m_dL_dgaussian_lod_scale)
            .def_readonly("gaussian_position", &Raytracer::m_gaussian_position)
            .def_readonly(
                "dL_dgaussian_position", &Raytracer::m_dL_dgaussian_position)
            .def_readonly("gaussian_normal", &Raytracer::m_gaussian_normal)
            .def_readonly(
                "dL_dgaussian_normal", &Raytracer::m_dL_dgaussian_normal)
            .def_readonly("gaussian_f0", &Raytracer::m_gaussian_f0)
            .def_readonly("dL_dgaussian_f0", &Raytracer::m_dL_dgaussian_f0)
            .def_readonly(
                "gaussian_roughness", &Raytracer::m_gaussian_roughness)
            .def_readonly(
                "dL_dgaussian_roughness", &Raytracer::m_dL_dgaussian_roughness)
            .def_readonly(
                "gaussian_total_weight", &Raytracer::m_gaussian_total_weight)
            .def_readonly(
                "densification_gradient_diffuse",
                &Raytracer::m_densification_gradient_diffuse)
            .def_readonly(
                "densification_gradient_glossy",
                &Raytracer::m_densification_gradient_glossy)
            // Output buffers
            .def_readonly("output_rgb", &Raytracer::m_output_rgb)
            .def_readonly("accumulated_rgb", &Raytracer::m_accumulated_rgb)
            .def_readonly(
                "accumulated_normal", &Raytracer::m_accumulated_normal)
            .def_readonly("accumulated_depth", &Raytracer::m_accumulated_depth)
            .def_readonly("accumulated_f0", &Raytracer::m_accumulated_f0)
            .def_readonly(
                "accumulated_roughness", &Raytracer::m_accumulated_roughness)
            .def_readonly(
                "accumulated_sample_count",
                &Raytracer::m_accumulated_sample_count)
            .def_readonly("output_t", &Raytracer::m_output_t)
            .def_readonly(
                "output_incident_radiance",
                &Raytracer::m_output_incident_radiance)
            .def_readonly("output_position", &Raytracer::m_output_position)
            .def_readonly("output_depth", &Raytracer::m_output_depth)
            .def_readonly("output_normal", &Raytracer::m_output_normal)
            .def_readonly("output_f0", &Raytracer::m_output_f0)
            .def_readonly("output_roughness", &Raytracer::m_output_roughness)
            .def_readonly("output_distortion", &Raytracer::m_output_distortion)
            .def_readonly("output_brdf", &Raytracer::m_output_brdf)
            .def_readonly(
                "output_diffuse_irradiance",
                &Raytracer::m_output_diffuse_irradiance)
            .def_readonly(
                "output_glossy_irradiance",
                &Raytracer::m_output_glossy_irradiance)
            // Debug output buffers
            .def_readonly("output_ray_origin", &Raytracer::m_output_ray_origin)
            .def_readonly(
                "output_ray_direction", &Raytracer::m_output_ray_direction)
            .def_readonly("output_lut_values", &Raytracer::m_output_lut_values)
            .def_readonly("output_n_dot_v", &Raytracer::m_output_n_dot_v)
            .def_readonly(
                "output_effective_reflection_position",
                &Raytracer::m_output_effective_reflection_position)
            .def_readonly(
                "output_effective_reflection_normal",
                &Raytracer::m_output_effective_reflection_normal)
            .def_readonly(
                "output_effective_F0", &Raytracer::m_output_effective_F0)
            .def_readonly(
                "output_effective_roughness",
                &Raytracer::m_output_effective_roughness)
            .def_readonly(
                "output_effective_normal",
                &Raytracer::m_output_effective_normal)
            .def_readonly("output_lod_mean", &Raytracer::m_output_lod_mean)
            .def_readonly("output_lod_scale", &Raytracer::m_output_lod_scale)
            .def_readonly("output_ray_lod", &Raytracer::m_output_ray_lod)
            // Target buffers
            .def_readonly("target_rgb", &Raytracer::m_target_rgb)
            .def_readonly("target_diffuse", &Raytracer::m_target_diffuse)
            .def_readonly("target_glossy", &Raytracer::m_target_glossy)
            .def_readonly("target_position", &Raytracer::m_target_position)
            .def_readonly("target_depth", &Raytracer::m_target_depth)
            .def_readonly("target_normal", &Raytracer::m_target_normal)
            .def_readonly("target_f0", &Raytracer::m_target_f0)
            .def_readonly("target_roughness", &Raytracer::m_target_roughness)
            .def_readonly("target_brdf", &Raytracer::m_target_brdf)
            .def_readonly(
                "target_diffuse_irradiance",
                &Raytracer::m_target_diffuse_irradiance)
            .def_readonly(
                "target_glossy_irradiance",
                &Raytracer::m_target_glossy_irradiance)
            .def_readonly("loss_tensor", &Raytracer::m_loss_tensor)
            // Other buffers
            .def_readonly(
                "num_hits_per_pixel", &Raytracer::m_num_hits_per_pixel)
            .def_readonly(
                "num_traversed_per_pixel",
                &Raytracer::m_num_traversed_per_pixel)
            .def_readonly("t_maxes", &Raytracer::m_t_maxes);
    }
};

TORCH_LIBRARY(gausstracer, m) { Raytracer::bind(m); }
