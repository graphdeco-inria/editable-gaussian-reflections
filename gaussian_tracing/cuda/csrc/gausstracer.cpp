#include <torch/extension.h>

#include <fstream>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <tuple>

#include "cublas_v2.h"
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "optix/bvh_wrapper.h"
#include "params.h"
#include "utils/exception.h"

#if DENOISER == true
#include <optix_denoiser_tiling.h>
#endif

using namespace at;

static void
context_log_cb(uint32_t level, const char *tag, const char *message, void *) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
              << "]: " << message << "\n";
}

static std::string loadPtxFile() {
    // create a string from CMAKE_BINARY_DIR
    std::string path = std::string(CMAKE_BINARY_DIR) +
                       "/CMakeFiles/optix_program.dir/csrc/shaders.ptx";
    std::ifstream file(path.c_str(), std::ios::binary);
    if (file.good()) {
        std::vector<unsigned char> buffer = std::vector<unsigned char>(
            std::istreambuf_iterator<char>(file), {});
        std::string str;
        str.assign(buffer.begin(), buffer.end());
        return str;
    } else {
        std::string error = "couldn't locate ptx file in path " + path;
        throw std::runtime_error(error);
    }
}

float GetEnvironmentVariableOrDefault(
    const std::string &variable_name, float default_value) {
    const char *value = getenv(variable_name.c_str());
    if (value == 0) {
        return default_value;
    }
    return std::stof(value);
}

// Create four channel float OptixImage2D with given dimension. Allocate memory
// on device and Copy data from host memory given in hmem to device if hmem is
// nonzero.
static OptixImage2D createOptixImage2D(
    unsigned int width, unsigned int height, CUdeviceptr tensor_data = 0) {
    OptixImage2D oi;

    if (tensor_data != 0) {
        oi.data = tensor_data;
    } else {
        const uint64_t frame_byte_size = width * height * sizeof(float3);
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&oi.data), frame_byte_size));
    }
    oi.width = width;
    oi.height = height;
    oi.rowStrideInBytes = width * sizeof(float3);
    oi.pixelStrideInBytes = sizeof(float3);
    oi.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    return oi;
}

struct Raytracer : torch::CustomClassHolder {
    float *m_output_transmittances;

    OptixShaderBindingTable m_sbt = {};
    OptixPipeline m_pipeline;
    OptixDeviceContext m_context;

    OptixBuildInput m_aabb_input = {};

    CUstream m_stream;
    CUdeviceptr m_d_aabb_buffer;
    CUdeviceptr m_unit_bbox;

    uint32_t m_aabb_input_flags[2] = {OPTIX_GEOMETRY_FLAG_NONE};
    unsigned int m_build_flags =
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
        OPTIX_BUILD_FLAG_ALLOW_UPDATE; // OPTIX_BUILD_FLAG_ALLOW_COMPACTION

    int m_width;
    int m_height;

    Params m_h_params;
    CUdeviceptr m_d_params;

    cublasHandle_t handle;

    Tensor m_iteration;

    Tensor m_grads_enabled =
        torch::ones({1}, torch::dtype(torch::kBool).device(torch::kCUDA));
    Tensor m_total_hits =
        torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));

#if LOG_ALL_HITS == true
    // PPLL storage for forward pass
    Tensor m_all_gaussian_ids = torch::zeros(
        {PPLL_STORAGE_SIZE}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    Tensor m_all_distances = torch::zeros(
        {PPLL_STORAGE_SIZE},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    // Tensor m_all_half_chord_lengths = torch::zeros({PPLL_STORAGE_SIZE},
    // torch::dtype(torch::kFloat32).device(torch::kCUDA));
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
#if NUM_SLABS > 1
    Tensor m_all_slab_idx = torch::zeros(
        {PPLL_STORAGE_SIZE}, torch::dtype(torch::kInt32).device(torch::kCUDA));
#endif
    Tensor m_prev_hit_per_pixel;

#if BACKWARDS_PASS == true
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
#endif
#endif

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
    Tensor m_gaussian_mask;
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
    Tensor m_global_scale_factor =
        torch::ones({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    Tensor m_gaussian_total_weight;
    Tensor m_densification_gradient_diffuse;
    Tensor m_densification_gradient_glossy;

#if USE_GT_BRDF == false
    Tensor m_lut;
#endif
    Tensor m_cheap_approx =
        torch::zeros({1}, torch::dtype(torch::kBool).device(torch::kCUDA));

    // Config
    float m_exp_power = EXP_POWER;
    float m_alpha_threshold = ALPHA_THRESHOLD;
    float m_transmittance_threshold = T_THRESHOLD;

#if DENOISER == true
    OptixDenoiser m_denoiser = nullptr;
    uint32_t m_scratch_size = 0;
    CUdeviceptr m_intensity = 0;
    CUdeviceptr m_avgColor = 0;
    CUdeviceptr m_scratch = 0;
    CUdeviceptr m_state = 0;
    uint32_t m_state_size = 0;
    unsigned int m_overlap = 0;
    OptixDenoiserParams m_params_denoiser = {};
    OptixDenoiserGuideLayer m_guideLayer = {};
    std::vector<OptixDenoiserLayer> m_layers;
#endif

    Tensor m_num_hits_per_pixel;
    Tensor m_num_traversed_per_pixel;

    Raytracer(
        int64_t image_width, int64_t image_height, int64_t num_gaussians) {
        if (num_gaussians <= 0) {
            num_gaussians =
                1; // default to 1 gaussian to avoid degenerate tensor shapes
        }

        m_width = image_width;
        m_height = image_height;

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
#if OPTIMIZE_EXP_POWER == true
        m_gaussian_exp_power = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dexp_powers = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if true
        // USE_LEVEL_OF_DETAIL == true
        // easier to always allocate these regardless of config even if its a
        // bit of a waste
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
#endif
        m_gaussian_mask = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kInt32).device(torch::kCUDA));
//
#if ATTACH_POSITION == true
        m_gaussian_position = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dgaussian_position = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if ATTACH_NORMALS == true
        m_gaussian_normal = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dgaussian_normal = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if ATTACH_F0 == true
        m_gaussian_f0 = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dgaussian_f0 = torch::zeros(
            {num_gaussians, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if ATTACH_ROUGHNESS == true
        m_gaussian_roughness = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_dL_dgaussian_roughness = torch::zeros(
            {num_gaussians, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif

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
#if MAX_BOUNCES > 0
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
#endif
#if SAVE_LUT_IMAGES == true
        m_output_effective_reflection_position = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_effective_reflection_normal = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_effective_F0 = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_effective_roughness = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_effective_normal = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#if USE_GT_BRDF == false
        m_output_lut_values = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 2},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_n_dot_v = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#endif
#if ATTACH_POSITION == true || POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
#if SAVE_ALL_MAPS == true
        m_output_position = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
        m_target_position = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
#if SAVE_ALL_MAPS == true
        m_output_depth = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
        m_target_depth = torch::zeros(
            {image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if ATTACH_NORMALS == true
#if SAVE_ALL_MAPS == true
        m_output_normal = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
        m_target_normal = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if ATTACH_F0 == true
#if SAVE_ALL_MAPS == true
        m_output_f0 = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
        m_target_f0 = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if ATTACH_ROUGHNESS == true
#if SAVE_ALL_MAPS == true
        m_output_roughness = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
        m_target_roughness = torch::zeros(
            {image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if RENDER_DISTORTION == true
        m_output_distortion = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if SAVE_LOD_IMAGES == true
        m_output_lod_mean = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_lod_scale = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        m_output_ray_lod = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if MAX_BOUNCES > 0 && SAVE_ALL_MAPS == true
        m_output_brdf = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if USE_GT_BRDF == true
        m_target_brdf = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if USE_GT_DIFFUSE_IRRADIANCE == true
#if SAVE_ALL_MAPS == true
        m_output_diffuse_irradiance = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
        m_target_diffuse_irradiance = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if REFLECTIONS_FROM_GT_GLOSSY_IRRADIANCE == true
#if SAVE_ALL_MAPS == true
        m_output_glossy_irradiance = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
        m_target_glossy_irradiance = torch::zeros(
            {image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if MEASURE_LOSS == true
        m_loss_tensor = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 1},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif
#if SAVE_ALL_MAPS == true
        m_output_incident_radiance = torch::zeros(
            {MAX_BOUNCES + 1, image_height, image_width, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
#endif

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

#if LOG_ALL_HITS == true
        m_prev_hit_per_pixel = torch::zeros(
            {m_height, m_width},
            torch::dtype(torch::kInt32).device(torch::kCUDA));
        m_prev_hit_per_pixel.fill_(999999999);
#if BACKWARDS_PASS == true
        m_prev_hit_per_pixel_for_backprop = torch::zeros(
            {m_height, m_width},
            torch::dtype(torch::kInt32).device(torch::kCUDA));
        m_prev_hit_per_pixel_for_backprop.fill_(999999999);
#endif
#endif

#if USE_GT_BRDF == false && USE_LUT == true
        std::ifstream fin(
            std::string(CMAKE_CURRENT_LIST_DIR) + "/assets/LUT_512x512x2.pt",
            std::ios::in | std::ios::binary);
        std::string pickled_str(
            (std::istreambuf_iterator<char>(fin)),
            std::istreambuf_iterator<char>());
        fin.close();
        std::vector<char> pickled_vec(pickled_str.begin(), pickled_str.end());
        const auto tensor_loaded = torch::pickle_load(pickled_vec);
        m_lut = tensor_loaded.toTensor().to(torch::kFloat32).to(torch::kCUDA);
#endif

        {
            std::cout << "initializing CUDA and creating OptiX context"
                      << std::endl;

            CUDA_CHECK(cudaFree(0));

            CUcontext cuCtx = 0; // zero means take the current context
            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));
        }

        // Create module
        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.optLevel =
                OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
            // module_compile_options.maxRegisterCount = 255; //
            pipeline_compile_options.usesMotionBlur = false;
            pipeline_compile_options.traversableGraphFlags =
                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS |
                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
#if STORAGE_MODE == PER_PIXEL_LINKED_LIST
            pipeline_compile_options.numPayloadValues =
                1 + (TILE_SIZE * TILE_SIZE) + 2 + 3;
#else
            pipeline_compile_options.numPayloadValues = 32;
#endif
            pipeline_compile_options.numAttributeValues = 0;
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
            pipeline_compile_options.pipelineLaunchParamsVariableName =
                "params";
#if USE_POLYCAGE == true
            pipeline_compile_options.usesPrimitiveTypeFlags =
                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
#else
            pipeline_compile_options.usesPrimitiveTypeFlags =
                OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
#endif

            auto ptxData = loadPtxFile();
            OPTIX_CHECK_LOG(optixModuleCreate(
                m_context,
                &module_compile_options,
                &pipeline_compile_options,
                ptxData.c_str(),
                ptxData.size(),
                LOG,
                &LOG_SIZE,
                &module));
        }

        build_bvh();

        { // Create params object
            m_h_params.image_width = m_width;
            m_h_params.image_height = m_height;

            // Render settings
            m_h_params.denoise = reinterpret_cast<bool *>(m_denoise.data_ptr());
            m_h_params.num_samples =
                reinterpret_cast<int *>(m_num_samples.data_ptr());
            m_h_params.num_bounces =
                reinterpret_cast<int *>(m_num_bounces.data_ptr());

            // Configuration
            m_h_params.exp_power = m_exp_power;
            m_h_params.alpha_threshold = m_alpha_threshold;
            m_h_params.transmittance_threshold = m_transmittance_threshold;
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

#if LOG_ALL_HITS == true
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
#if NUM_SLABS > 1
            m_h_params.all_slab_idx =
                reinterpret_cast<uint32_t *>(m_all_slab_idx.data_ptr());
#endif
            m_h_params.total_hits =
                reinterpret_cast<uint32_t *>(m_total_hits.data_ptr());

#if BACKWARDS_PASS == true
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
#endif
#endif

            m_gaussian_rgb.mutable_grad() = m_dL_drgb;
            m_gaussian_opacity.mutable_grad() = m_dL_dopacity;
            m_gaussian_scales.mutable_grad() = m_dL_dscales;
            m_gaussian_rotations.mutable_grad() = m_dL_drotations;
            m_gaussian_means.mutable_grad() = m_dL_dmeans;
#if OPTIMIZE_EXP_POWER == true
            m_gaussian_exp_power.mutable_grad() = m_dL_dexp_powers;
#endif
#if true
            // USE_LEVEL_OF_DETAIL == true
            m_gaussian_lod_mean.mutable_grad() = m_dL_dgaussian_lod_mean;
            m_gaussian_lod_scale.mutable_grad() = m_dL_dgaussian_lod_scale;
#endif
#if ATTACH_POSITION == true
            m_gaussian_position.mutable_grad() = m_dL_dgaussian_position;
#endif
#if ATTACH_NORMALS == true
            m_gaussian_normal.mutable_grad() = m_dL_dgaussian_normal;
#endif
#if ATTACH_F0 == true
            m_gaussian_f0.mutable_grad() = m_dL_dgaussian_f0;
#endif
#if ATTACH_ROUGHNESS == true
            m_gaussian_roughness.mutable_grad() = m_dL_dgaussian_roughness;
#endif

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
#if OPTIMIZE_EXP_POWER == true
            m_h_params.gaussian_exp_power =
                reinterpret_cast<float *>(gaussian_exp_power.data_ptr());
            m_h_params.dL_dexp_powers =
                reinterpret_cast<float *>(dL_dexp_powers.data_ptr());
#endif
#if true
            // USE_LEVEL_OF_DETAIL == true
            m_h_params.gaussian_lod_mean =
                reinterpret_cast<float *>(m_gaussian_lod_mean.data_ptr());
            m_h_params.dL_dgaussian_lod_mean =
                reinterpret_cast<float *>(m_dL_dgaussian_lod_mean.data_ptr());
            m_h_params.gaussian_lod_scale =
                reinterpret_cast<float *>(m_gaussian_lod_scale.data_ptr());
            m_h_params.dL_dgaussian_lod_scale =
                reinterpret_cast<float *>(m_dL_dgaussian_lod_scale.data_ptr());
#endif
#if ATTACH_POSITION == true
            m_h_params.gaussian_position =
                reinterpret_cast<float3 *>(m_gaussian_position.data_ptr());
            m_h_params.dL_dgaussian_position =
                reinterpret_cast<float3 *>(m_dL_dgaussian_position.data_ptr());
#endif
#if ATTACH_NORMALS == true
            m_h_params.gaussian_normal =
                reinterpret_cast<float3 *>(m_gaussian_normal.data_ptr());
            m_h_params.dL_dgaussian_normal =
                reinterpret_cast<float3 *>(m_dL_dgaussian_normal.data_ptr());
#endif
#if ATTACH_F0 == true
            m_h_params.gaussian_f0 =
                reinterpret_cast<float3 *>(m_gaussian_f0.data_ptr());
            m_h_params.dL_dgaussian_f0 =
                reinterpret_cast<float3 *>(m_dL_dgaussian_f0.data_ptr());
#endif
#if ATTACH_ROUGHNESS == true
            m_h_params.gaussian_roughness =
                reinterpret_cast<float *>(m_gaussian_roughness.data_ptr());
            m_h_params.dL_dgaussian_roughness =
                reinterpret_cast<float *>(m_dL_dgaussian_roughness.data_ptr());
#endif

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
#if MAX_BOUNCES > 0
            m_h_params.target_diffuse =
                reinterpret_cast<float3 *>(m_target_diffuse.data_ptr());
            m_h_params.target_glossy =
                reinterpret_cast<float3 *>(m_target_glossy.data_ptr());
#endif
#if SAVE_RAY_IMAGES == true
            m_h_params.output_ray_origin =
                reinterpret_cast<float3 *>(m_output_ray_origin.data_ptr());
            m_h_params.output_ray_direction =
                reinterpret_cast<float3 *>(m_output_ray_direction.data_ptr());
#endif
#if SAVE_LUT_IMAGES == true
            m_h_params.output_effective_reflection_position =
                reinterpret_cast<float3 *>(
                    m_output_effective_reflection_position.data_ptr());
            m_h_params.output_effective_reflection_normal =
                reinterpret_cast<float3 *>(
                    m_output_effective_reflection_normal.data_ptr());
            m_h_params.output_effective_F0 =
                reinterpret_cast<float3 *>(m_output_effective_F0.data_ptr());
            m_h_params.output_effective_roughness = reinterpret_cast<float *>(
                m_output_effective_roughness.data_ptr());
            m_h_params.output_effective_normal = reinterpret_cast<float3 *>(
                m_output_effective_normal.data_ptr());
#if USE_GT_BRDF == false
            m_h_params.output_lut_values =
                reinterpret_cast<float2 *>(m_output_lut_values.data_ptr());
            m_h_params.output_n_dot_v =
                reinterpret_cast<float *>(m_output_n_dot_v.data_ptr());
#endif
#endif
#if SAVE_ALL_MAPS == true
            m_h_params.output_incident_radiance = reinterpret_cast<float3 *>(
                m_output_incident_radiance.data_ptr());
#endif
#if ATTACH_POSITION == true || POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
#if SAVE_ALL_MAPS == true
            m_h_params.output_position =
                reinterpret_cast<float3 *>(m_output_position.data_ptr());
#endif
            m_h_params.target_position =
                reinterpret_cast<float3 *>(m_target_position.data_ptr());
#endif
#if ATTACH_DEPTH == true || POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
#if SAVE_ALL_MAPS == true
            m_h_params.output_depth =
                reinterpret_cast<float *>(m_output_depth.data_ptr());
#endif
            m_h_params.target_depth =
                reinterpret_cast<float *>(m_target_depth.data_ptr());
#endif
#if ATTACH_NORMALS == true
#if SAVE_ALL_MAPS == true
            m_h_params.output_normal =
                reinterpret_cast<float3 *>(m_output_normal.data_ptr());
#endif
            m_h_params.target_normal =
                reinterpret_cast<float3 *>(m_target_normal.data_ptr());
#endif
#if ATTACH_F0 == true
#if SAVE_ALL_MAPS == true
            m_h_params.output_f0 =
                reinterpret_cast<float3 *>(m_output_f0.data_ptr());
#endif
            m_h_params.target_f0 =
                reinterpret_cast<float3 *>(m_target_f0.data_ptr());
#endif
#if ATTACH_ROUGHNESS == true
#if SAVE_ALL_MAPS == true
            m_h_params.output_roughness =
                reinterpret_cast<float *>(m_output_roughness.data_ptr());
#endif
            m_h_params.target_roughness =
                reinterpret_cast<float *>(m_target_roughness.data_ptr());
#endif
#if RENDER_DISTORTION == true
            m_h_params.output_distortion =
                reinterpret_cast<float *>(m_output_distortion.data_ptr());
#endif
#if MAX_BOUNCES > 0
#if SAVE_ALL_MAPS == true
            m_h_params.output_brdf =
                reinterpret_cast<float3 *>(m_output_brdf.data_ptr());
#endif
#if USE_GT_BRDF == true
            m_h_params.target_brdf =
                reinterpret_cast<float3 *>(m_target_brdf.data_ptr());
#endif
#endif
#if USE_GT_DIFFUSE_IRRADIANCE == true
#if SAVE_ALL_MAPS == true
            m_h_params.output_diffuse_irradiance = reinterpret_cast<float3 *>(
                m_output_diffuse_irradiance.data_ptr());
#endif
            m_h_params.target_diffuse_irradiance = reinterpret_cast<float3 *>(
                m_target_diffuse_irradiance.data_ptr());
#endif
#if REFLECTIONS_FROM_GT_GLOSSY_IRRADIANCE == true
#if SAVE_ALL_MAPS == true
            m_h_params.output_glossy_irradiance = reinterpret_cast<float3 *>(
                m_output_glossy_irradiance.data_ptr());
#endif
            m_h_params.target_glossy_irradiance = reinterpret_cast<float3 *>(
                m_target_glossy_irradiance.data_ptr());
#endif
#if MEASURE_LOSS == true
            m_h_params.loss_tensor =
                reinterpret_cast<float *>(m_loss_tensor.data_ptr());
#endif
#if SAVE_LOD_IMAGES == true
            m_h_params.output_lod_mean =
                reinterpret_cast<float *>(m_output_lod_mean.data_ptr());
            m_h_params.output_lod_scale =
                reinterpret_cast<float *>(m_output_lod_scale.data_ptr());
            m_h_params.output_ray_lod =
                reinterpret_cast<float *>(m_output_ray_lod.data_ptr());
#endif

            // Other buffers
            m_h_params.t_maxes =
                reinterpret_cast<float *>(m_t_maxes.data_ptr());
            m_h_params.t_mins = reinterpret_cast<float *>(m_t_mins.data_ptr());
            m_h_params.random_seeds =
                reinterpret_cast<uint32_t *>(m_random_seeds.data_ptr());
            m_h_params.iteration =
                reinterpret_cast<int *>(m_iteration.data_ptr());
            m_h_params.global_scale_factor =
                reinterpret_cast<float *>(m_global_scale_factor.data_ptr());
            m_h_params.grads_enabled =
                reinterpret_cast<bool *>(m_grads_enabled.data_ptr());
            m_h_params.cheap_approx =
                reinterpret_cast<bool *>(m_cheap_approx.data_ptr());
            m_h_params.num_hits_per_pixel =
                reinterpret_cast<int *>(m_num_hits_per_pixel.data_ptr());
            m_h_params.num_traversed_per_pixel =
                reinterpret_cast<int *>(m_num_traversed_per_pixel.data_ptr());
// m_h_params.mask = reinterpret_cast<int*>(mask.data_ptr());

// BRDF LUT
#if USE_GT_BRDF == false && USE_LUT == true
            m_h_params.lut = reinterpret_cast<float2 *>(m_lut.data_ptr());
#endif

            // Set TLAS and copy
            m_h_params.handle = m_tlas_handle;
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&m_d_params), sizeof(Params)));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(m_d_params),
                &m_h_params,
                sizeof(Params),
                cudaMemcpyHostToDevice));
        }

        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            std::cout << "Creating program groups" << std::endl;

            OptixProgramGroupOptions program_group_options =
                {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc = {};
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                m_context,
                &raygen_prog_group_desc,
                1, // num program groups
                &program_group_options,
                LOG,
                &LOG_SIZE,
                &raygen_prog_group));

            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module = nullptr;
            miss_prog_group_desc.miss.entryFunctionName = nullptr;
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                m_context,
                &miss_prog_group_desc,
                1, // num program groups
                &program_group_options,
                LOG,
                &LOG_SIZE,
                &miss_prog_group));

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH = nullptr;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;

#if USE_POLYCAGE == true
            hitgroup_prog_group_desc.hitgroup.moduleAH = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH =
                "__anyhit__ah";
            hitgroup_prog_group_desc.hitgroup.moduleIS = nullptr;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
#else
            hitgroup_prog_group_desc.hitgroup.moduleAH = nullptr;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
            hitgroup_prog_group_desc.hitgroup.moduleIS = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS =
                "__intersection__gaussian";
#endif
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                m_context,
                &hitgroup_prog_group_desc,
                1, // num program groups
                &program_group_options,
                LOG,
                &LOG_SIZE,
                &hitgroup_prog_group));
        }

        {
            const uint32_t max_trace_depth = 1;
            OptixProgramGroup program_groups[] = {
                raygen_prog_group, miss_prog_group, hitgroup_prog_group};

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth;
            OPTIX_CHECK_LOG(optixPipelineCreate(
                m_context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof(program_groups) / sizeof(program_groups[0]),
                LOG,
                &LOG_SIZE,
                &m_pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto &prog_group : program_groups) {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(
                    prog_group, &stack_sizes, m_pipeline));
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(
                &stack_sizes,
                max_trace_depth,
                0, // maxCCDepth
                0, // maxDCDEpth
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state,
                &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(
                m_pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                1 // maxTraversableDepth
                ));
        }

        {
            CUdeviceptr raygen_record;
            const size_t raygen_record_size = sizeof(SbtRecord<RayGenData>);
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&raygen_record), raygen_record_size));
            SbtRecord<RayGenData> rg_sbt;
            rg_sbt.data = {2.1f, 2.1f, 2.1f};
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(raygen_record),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice));
            m_sbt.raygenRecord = raygen_record;

            CUdeviceptr miss_record;
            size_t miss_record_size = sizeof(SbtRecord<MissData>);
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&miss_record), miss_record_size));
            SbtRecord<MissData> ms_sbt;
            ms_sbt.data = {0.3f, 0.1f, 0.2f};
            OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(miss_record),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice));
            m_sbt.missRecordBase = miss_record;
            m_sbt.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
            m_sbt.missRecordCount = 1;

            CUdeviceptr hitgroup_record;
            auto num_records = 1;
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&hitgroup_record),
                sizeof(SbtRecord<HitData>) * num_records));
            for (int i = 0; i < num_records; i++) {
                SbtRecord<HitData> hg_sb;
                hg_sb.data = {i * 0.001f, i * 0.002f, i * 0.003f};
                OPTIX_CHECK(
                    optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sb));
                CUDA_CHECK(cudaMemcpy(
                    reinterpret_cast<void *>(
                        reinterpret_cast<uint8_t *>(hitgroup_record) +
                        sizeof(SbtRecord<HitData>) * i),
                    &hg_sb,
                    sizeof(SbtRecord<HitData>),
                    cudaMemcpyHostToDevice));
            }
            m_sbt.hitgroupRecordBase = hitgroup_record;
            m_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitData>);
            m_sbt.hitgroupRecordCount = num_records;
        }

        CUDA_CHECK(cudaStreamCreate(&m_stream));

        init_denoiser();

        cudaDeviceSynchronize();
    }

    void init_denoiser() {
#if DENOISER == true
        {
            OptixDenoiserOptions options = {};
            options.guideNormal = 1; // data.normal ? 1 : 0;
            options.denoiseAlpha = (OptixDenoiserAlphaMode)0; // alphaMode;

            OptixDenoiserModelKind modelKind;
            // if( upscale2xMode )
            //     modelKind = temporalMode ?
            //     OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X :
            //     OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
            // else if( kpMode || data.aovs.size() > 0 )
            //     modelKind = temporalMode ?
            //     OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV :
            //     OPTIX_DENOISER_MODEL_KIND_AOV;
            // else
            //     modelKind = temporalMode ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL
            //     : OPTIX_DENOISER_MODEL_KIND_HDR;
            modelKind = OPTIX_DENOISER_MODEL_KIND_HDR;
            // modelKind = OPTIX_DENOISER_MODEL_KIND_TEMPORAL;  // implies HDR
            OPTIX_CHECK(optixDenoiserCreate(
                m_context, modelKind, &options, &m_denoiser));
        }

        //
        // Allocate device memory for denoiser
        //
        {
            OptixDenoiserSizes denoiser_sizes;

            OPTIX_CHECK(optixDenoiserComputeMemoryResources(
                m_denoiser,
                m_width,
                m_height,
                // m_tileWidth,
                // m_tileHeight,
                &denoiser_sizes));

            // if( tileWidth == 0 )
            // {
            m_scratch_size = static_cast<uint32_t>(
                denoiser_sizes.withoutOverlapScratchSizeInBytes);
            m_overlap = 0;
            // }
            // else
            // {
            //     m_scratch_size = static_cast<uint32_t>(
            //     denoiser_sizes.withOverlapScratchSizeInBytes ); m_overlap =
            //     denoiser_sizes.overlapWindowSizeInPixels;
            // }

            // if( data.aovs.size() == 0 && kpMode == false )
            // {
            //     CUDA_CHECK( cudaMalloc(
            //                 reinterpret_cast<void**>( &m_intensity ),
            //                 sizeof( float )
            //                 ) );
            // }
            // else
            // {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&m_avgColor), 3 * sizeof(float)));
            // }

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&m_scratch), m_scratch_size));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&m_state),
                denoiser_sizes.stateSizeInBytes));

            m_state_size =
                static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

            OptixDenoiserLayer layer = {};
            layer.input = createOptixImage2D(
                m_width,
                m_height,
                reinterpret_cast<CUdeviceptr>(
                    m_output_rgb.index({MAX_BOUNCES + 1}).data_ptr()));
            layer.output = createOptixImage2D(
                m_width,
                m_height,
                reinterpret_cast<CUdeviceptr>(
                    m_output_rgb.index({MAX_BOUNCES + 1}).data_ptr()));

            m_layers.push_back(layer);
        }

        m_guideLayer.normal = createOptixImage2D(
            m_width,
            m_height,
            reinterpret_cast<CUdeviceptr>(
                m_output_normal.index({0}).data_ptr()));

        {
            OPTIX_CHECK(optixDenoiserSetup(
                m_denoiser,
                nullptr,  // CUDA stream
                m_width,  // m_tileWidth + 2 * m_overlap,
                m_height, // m_tileHeight + 2 * m_overlap,
                m_state,
                m_state_size,
                m_scratch,
                m_scratch_size));

            m_params_denoiser.hdrIntensity = m_intensity;
            m_params_denoiser.hdrAverageColor = m_avgColor;
            m_params_denoiser.blendFactor = 0.0f;
            m_params_denoiser.temporalModeUsePreviousLayers = 0;
        }
#endif
    }

    ~Raytracer() {
        CUDA_CHECK(cudaStreamDestroy(m_stream));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_sbt.raygenRecord)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_sbt.missRecordBase)));
        CUDA_CHECK(
            cudaFree(reinterpret_cast<void *>(m_sbt.hitgroupRecordBase)));
        OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
        OPTIX_CHECK(optixDeviceContextDestroy(m_context));
    }

    OptixBuildInput m_blas_input = {};
    OptixBuildInput m_tlas_input = {};
    OptixTraversableHandle m_blas_handle;
    OptixTraversableHandle m_tlas_handle;

    Tensor unit_bbox_tensor = torch::tensor(
        {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, torch::device(torch::kCUDA));

#if USE_POLYCAGE == 1
    Tensor mesh_cage_verts = torch::tensor(
                                 {{0., 0., -1.},
                                  {0.72359997, -0.52572, -0.44721499},
                                  {-0.27638501, -0.85064, -0.44721499},
                                  {-0.89442497, 0., -0.44721499},
                                  {-0.27638501, 0.85064, -0.44721499},
                                  {0.72359997, 0.52572, -0.44721499},
                                  {0.27638501, -0.85064, 0.44721499},
                                  {-0.72359997, -0.52572, 0.44721499},
                                  {-0.72359997, 0.52572, 0.44721499},
                                  {0.27638501, 0.85064, 0.44721499},
                                  {0.89442497, 0., 0.44721499},
                                  {0., 0., 1.}},
                                 torch::device(torch::kCUDA)) *
                             1.24;
    Tensor mesh_cage_verts_half =
        torch::tensor(
            {{0., 0., -1.},
             {0.72359997, -0.52572, -0.44721499},
             {-0.27638501, -0.85064, -0.44721499},
             {-0.89442497, 0., -0.44721499},
             {-0.27638501, 0.85064, -0.44721499},
             {0.72359997, 0.52572, -0.44721499},
             {0.27638501, -0.85064, 0.44721499},
             {-0.72359997, -0.52572, 0.44721499},
             {-0.72359997, 0.52572, 0.44721499},
             {0.27638501, 0.85064, 0.44721499},
             {0.89442497, 0., 0.44721499},
             {0., 0., 1.}},
            torch::device(torch::kCUDA).dtype(torch::kfloat6)) *
        1.24;
    Tensor mesh_cage_faces = torch::tensor(
        {{0, 1, 2},   {1, 0, 5},  {0, 2, 3},  {0, 3, 4},  {0, 4, 5},
         {1, 5, 10},  {2, 1, 6},  {3, 2, 7},  {4, 3, 8},  {5, 4, 9},
         {1, 10, 6},  {2, 6, 7},  {3, 7, 8},  {4, 8, 9},  {5, 9, 10},
         {6, 10, 11}, {7, 6, 11}, {8, 7, 11}, {9, 8, 11}, {10, 9, 11}},
        torch::dtype(torch::kInt32).device(torch::kCUDA));

    Tensor mesh_cage_verts_tri_soup =
        torch::tensor(
            {{-0.8944, 0.0000, -0.4472},  {-0.7236, -0.5257, 0.4472},
             {-0.7236, 0.5257, 0.4472},   {0.2764, -0.8506, 0.4472},
             {-0.7236, -0.5257, 0.4472},  {0.0000, 0.0000, 1.0000},
             {-0.7236, -0.5257, 0.4472},  {-0.7236, 0.5257, 0.4472},
             {0.0000, 0.0000, 1.0000},    {-0.2764, -0.8506, -0.4472},
             {0.2764, -0.8506, 0.4472},   {-0.7236, -0.5257, 0.4472},
             {-0.2764, -0.8506, -0.4472}, {-0.8944, 0.0000, -0.4472},
             {-0.7236, -0.5257, 0.4472},  {-0.8944, 0.0000, -0.4472},
             {-0.2764, 0.8506, -0.4472},  {-0.7236, 0.5257, 0.4472},
             {0.0000, 0.0000, -1.0000},   {-0.2764, -0.8506, -0.4472},
             {-0.8944, 0.0000, -0.4472},  {0.0000, 0.0000, -1.0000},
             {-0.8944, 0.0000, -0.4472},  {-0.2764, 0.8506, -0.4472},
             {0.0000, 0.0000, -1.0000},   {0.7236, -0.5257, -0.4472},
             {-0.2764, -0.8506, -0.4472}, {0.7236, -0.5257, -0.4472},
             {-0.2764, -0.8506, -0.4472}, {0.2764, -0.8506, 0.4472},
             {0.7236, -0.5257, -0.4472},  {0.2764, -0.8506, 0.4472},
             {0.8944, 0.0000, 0.4472},    {0.7236, -0.5257, -0.4472},
             {0.7236, 0.5257, -0.4472},   {0.8944, 0.0000, 0.4472},
             {0.0000, 0.0000, -1.0000},   {0.7236, -0.5257, -0.4472},
             {0.7236, 0.5257, -0.4472},   {0.0000, 0.0000, -1.0000},
             {-0.2764, 0.8506, -0.4472},  {0.7236, 0.5257, -0.4472},
             {-0.2764, 0.8506, -0.4472},  {0.7236, 0.5257, -0.4472},
             {0.2764, 0.8506, 0.4472},    {-0.2764, 0.8506, -0.4472},
             {-0.7236, 0.5257, 0.4472},   {0.2764, 0.8506, 0.4472},
             {-0.7236, 0.5257, 0.4472},   {0.2764, 0.8506, 0.4472},
             {0.0000, 0.0000, 1.0000},    {0.2764, 0.8506, 0.4472},
             {0.8944, 0.0000, 0.4472},    {0.0000, 0.0000, 1.0000},
             {0.7236, 0.5257, -0.4472},   {0.2764, 0.8506, 0.4472},
             {0.8944, 0.0000, 0.4472},    {0.2764, -0.8506, 0.4472},
             {0.8944, 0.0000, 0.4472},    {0.0000, 0.0000, 1.0000}},
            torch::device(torch::kCUDA)) *
        1.24;
#elif USE_POLYCAGE == 2
    Tensor mesh_cage_verts =
        torch::tensor(
            {{0.0, 0.0, -1.0},
             {0.7236073017120361, -0.5257253050804138, -0.44721952080726624},
             {-0.276388019323349, -0.8506492376327515, -0.4472198486328125},
             {-0.8944262266159058, 0.0, -0.44721561670303345},
             {-0.276388019323349, 0.8506492376327515, -0.4472198486328125},
             {0.7236073017120361, 0.5257253050804138, -0.44721952080726624},
             {0.276388019323349, -0.8506492376327515, 0.4472198486328125},
             {-0.7236073017120361, -0.5257253050804138, 0.44721952080726624},
             {-0.7236073017120361, 0.5257253050804138, 0.44721952080726624},
             {0.276388019323349, 0.8506492376327515, 0.4472198486328125},
             {0.8944262266159058, 0.0, 0.44721561670303345},
             {0.0, 0.0, 1.0},
             {-0.16245555877685547, -0.49999526143074036, -0.8506544232368469},
             {0.42532268166542053, -0.30901139974594116, -0.8506541848182678},
             {0.26286882162094116, -0.8090116381645203, -0.5257376432418823},
             {0.8506478667259216, 0.0, -0.5257359147071838},
             {0.42532268166542053, 0.30901139974594116, -0.8506541848182678},
             {-0.525729775428772, 0.0, -0.8506516814231873},
             {-0.6881893873214722, -0.49999693036079407, -0.5257362127304077},
             {-0.16245555877685547, 0.49999526143074036, -0.8506544232368469},
             {-0.6881893873214722, 0.49999693036079407, -0.5257362127304077},
             {0.26286882162094116, 0.8090116381645203, -0.5257376432418823},
             {0.9510578513145447, -0.30901262164115906, 0.0},
             {0.9510578513145447, 0.30901262164115906, 0.0},
             {0.0, -0.9999999403953552, 0.0},
             {0.5877856016159058, -0.8090167045593262, 0.0},
             {-0.9510578513145447, -0.30901262164115906, 0.0},
             {-0.5877856016159058, -0.8090167045593262, 0.0},
             {-0.5877856016159058, 0.8090167045593262, 0.0},
             {-0.9510578513145447, 0.30901262164115906, 0.0},
             {0.5877856016159058, 0.8090167045593262, 0.0},
             {0.0, 0.9999999403953552, 0.0},
             {0.6881893873214722, -0.49999693036079407, 0.5257362127304077},
             {-0.26286882162094116, -0.8090116381645203, 0.5257376432418823},
             {-0.8506478667259216, 0.0, 0.5257359147071838},
             {-0.26286882162094116, 0.8090116381645203, 0.5257376432418823},
             {0.6881893873214722, 0.49999693036079407, 0.5257362127304077},
             {0.16245555877685547, -0.49999526143074036, 0.8506543636322021},
             {0.525729775428772, 0.0, 0.8506516814231873},
             {-0.42532268166542053, -0.30901139974594116, 0.8506541848182678},
             {-0.42532268166542053, 0.30901139974594116, 0.8506541848182678},
             {0.16245555877685547, 0.49999526143074036, 0.8506543636322021}},
            torch::device(torch::kCUDA)) *
        1.09;
    Tensor mesh_cage_faces = torch::tensor(
        {{0, 13, 12},  {1, 13, 15},  {0, 12, 17},  {0, 17, 19},  {0, 19, 16},
         {1, 15, 22},  {2, 14, 24},  {3, 18, 26},  {4, 20, 28},  {5, 21, 30},
         {1, 22, 25},  {2, 24, 27},  {3, 26, 29},  {4, 28, 31},  {5, 30, 23},
         {6, 32, 37},  {7, 33, 39},  {8, 34, 40},  {9, 35, 41},  {10, 36, 38},
         {38, 41, 11}, {38, 36, 41}, {36, 9, 41},  {41, 40, 11}, {41, 35, 40},
         {35, 8, 40},  {40, 39, 11}, {40, 34, 39}, {34, 7, 39},  {39, 37, 11},
         {39, 33, 37}, {33, 6, 37},  {37, 38, 11}, {37, 32, 38}, {32, 10, 38},
         {23, 36, 10}, {23, 30, 36}, {30, 9, 36},  {31, 35, 9},  {31, 28, 35},
         {28, 8, 35},  {29, 34, 8},  {29, 26, 34}, {26, 7, 34},  {27, 33, 7},
         {27, 24, 33}, {24, 6, 33},  {25, 32, 6},  {25, 22, 32}, {22, 10, 32},
         {30, 31, 9},  {30, 21, 31}, {21, 4, 31},  {28, 29, 8},  {28, 20, 29},
         {20, 3, 29},  {26, 27, 7},  {26, 18, 27}, {18, 2, 27},  {24, 25, 6},
         {24, 14, 25}, {14, 1, 25},  {22, 23, 10}, {22, 15, 23}, {15, 5, 23},
         {16, 21, 5},  {16, 19, 21}, {19, 4, 21},  {19, 20, 4},  {19, 17, 20},
         {17, 3, 20},  {17, 18, 3},  {17, 12, 18}, {12, 2, 18},  {15, 16, 5},
         {15, 13, 16}, {13, 0, 16},  {12, 14, 2},  {12, 13, 14}, {13, 1, 14}},
        torch::dtype(torch::kInt32).device(torch::kCUDA));
#endif

    CUdeviceptr m_vert_buffer_ptr;
    CUdeviceptr m_face_buffer_ptr;
    CUdeviceptr m_d_tlas_output_buffer;
    CUdeviceptr m_d_temp_tlas_buffer_sizes;
    CUdeviceptr m_d_blas_output_buffer;
    CUdeviceptr m_d_instances;

    OptixAccelBufferSizes m_tlas_buffer_sizes;

    void set_global_scale_factor(float scale_factor) {
        m_global_scale_factor[0] = scale_factor;
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
//
#if MAX_BOUNCES > 0
#if OPTIMIZE_EXP_POWER == true
        m_gaussian_exp_power.resize_({num_new_gaussians, 1});
        m_dL_dexp_powers.resize_({num_new_gaussians, 1});
        m_h_params.gaussian_exp_power =
            reinterpret_cast<float *>(m_gaussian_exp_power.data_ptr());
        m_h_params.dL_dexp_powers =
            reinterpret_cast<float *>(m_dL_dexp_powers.data_ptr());
#endif
#if true
        // USE_LEVEL_OF_DETAIL == true
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
#endif
//
#if ATTACH_POSITION == true
        m_gaussian_position.resize_({num_new_gaussians, 3});
        m_dL_dgaussian_position.resize_({num_new_gaussians, 3});
        m_h_params.gaussian_position =
            reinterpret_cast<float3 *>(m_gaussian_position.data_ptr());
        m_h_params.dL_dgaussian_position =
            reinterpret_cast<float3 *>(m_dL_dgaussian_position.data_ptr());
#endif
#if ATTACH_NORMALS == true
        m_gaussian_normal.resize_({num_new_gaussians, 3});
        m_dL_dgaussian_normal.resize_({num_new_gaussians, 3});
        m_h_params.gaussian_normal =
            reinterpret_cast<float3 *>(m_gaussian_normal.data_ptr());
        m_h_params.dL_dgaussian_normal =
            reinterpret_cast<float3 *>(m_dL_dgaussian_normal.data_ptr());
#endif
#if ATTACH_F0 == true
        m_gaussian_f0.resize_({num_new_gaussians, 3});
        m_dL_dgaussian_f0.resize_({num_new_gaussians, 3});
        m_h_params.gaussian_f0 =
            reinterpret_cast<float3 *>(m_gaussian_f0.data_ptr());
        m_h_params.dL_dgaussian_f0 =
            reinterpret_cast<float3 *>(m_dL_dgaussian_f0.data_ptr());
#endif
#if ATTACH_ROUGHNESS == true
        m_gaussian_roughness.resize_({num_new_gaussians, 1});
        m_dL_dgaussian_roughness.resize_({num_new_gaussians, 1});
        m_h_params.gaussian_roughness =
            reinterpret_cast<float *>(m_gaussian_roughness.data_ptr());
        m_h_params.dL_dgaussian_roughness =
            reinterpret_cast<float *>(m_dL_dgaussian_roughness.data_ptr());
#endif
        //
        m_gaussian_mask.resize_({num_new_gaussians, 1});
#endif

        cudaDeviceSynchronize();
    }

    void rebuild_bvh() {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_instances)));
        CUDA_CHECK(
            cudaFree(reinterpret_cast<void *>(m_d_temp_tlas_buffer_sizes)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_tlas_output_buffer)));

        build_tlas();

        m_h_params.handle = m_tlas_handle;

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_d_params),
            &m_h_params,
            sizeof(Params),
            cudaMemcpyHostToDevice));

        cudaDeviceSynchronize();
    }

    void build_bvh() {
#if FUSED_MESH == true
        build_fused_bvh(); //!!!
#else
        // todo free the exsitng ones (free CUdeviceptr)
        build_blas();
        build_tlas();
#endif

        cudaDeviceSynchronize();
    }

    void build_fused_bvh() {
#if USE_POLYCAGE == false
        throw std::runtime_error("Configuration error");
#else
        auto num_gaussians = m_gaussian_means.sizes()[0];

        OptixAccelBuildOptions accel_options_tlas = {};
        accel_options_tlas.buildFlags = m_build_flags;
        accel_options_tlas.operation = OPTIX_BUILD_OPERATION_BUILD;

#if TRI_SOUP_FOR_MESHES == true
        int num_verts = mesh_cage_verts_tri_soup.sizes()[0];
#else
        int num_verts = mesh_cage_verts.sizes()[0];
#endif

        Tensor verts = torch::zeros(
            {num_gaussians * num_verts, 3},
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        Tensor faces = torch::zeros(
            {num_gaussians * mesh_cage_faces.sizes()[0], 3},
            torch::dtype(torch::kInt32).device(torch::kCUDA));

        transformVerts(
#if TRI_SOUP_FOR_MESHES == true
            reinterpret_cast<float *>(mesh_cage_verts_tri_soup.data_ptr()),
#else
            reinterpret_cast<float *>(mesh_cage_verts.data_ptr()),
#endif
            num_verts,
            reinterpret_cast<int *>(mesh_cage_faces.data_ptr()),
            mesh_cage_faces.sizes()[0],
            reinterpret_cast<float *>(verts.data_ptr()),
            reinterpret_cast<int *>(faces.data_ptr()),
            num_gaussians,
            reinterpret_cast<float3 *>(m_camera_position_world.data_ptr()),
            reinterpret_cast<float3 *>(m_gaussian_scales.data_ptr()),
            reinterpret_cast<float4 *>(m_gaussian_rotations.data_ptr()),
            reinterpret_cast<float3 *>(m_gaussian_means.data_ptr()),
            reinterpret_cast<float *>(m_gaussian_opacity.data_ptr()),
            reinterpret_cast<bool *>(m_gaussian_mask.data_ptr()),
            m_global_scale_factor[0].item<float>(),
            m_alpha_threshold,
            m_exp_power);

        verts = verts.to(torch::kHalf).contiguous();

        m_vert_buffer_ptr = reinterpret_cast<CUdeviceptr>(verts.data_ptr());
        m_tlas_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        m_tlas_input.triangleArray.flags = m_aabb_input_flags;
        m_tlas_input.triangleArray.numSbtRecords = 1;
        m_tlas_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_HALF3;
        m_tlas_input.triangleArray.vertexStrideInBytes = sizeof(float3) / 2;
        m_tlas_input.triangleArray.numVertices = num_verts * num_gaussians;
        m_tlas_input.triangleArray.vertexBuffers = &m_vert_buffer_ptr;

#if TRI_SOUP_FOR_MESHES == true
        m_face_buffer_ptr = reinterpret_cast<CUdeviceptr>(faces.data_ptr());
        m_tlas_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
        m_tlas_input.triangleArray.numIndexTriplets = 0;
#else
        m_face_buffer_ptr = reinterpret_cast<CUdeviceptr>(faces.data_ptr());
        m_tlas_input.triangleArray.indexFormat =
            OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        m_tlas_input.triangleArray.indexStrideInBytes = sizeof(uint3);
        m_tlas_input.triangleArray.numIndexTriplets =
            mesh_cage_faces.sizes()[0] * num_gaussians;
        m_tlas_input.triangleArray.indexBuffer = m_face_buffer_ptr;
#endif

        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            m_context,
            &accel_options_tlas,
            &m_tlas_input,
            1,
            &m_tlas_buffer_sizes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&m_d_temp_tlas_buffer_sizes),
            m_tlas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&m_d_tlas_output_buffer),
            m_tlas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(
            m_context,
            0,
            &accel_options_tlas,
            &m_tlas_input,
            1,
            m_d_temp_tlas_buffer_sizes,
            m_tlas_buffer_sizes.tempSizeInBytes,
            m_d_tlas_output_buffer,
            m_tlas_buffer_sizes.outputSizeInBytes,
            &m_tlas_handle,
            nullptr,
            0));
#endif
    }

    void build_blas() {
        OptixAccelBuildOptions accel_options_blas = {};
        accel_options_blas.buildFlags = m_build_flags;
        accel_options_blas.operation = OPTIX_BUILD_OPERATION_BUILD;

#if USE_POLYCAGE == true
        m_vert_buffer_ptr =
            reinterpret_cast<CUdeviceptr>(mesh_cage_verts_half.data_ptr());
        m_blas_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        m_blas_input.triangleArray.flags = m_aabb_input_flags;
        m_blas_input.triangleArray.numSbtRecords = 1;

        m_blas_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_HALF3;
        m_blas_input.triangleArray.vertexStrideInBytes = sizeof(float3) / 2;
        m_blas_input.triangleArray.numVertices =
            mesh_cage_verts_half.sizes()[0];
        m_blas_input.triangleArray.vertexBuffers = &m_vert_buffer_ptr;

        m_face_buffer_ptr =
            reinterpret_cast<CUdeviceptr>(mesh_cage_faces.data_ptr());
        m_blas_input.triangleArray.indexFormat =
            OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        m_blas_input.triangleArray.indexStrideInBytes = sizeof(uint3);
        m_blas_input.triangleArray.numIndexTriplets =
            mesh_cage_faces.sizes()[0];
        m_blas_input.triangleArray.indexBuffer = m_face_buffer_ptr;
#else
        m_d_aabb_buffer =
            reinterpret_cast<CUdeviceptr>(unit_bbox_tensor.data_ptr());
        m_blas_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        m_blas_input.customPrimitiveArray.aabbBuffers = &m_d_aabb_buffer;
        m_blas_input.customPrimitiveArray.numPrimitives = 1;
        m_blas_input.customPrimitiveArray.flags = m_aabb_input_flags;
        m_blas_input.customPrimitiveArray.numSbtRecords = 1;
#endif

        OptixAccelBufferSizes blas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            m_context,
            &accel_options_blas,
            &m_blas_input,
            1,
            &blas_buffer_sizes));

        CUdeviceptr d_temp_blas_buffer_sizes;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_temp_blas_buffer_sizes),
            blas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&m_d_blas_output_buffer),
            blas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(
            m_context,
            0,
            &accel_options_blas,
            &m_blas_input,
            1,
            d_temp_blas_buffer_sizes,
            blas_buffer_sizes.tempSizeInBytes,
            m_d_blas_output_buffer,
            blas_buffer_sizes.outputSizeInBytes,
            &m_blas_handle,
            nullptr,
            0));

        CUDA_CHECK(
            cudaFree(reinterpret_cast<void *>(d_temp_blas_buffer_sizes)));

#if COMPACTION == true
        size_t *d_compactedSize;
        OptixAccelEmitDesc property = {};
        property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        property.result = d_compactedSize;

        size_t compactedSize;
        cudaMemcpy(
            &compactedSize,
            d_compactedSize,
            sizeof(size_t),
            cudaMemcpyDeviceToHost);

        void *d_compactedOutputBuffer;
        cudaMalloc(&d_compactedOutputBuffer, compactedSize);

        if (compactedSize < bufferSizes.outputSizeInBytes) {
            optixAccelCompact(
                optixContext,
                cuStream,
                accelHandle,
                d_compactedOutputBuffer,
                compactedSize,
                &compactedAccelHandle);
        }
#endif
    }

    void build_tlas() {
        auto num_gaussians = m_gaussian_means.sizes()[0];

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&m_d_instances),
            sizeof(OptixInstance) * num_gaussians));
        populateBVH(
            reinterpret_cast<OptixInstance *>(m_d_instances),
            m_blas_handle,
            num_gaussians,
            reinterpret_cast<float3 *>(m_camera_position_world.data_ptr()),
            reinterpret_cast<float3 *>(m_gaussian_scales.data_ptr()),
            reinterpret_cast<float4 *>(m_gaussian_rotations.data_ptr()),
            reinterpret_cast<float3 *>(m_gaussian_means.data_ptr()),
            reinterpret_cast<float *>(m_gaussian_opacity.data_ptr()),
            reinterpret_cast<float *>(m_gaussian_lod_mean.data_ptr()),
            reinterpret_cast<float *>(m_gaussian_lod_scale.data_ptr()),
            reinterpret_cast<bool *>(m_gaussian_mask.data_ptr()),
            m_global_scale_factor[0].item<float>(),
            m_alpha_threshold,
#if OPTIMIZE_EXP_POWER
            reinterpret_cast<float *>(m_gaussian_exp_power.data_ptr())
#else
            m_exp_power
#endif
        );

        OptixAccelBuildOptions accel_options_tlas = {};
        accel_options_tlas.buildFlags =
            m_build_flags | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
        accel_options_tlas.operation = OPTIX_BUILD_OPERATION_BUILD;

        m_tlas_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        m_tlas_input.instanceArray.instances = m_d_instances;
        m_tlas_input.instanceArray.numInstances = num_gaussians;

        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            m_context,
            &accel_options_tlas,
            &m_tlas_input,
            1,
            &m_tlas_buffer_sizes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&m_d_temp_tlas_buffer_sizes),
            m_tlas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&m_d_tlas_output_buffer),
            m_tlas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(
            m_context,
            0,
            &accel_options_tlas,
            &m_tlas_input,
            1,
            m_d_temp_tlas_buffer_sizes,
            m_tlas_buffer_sizes.tempSizeInBytes,
            m_d_tlas_output_buffer,
            m_tlas_buffer_sizes.outputSizeInBytes,
            &m_tlas_handle,
            nullptr,
            0));
    }

    void update_bvh() {
#if FUSED_MESH == true
        return; //!! tmp
#endif
        // Update XForms
        auto num_gaussians = m_gaussian_means.sizes()[0];
        populateBVH(
            reinterpret_cast<OptixInstance *>(m_d_instances),
            m_blas_handle,
            num_gaussians,
            reinterpret_cast<float3 *>(m_camera_position_world.data_ptr()),
            reinterpret_cast<float3 *>(m_gaussian_scales.data_ptr()),
            reinterpret_cast<float4 *>(m_gaussian_rotations.data_ptr()),
            reinterpret_cast<float3 *>(m_gaussian_means.data_ptr()),
            reinterpret_cast<float *>(m_gaussian_opacity.data_ptr()),
            reinterpret_cast<float *>(m_gaussian_lod_mean.data_ptr()),
            reinterpret_cast<float *>(m_gaussian_lod_scale.data_ptr()),
            reinterpret_cast<bool *>(m_gaussian_mask.data_ptr()),
            m_global_scale_factor[0].item<float>(),
            m_alpha_threshold,
#if OPTIMIZE_EXP_POWER
            reinterpret_cast<float *>(m_gaussian_exp_power.data_ptr())
#else
            m_exp_power
#endif
        );

        // Update TLAS
        OptixAccelBuildOptions accel_options_tlas = {};
        accel_options_tlas.buildFlags =
            m_build_flags | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
        accel_options_tlas.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(
            m_context,
            0,
            &accel_options_tlas,
            &m_tlas_input,
            1,
            m_d_temp_tlas_buffer_sizes,
            m_tlas_buffer_sizes.tempSizeInBytes,
            m_d_tlas_output_buffer,
            m_tlas_buffer_sizes.outputSizeInBytes,
            &m_tlas_handle,
            nullptr,
            0));
    }

    Tensor m_denoise =
        torch::tensor({true}, torch::dtype(torch::kBool).device(torch::kCUDA));
    Tensor m_accumulate =
        torch::tensor({false}, torch::dtype(torch::kBool).device(torch::kCUDA));
    Tensor m_denoise_glossy =
        torch::tensor({false}, torch::dtype(torch::kBool).device(torch::kCUDA));
    Tensor m_num_samples =
        torch::tensor({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    Tensor m_num_bounces = torch::tensor(
        {MAX_BOUNCES}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    void sample() {
        m_num_hits_per_pixel.zero_();
        m_num_traversed_per_pixel.zero_();

#if LOG_ALL_HITS == true
        m_prev_hit_per_pixel.fill_(999999999);
        m_total_hits.fill_(0);
#if BACKWARDS_PASS == true
        m_prev_hit_per_pixel_for_backprop.fill_(999999999);
        m_total_hits_for_backprop.fill_(0);
#endif
#endif

        for (int h = 0; h < NUM_CHUNKS * NUM_CHUNKS; h++) {
            OPTIX_CHECK(optixLaunch(
                m_pipeline,
                nullptr,
                m_d_params,
                sizeof(Params),
                &m_sbt,
                m_width / TILE_SIZE / NUM_CHUNKS,
                m_height / TILE_SIZE / NUM_CHUNKS,
                1));
        }

        m_iteration +=
            1; // used to seed the random noise, need to increment every sample
    }

    void raytrace() {
#if MEASURE_LOSS == true
        m_loss_tensor.fill_(0.0);
#endif

#if BACKWARDS_PASS == true
        bool grads_enabled = torch::GradMode::is_enabled();
        m_grads_enabled.fill_(grads_enabled);
#endif

#if DEBUG_CHEAP_TMAX_ESTIMATE == true
        m_t_maxes.fill_(100000.0);
#if BACKWARDS_PASS == true
        m_grads_enabled.fill_(false);
#endif
        m_cheap_approx.fill_(true);
        OPTIX_CHECK(optixLaunch(
            m_pipeline,
            nullptr,
            m_d_params,
            sizeof(Params),
            &m_sbt,
            m_width / CHEAP_TMAX_DOWNSAMPLING,
            m_height / CHEAP_TMAX_DOWNSAMPLING,
            1));
        m_cheap_approx.fill_(false);
#if BACKWARDS_PASS == true
        m_grads_enabled.fill_(grads_enabled);
#endif
#endif

        if (m_num_samples.item<int>() > 1) {
            Tensor m_output = torch::zeros_like(m_output_rgb);
            for (int i = 0; i < m_num_samples.item<int>(); i++) {
                sample();
                m_output += m_output_rgb;
            }
            m_output_rgb.copy_(m_output);
        } else {
            sample();
        }

#if DENOISER == true
        if (!torch::GradMode::is_enabled() && m_denoise.item<bool>()) {

            OPTIX_CHECK(optixDenoiserInvoke(
                m_denoiser,
                nullptr, // CUDA stream
                &m_params_denoiser,
                m_state,
                m_state_size,
                &m_guideLayer,
                m_layers.data(),
                static_cast<unsigned int>(m_layers.size()),
                0, // input offset X
                0, // input offset y
                m_scratch,
                m_scratch_size));
        }
#endif

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

    void configure(
        double transmittance_threshold = -1,
        double alpha_threshold = -1,
        double exp_power = -1) {
        if (alpha_threshold != -1) {
            m_alpha_threshold = (float)alpha_threshold;
            m_h_params.alpha_threshold = m_alpha_threshold;
        }
        if (transmittance_threshold != -1) {
            m_transmittance_threshold = (float)transmittance_threshold;
            m_h_params.transmittance_threshold = m_transmittance_threshold;
        }
        if (exp_power != -1) {
            m_exp_power = (float)exp_power;
            m_h_params.exp_power = m_exp_power;
        }

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_d_params),
            &m_h_params,
            sizeof(Params),
            cudaMemcpyHostToDevice));
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
};

TORCH_LIBRARY(gausstracer, m) {
    m.class_<Raytracer>("Raytracer")
        .def(torch::init<int64_t, int64_t, int64_t>())
        .def("rebuild_bvh", &Raytracer::rebuild_bvh)
        .def("update_bvh", &Raytracer::update_bvh)
        .def("set_losses", &Raytracer::set_losses)
        .def("raytrace", &Raytracer::raytrace)
        .def("configure", &Raytracer::configure)
        .def("set_camera", &Raytracer::set_camera)
        .def("resize", &Raytracer::resize)
        // Render settings
        .def_readonly("denoise", &Raytracer::m_denoise)
        .def_readonly("accumulate", &Raytracer::m_accumulate)
        .def_readonly("num_samples", &Raytracer::m_num_samples)
        .def_readonly("num_bounces", &Raytracer::m_num_bounces)
        // Camera params
        .def_readonly("camera_rotation_c2w", &Raytracer::m_camera_rotation_c2w)
        .def_readonly("camera_rotation_w2c", &Raytracer::m_camera_rotation_w2c)
        .def_readonly(
            "camera_position_world", &Raytracer::m_camera_position_world)
        .def_readonly(
            "vertical_fov_radians", &Raytracer::m_vertical_fov_radians)
        .def_readonly("init_blur_sigma", &Raytracer::m_init_blur_sigma)
        // Gaussian params
        .def_readonly("global_scale_factor", &Raytracer::m_global_scale_factor)
        .def_readonly("gaussian_rgb", &Raytracer::m_gaussian_rgb)
        .def_readonly("dL_drgb", &Raytracer::m_dL_drgb)
        .def_readonly("gaussian_opacity", &Raytracer::m_gaussian_opacity)
        .def_readonly("dL_dopacity", &Raytracer::m_dL_dopacity)
        .def_readonly("gaussian_scales", &Raytracer::m_gaussian_scales)
        .def_readonly("dL_dscales", &Raytracer::m_dL_dscales)
        .def_readonly("gaussian_rotations", &Raytracer::m_gaussian_rotations)
        .def_readonly("dL_drotations", &Raytracer::m_dL_drotations)
        .def_readonly("gaussian_means", &Raytracer::m_gaussian_means)
        .def_readonly("dL_dmeans", &Raytracer::m_dL_dmeans)
        .def_readonly("gaussian_exp_power", &Raytracer::m_gaussian_exp_power)
        .def_readonly("dL_dexp_powers", &Raytracer::m_dL_dexp_powers)
        .def_readonly("gaussian_lod_mean", &Raytracer::m_gaussian_lod_mean)
        .def_readonly(
            "dL_dgaussian_lod_mean", &Raytracer::m_dL_dgaussian_lod_mean)
        .def_readonly("gaussian_lod_scale", &Raytracer::m_gaussian_lod_scale)
        .def_readonly(
            "dL_dgaussian_lod_scale", &Raytracer::m_dL_dgaussian_lod_scale)
        .def_readonly("gaussian_mask", &Raytracer::m_gaussian_mask)
        .def_readonly("gaussian_position", &Raytracer::m_gaussian_position)
        .def_readonly(
            "dL_dgaussian_position", &Raytracer::m_dL_dgaussian_position)
        .def_readonly("gaussian_normal", &Raytracer::m_gaussian_normal)
        .def_readonly("dL_dgaussian_normal", &Raytracer::m_dL_dgaussian_normal)
        .def_readonly("gaussian_f0", &Raytracer::m_gaussian_f0)
        .def_readonly("dL_dgaussian_f0", &Raytracer::m_dL_dgaussian_f0)
        .def_readonly("gaussian_roughness", &Raytracer::m_gaussian_roughness)
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
        .def_readonly("accumulated_normal", &Raytracer::m_accumulated_normal)
        .def_readonly("accumulated_depth", &Raytracer::m_accumulated_depth)
        .def_readonly("accumulated_f0", &Raytracer::m_accumulated_f0)
        .def_readonly(
            "accumulated_roughness", &Raytracer::m_accumulated_roughness)
        .def_readonly(
            "accumulated_sample_count", &Raytracer::m_accumulated_sample_count)
        .def_readonly("output_t", &Raytracer::m_output_t)
        .def_readonly(
            "output_incident_radiance", &Raytracer::m_output_incident_radiance)
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
            "output_glossy_irradiance", &Raytracer::m_output_glossy_irradiance)
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
        .def_readonly("output_effective_F0", &Raytracer::m_output_effective_F0)
        .def_readonly(
            "output_effective_roughness",
            &Raytracer::m_output_effective_roughness)
        .def_readonly(
            "output_effective_normal", &Raytracer::m_output_effective_normal)
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
            "target_glossy_irradiance", &Raytracer::m_target_glossy_irradiance)
        .def_readonly("loss_tensor", &Raytracer::m_loss_tensor)
        // Other buffers
        .def_readonly("num_hits_per_pixel", &Raytracer::m_num_hits_per_pixel)
        .def_readonly(
            "num_traversed_per_pixel", &Raytracer::m_num_traversed_per_pixel)
        .def_readonly("t_maxes", &Raytracer::m_t_maxes);
}
