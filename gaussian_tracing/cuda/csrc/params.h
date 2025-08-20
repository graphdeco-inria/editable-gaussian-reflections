#pragma once

#include "flags.h"
#include <cuda.h>
#include <optix.h>

struct Params {
    uint32_t image_width;
    uint32_t image_height;

    float *vertical_fov_radians;
    float *camera_znear;
    float *camera_zfar;
    float *max_lod_size;

    float alpha_threshold;
    float exp_power;
    float transmittance_threshold;

    bool *grads_enabled;
    bool *cheap_approx;
    int *iteration;

    const float3 *__restrict__ camera_rotation_c2w;
    const float3 *__restrict__ camera_rotation_w2c;
    const float3 *__restrict__ camera_position_world;

    const float3 *__restrict__ origins;
    const float3 *__restrict__ directions;

    const int *__restrict__ mask;

    const float4 *__restrict__ gaussian_xform;
    float4 *__restrict__ dL_dxform;

    const float3 *__restrict__ gaussian_rgb;
    float3 *__restrict__ dL_drgb;

    const bool *__restrict__ denoise;
    const int *__restrict__ num_samples;
    const int *__restrict__ num_bounces;

    const float3 *__restrict__ gaussian_position;
    float3 *__restrict__ dL_dgaussian_position;
    const float3 *__restrict__ gaussian_normal;
    float3 *__restrict__ dL_dgaussian_normal;
    const float3 *__restrict__ gaussian_f0;
    float3 *__restrict__ dL_dgaussian_f0;
    const float *__restrict__ gaussian_roughness;
    float *__restrict__ dL_dgaussian_roughness;

    const float *__restrict__ gaussian_opacity;
    float *__restrict__ dL_dopacity;

    const float *__restrict__ gaussian_lod_mean;
    float *__restrict__ dL_dgaussian_lod_mean;

    const float *__restrict__ gaussian_lod_scale;
    float *__restrict__ dL_dgaussian_lod_scale;

    const float *__restrict__ gaussian_alpha;

    float transmittance_weight;
    const float3 *__restrict__ target_rgb;
    const float3 *__restrict__ target_position;
    const float *__restrict__ target_depth;
    const float3 *__restrict__ target_normal;
    const float3 *__restrict__ target_f0;
    const float *__restrict__ target_roughness;

    float3 *__restrict__ output_rgb;
    float3 *__restrict__ accumulated_rgb;
    float3 *__restrict__ accumulated_normal;
    float *__restrict__ accumulated_depth;
    float3 *__restrict__ accumulated_f0;
    float *__restrict__ accumulated_roughness;

    int *accumulated_sample_count;
    float2 *__restrict__ output_t; // first channel is where integration ends,
                                   // second channel is the exact transmittance
                                   // of all gaussians

    float3 *__restrict__ output_incident_radiance;

#if MAX_BOUNCES > 0
    float3 *__restrict__ target_diffuse;
    float3 *__restrict__ target_glossy;
    float3 *__restrict__ output_brdf;
#endif

    float3 *__restrict__ output_position;
    float *__restrict__ output_depth;
    float3 *__restrict__ output_normal;
    float3 *__restrict__ output_f0;
    float *__restrict__ output_roughness;
    float *__restrict__ output_distortion;

    float *__restrict__ t_maxes;
    float *__restrict__ t_mins;

    const float4 *__restrict__ gaussian_rotations;
    float4 *__restrict__ dL_drotations;

    const float3 *__restrict__ gaussian_scales;
    float3 *__restrict__ dL_dscales;

    const float3 *__restrict__ gaussian_means;
    float3 *__restrict__ dL_dmeans;

    float *__restrict__ gaussian_total_weight;
    float3 *__restrict__ densification_gradient_diffuse;
    float3 *__restrict__ densification_gradient_glossy;

    float4 *__restrict__ random_numbers; // input from torch
    uint32_t *__restrict__ random_seeds; // set from within the optix shader

    float diffuse_loss_weight;
    float glossy_loss_weight;
    float position_loss_weight;
    float normal_loss_weight;
    float f0_loss_weight;
    float roughness_loss_weight;

    float *global_scale_factor;

    float *init_blur_sigma;

    float *__restrict__ loss_tensor;

    int *__restrict__ num_hits_per_pixel;
    int *__restrict__ num_traversed_per_pixel;

    float3 *__restrict__ output_ray_origin;
    float3 *__restrict__ output_ray_direction;

    float2 *lut;

    uint32_t *total_hits; // total # of hits

    // hit data, total_hits entries
    uint32_t *__restrict__ all_gaussian_ids;
    float *__restrict__ all_distances;
    float *__restrict__ all_half_chord_lengths; // contains TILE_SIZE*TILE_SIZE
                                                // entries

    float *__restrict__ all_alphas;      // contains TILE_SIZE*TILE_SIZE entries
    float *__restrict__ all_gaussvals;   // contains TILE_SIZE*TILE_SIZE entries
    float3 *__restrict__ all_local_hits; // contains TILE_SIZE*TILE_SIZE entries

    uint32_t *__restrict__ all_prev_hits;
    uint32_t *__restrict__ all_slab_idx;

    // per-pixel record of the last hit, 1 entry per slab
    uint32_t *__restrict__ prev_hit_per_pixel;

    uint32_t *total_hits_for_backprop; // total # of hits

    // hit data, total_hits entries
    uint32_t *__restrict__ all_gaussian_ids_for_backprop;

    float *__restrict__ all_alphas_for_backprop; // contains TILE_SIZE*TILE_SIZE
                                                 // entries
    float3 *__restrict__ all_local_hits_for_backprop; // contains
                                                      // TILE_SIZE*TILE_SIZE
                                                      // entries
    float *__restrict__ all_distances_for_backprop;
    float *__restrict__ all_Ts_for_backprop; // contains TILE_SIZE*TILE_SIZE
                                             // entries
    float
        *__restrict__ all_gaussvals_for_backprop; // contains
                                                  // TILE_SIZE*TILE_SIZE entries

    uint32_t *__restrict__ all_prev_hits_for_backprop;

    // per-pixel record of the last hit
    uint32_t *__restrict__ prev_hit_per_pixel_for_backprop;

    OptixTraversableHandle handle;
};

struct HitData {
    float r, g, b;
};

struct RayGenData {
    float r, g, b;
};

struct MissData {
    float r, g, b;
};

template <typename T> struct SbtRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

extern "C" {
__constant__ Params params;
}

#define NULL_GAUSSIAN_ID 1U << 30
