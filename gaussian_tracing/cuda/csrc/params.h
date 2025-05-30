//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include "flags.h"
#include <cuda.h>
#include <optix.h>

#define GLM_FORCE_CUDA

#if ENABLE_DEBUG_DUMP == true
#include "dump.h"
#endif

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

#if ATTACH_POSITION == true
    const float3 *__restrict__ gaussian_position;
    float3 *__restrict__ dL_dgaussian_position;
#endif
#if ATTACH_NORMALS == true
    const float3 *__restrict__ gaussian_normal;
    float3 *__restrict__ dL_dgaussian_normal;
#endif
#if ATTACH_F0 == true
    const float3 *__restrict__ gaussian_f0;
    float3 *__restrict__ dL_dgaussian_f0;
#endif
#if ATTACH_ROUGHNESS == true
    const float *__restrict__ gaussian_roughness;
    float *__restrict__ dL_dgaussian_roughness;
#endif
#if ATTACH_SPECULAR == true
    const float *__restrict__ gaussian_specular;
    float *__restrict__ dL_dgaussian_specular;
#endif
#if ATTACH_ALBEDO == true
    const float3 *__restrict__ gaussian_albedo;
    float3 *__restrict__ dL_dgaussian_albedo;
#endif
#if ATTACH_METALNESS == true
    const float *__restrict__ gaussian_metalness;
    float *__restrict__ dL_dgaussian_metalness;
#endif

    const float *__restrict__ gaussian_opacity;
    float *__restrict__ dL_dopacity;

#if true
    // USE_LEVEL_OF_DETAIL == true
    const float *__restrict__ gaussian_lod_mean;
    float *__restrict__ dL_dgaussian_lod_mean;

    const float *__restrict__ gaussian_lod_scale;
    float *__restrict__ dL_dgaussian_lod_scale;
#endif

    const float *__restrict__ gaussian_alpha;

    float transmittance_weight;
    const float3 *__restrict__ target_rgb;
#if ATTACH_POSITION == true
    const float3 *__restrict__ target_position;
#endif
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
    const float *__restrict__ target_depth;
#endif
#if ATTACH_NORMALS == true
    const float3 *__restrict__ target_normal;
#endif
#if ATTACH_F0 == true
    const float3 *__restrict__ target_f0;
#endif
#if ATTACH_ROUGHNESS == true
    const float *__restrict__ target_roughness;
#endif
#if ATTACH_SPECULAR == true
    const float *__restrict__ target_specular;
#endif
#if ATTACH_ALBEDO == true
    const float3 *__restrict__ target_albedo;
#endif
#if ATTACH_METALNESS == true
    const float *__restrict__ target_metalness;
#endif

#if USE_GT_DIFFUSE_IRRADIANCE == true
    const float3 *__restrict__ output_diffuse_irradiance;
    const float3 *__restrict__ target_diffuse_irradiance;
#endif
#if REFLECTIONS_FROM_GT_GLOSSY_IRRADIANCE == true
    const float3 *__restrict__ output_glossy_irradiance;
    const float3 *__restrict__ target_glossy_irradiance;
#endif

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
    float *__restrict__ output_specular;
    float3 *__restrict__ output_albedo;
    float *__restrict__ output_metalness;
    float *__restrict__ output_distortion;
#if SAVE_LOD_IMAGES == true
    float *__restrict__ output_lod_mean;
    float *__restrict__ output_lod_scale;
    float *__restrict__ output_ray_lod;
#endif

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
    float specular_loss_weight;
    float albedo_loss_weight;
    float metalness_loss_weight;
    float regular_loss_weight;

    float *global_scale_factor;

    float *init_blur_sigma;

    float *__restrict__ loss_tensor;

    int *__restrict__ num_hits_per_pixel;
    int *__restrict__ num_traversed_per_pixel;

#if SAVE_RAY_IMAGES == true
    float3 *__restrict__ output_ray_origin;
    float3 *__restrict__ output_ray_direction;
#endif

#if USE_GT_BRDF == true
    const float3 *__restrict__ target_brdf;
#else
    float2 *lut;
#endif
#if SAVE_LUT_IMAGES == true
    float3 *output_effective_reflection_position;
    float3 *output_effective_reflection_normal;
    float3 *output_effective_F0;
    float *output_effective_roughness;
    float3 *output_effective_normal;
#if USE_GT_BRDF == false
    float2 *__restrict__ output_lut_values;
    float *__restrict__ output_n_dot_v;
#endif
#endif

#if OPTIMIZE_EXP_POWER == true
    const float *gaussian_exp_power;
    float *__restrict__ dL_dexp_powers;
#endif

#if ENABLE_DEBUG_DUMP == true
    Dump *dump;
#endif

#if LOG_ALL_HITS == true
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
#endif

#if BACKWARDS_PASS == true
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
#endif

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

#define NULL_GAUSSIAN_ID 1U << 30
