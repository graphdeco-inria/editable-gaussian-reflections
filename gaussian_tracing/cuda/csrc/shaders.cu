#include "params.h"

#include "utils/common.h"
#include "utils/helpers.cu"

#include "backward_pass.cu"
#include "forward_pass.cu"

extern "C" __global__ void __intersection__gaussian() {
    // * Fetch config
    int step = optixGetPayload_2();
    float full_T = __uint_as_float(optixGetPayload_3());
    float alpha_threshold = __uint_as_float(optixGetPayload_4());
    float exp_power = __uint_as_float(optixGetPayload_5());
    float backfacing_max_dist = 0.1f;
    float backfacing_invalid_normal_threshold = 0.9f;

    // * Fetch ray data
    float3 local_origin = optixGetObjectRayOrigin();
    float3 local_direction = optixGetObjectRayDirection();

    // * Load gaussian data
    const uint32_t gaussian_id = optixGetInstanceIndex();
    float opacity = sigmoid_act(params.gaussian_opacity[gaussian_id]);
    float3 scale = exp_act(params.gaussian_scales[gaussian_id]);

    // * Compute pixel index
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();

    uint32_t ray_id =
        (idx.y * TILE_SIZE) * params.image_width + idx.x * TILE_SIZE;

    float norm = length(local_direction);
    local_direction /= norm;
    float local_hit_distance_along_ray = dot(-local_origin, local_direction);

    // * Compute the hit point along the ray
    float sorting_distance = local_hit_distance_along_ray / norm;
    float3 local_hit_unscaled =
        (local_origin + local_hit_distance_along_ray * local_direction);

    // * Clip the gaussian at alpha_threshold, taking bounding box scale
    // into account todo: if end up implementing tiling, it is possible to
    // optimize further considering the worse ray of the tile, or by using
    // capsules
    float sq_dist = dot(local_hit_unscaled, local_hit_unscaled);
    if (sq_dist > 1.0f) {
        return;
    }

    if (dot(optixGetObjectRayOrigin(), optixGetObjectRayDirection()) > 0.0) {
        return;
    }

    if (step != 0 && sorting_distance < backfacing_max_dist) {
        float3 gaussian_normal = params.gaussian_normal[gaussian_id];
        if (length(gaussian_normal) > backfacing_invalid_normal_threshold &&
            dot(gaussian_normal, local_direction) > 0.0f) {
            return;
        }
    }

    // * Compute alpha value
    float3 local_hit =
        local_hit_unscaled *
        compute_scaling_factor(opacity, alpha_threshold, exp_power);
    float gaussval = eval_gaussian(local_hit, exp_power);
    float alpha = compute_alpha(gaussval, opacity, alpha_threshold);

    // * Compute the exact total transmittance for the ray
    full_T *= 1.0 - alpha;
    optixSetPayload_3(__float_as_uint(full_T));

    int hit_idx = atomicAdd(params.total_hits, 1);
    params.all_gaussian_ids[hit_idx] = gaussian_id;
    params.all_distances[hit_idx] = sorting_distance;
    params.all_alphas[hit_idx] = alpha;
    if (*params.grads_enabled) { // todo check impact of this on forward pass
        params.all_gaussvals[hit_idx] = gaussval;
        params.all_local_hits[hit_idx] = local_hit;
    }
    params.all_prev_hits[hit_idx] = params.prev_hit_per_pixel[ray_id];
    params.prev_hit_per_pixel[ray_id] = hit_idx;

    return;
}

extern "C" __global__ void __raygen__rg() {
    // * Compute pixel index
    int num_pixels = params.image_width * params.image_height;
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
    float eps_ray_surface_offset = 0.01f;
    float eps_min_roughness = 0.01f;
    float reflection_invalid_normal_threshold = 0.7f;

    uint currrent_chunk_x = 0;
    uint currrent_chunk_y = 0; // todo
    uint32_t ray_id =
        (idx.y * (currrent_chunk_y + 1) * TILE_SIZE) * params.image_width +
        idx.x * TILE_SIZE * (currrent_chunk_x + 1);

    unsigned int seed =
        tea<4>(idx.y * params.image_width + idx.x, *params.iteration);

    // * Compute the primary ray coordinates
    float3 tile_origin;
    float3 tile_direction;
    float3 origin[TILE_SIZE * TILE_SIZE];
    float3 direction[TILE_SIZE * TILE_SIZE];
    const float2 jitter_offset =
        make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
    float2 idxf = make_float2(idx.x, idx.y) + jitter_offset;

    float view_size = tan(*params.vertical_fov_radians / 2);
    float aspect_ratio = float(dim.x) / float(dim.y);

    float _y = (idxf.y * TILE_SIZE + float(TILE_SIZE) / 2) /
               (float(dim.y) * TILE_SIZE);
    float _x = (idxf.x * TILE_SIZE + float(TILE_SIZE) / 2) /
               (float(dim.x) * TILE_SIZE);
    float y = view_size * (1.0f - 2.0f * _y);
    float x = aspect_ratio * view_size * (2.0f * _x - 1.0f);
    tile_direction = normalize(
        params.camera_rotation_w2c[0] * x + params.camera_rotation_w2c[1] * y -
        params.camera_rotation_w2c[2]);
    tile_origin = *params.camera_position_world;

    for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
        origin[k] = tile_origin;
        direction[k] = tile_direction;
    }

    float3 output_rgb[MAX_BOUNCES + 2][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 2; i++)
        fill_array(
            output_rgb[i],
            TILE_SIZE * TILE_SIZE,
            make_float3(0.0f, 0.0f, 0.0f));
    float3 output_rgb_raw[MAX_BOUNCES + 2][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 2; i++)
        fill_array(
            output_rgb_raw[i],
            TILE_SIZE * TILE_SIZE,
            make_float3(0.0f, 0.0f, 0.0f));
    float3 incident_radiance[MAX_BOUNCES + 1][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        fill_array(
            incident_radiance[i],
            TILE_SIZE * TILE_SIZE,
            make_float3(0.0f, 0.0f, 0.0f));
    float2 output_t[MAX_BOUNCES + 1][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        fill_array(output_t[i], TILE_SIZE * TILE_SIZE, make_float2(1.0f, 1.0f));
    //
    float3 output_position[MAX_BOUNCES + 1][NUM_CLUSTERS]
                          [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                output_position[i][c],
                TILE_SIZE * TILE_SIZE,
                make_float3(0.0f, 0.0f, 0.0f));
    float output_depth[MAX_BOUNCES + 1][NUM_CLUSTERS][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(output_depth[i][c], TILE_SIZE * TILE_SIZE, 0.0f);
    float3 output_normal[MAX_BOUNCES + 1][NUM_CLUSTERS][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                output_normal[i][c],
                TILE_SIZE * TILE_SIZE,
                make_float3(0.0f, 0.0f, 0.0f));
    float3 output_f0[MAX_BOUNCES + 1][NUM_CLUSTERS][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                output_f0[i][c],
                TILE_SIZE * TILE_SIZE,
                make_float3(0.0f, 0.0f, 0.0f));
    float output_roughness[MAX_BOUNCES + 1][NUM_CLUSTERS]
                          [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(output_roughness[i][c], TILE_SIZE * TILE_SIZE, 0.0f);
    float output_distortion[MAX_BOUNCES + 1][NUM_CLUSTERS]
                           [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(output_distortion[i][c], TILE_SIZE * TILE_SIZE, 0.0f);
    float output_lod_mean[MAX_BOUNCES + 1][NUM_CLUSTERS][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(output_lod_mean[i][c], TILE_SIZE * TILE_SIZE, 0.0f);
    float output_lod_scale[MAX_BOUNCES + 1][NUM_CLUSTERS]
                          [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(output_lod_scale[i][c], TILE_SIZE * TILE_SIZE, 0.0f);
    float output_ray_lod[MAX_BOUNCES + 1][NUM_CLUSTERS][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(output_ray_lod[i][c], TILE_SIZE * TILE_SIZE, 0.0f);
    //
    float3 remaining_rgb[MAX_BOUNCES + 1][NUM_CLUSTERS][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                remaining_rgb[i][c],
                TILE_SIZE * TILE_SIZE,
                make_float3(0.0f, 0.0f, 0.0f));
    float3 remaining_position[MAX_BOUNCES + 1][NUM_CLUSTERS]
                             [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                remaining_position[i][c],
                TILE_SIZE * TILE_SIZE,
                make_float3(0.0f, 0.0f, 0.0f));
    float remaining_depth[MAX_BOUNCES + 1][NUM_CLUSTERS][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                remaining_position[i][c],
                TILE_SIZE * TILE_SIZE,
                make_float3(0.0f, 0.0f, 0.0f));
    float3 remaining_normal[MAX_BOUNCES + 1][NUM_CLUSTERS]
                           [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                remaining_normal[i][c],
                TILE_SIZE * TILE_SIZE,
                make_float3(0.0f, 0.0f, 0.0f));
    float3 remaining_f0[MAX_BOUNCES + 1][NUM_CLUSTERS][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                remaining_f0[i][c],
                TILE_SIZE * TILE_SIZE,
                make_float3(0.0f, 0.0f, 0.0f));
    float remaining_roughness[MAX_BOUNCES + 1][NUM_CLUSTERS]
                             [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(remaining_roughness[i][c], TILE_SIZE * TILE_SIZE, 0.0f);
    float remaining_distortion[MAX_BOUNCES + 1][NUM_CLUSTERS]
                              [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(remaining_distortion[i][c], TILE_SIZE * TILE_SIZE, 0.0f);
    float remaining_lod_mean[MAX_BOUNCES + 1][NUM_CLUSTERS]
                            [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(remaining_lod_mean[i][c], TILE_SIZE * TILE_SIZE, 0.0f);
    float remaining_used_lod[MAX_BOUNCES + 1][NUM_CLUSTERS]
                            [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(remaining_used_lod[i][c], TILE_SIZE * TILE_SIZE, 0.0f);
    //
    float3 output_throughput[MAX_BOUNCES + 1][NUM_CLUSTERS]
                            [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                output_throughput[i][c],
                TILE_SIZE * TILE_SIZE,
                make_float3(1.0f, 1.0f, 1.0f));

    float2 output_lut_values[MAX_BOUNCES + 1][NUM_CLUSTERS]
                            [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                output_lut_values[i][c],
                TILE_SIZE * TILE_SIZE,
                make_float2(1.0f, 1.0f));

    float pixel_size =
        2.0f * tan(*params.vertical_fov_radians / 2) / params.image_height;

    float initial_lod[MAX_BOUNCES + 1];
    fill_array(initial_lod, MAX_BOUNCES + 1, 0.0f);
    float lod_by_distance[MAX_BOUNCES + 1];
    fill_array(
        lod_by_distance,
        MAX_BOUNCES + 1,
        (0.637f + *params.init_blur_sigma) * pixel_size);

    int num_hits[MAX_BOUNCES + 1];
    fill_array(num_hits, MAX_BOUNCES + 1, 0);

    float cluster_weights[MAX_BOUNCES + 1][NUM_CLUSTERS][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(cluster_weights[i][c], TILE_SIZE * TILE_SIZE, 0.0f);
    int selected_clusters[MAX_BOUNCES + 1][TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        fill_array(selected_clusters[i], TILE_SIZE * TILE_SIZE, 0);

    int total_effective_steps = 0;
    float3 reflected_origin = tile_origin;

// * Forward pass
#pragma unroll
    for (int step = 0; step < *params.num_bounces + 1; step++) {
        total_effective_steps++;

        // * Compute the indicent radiance
        forward_pass(
            step,
            initial_lod[step],
            lod_by_distance[step],
            ray_id,
            tile_origin,
            reflected_origin,
            tile_direction,
            origin,
            direction,
            cluster_weights[step],
            output_rgb[step],
            output_t[step],
            output_position[step],
            output_depth[step],
            output_normal[step],
            output_f0[step],
            output_roughness[step],
            output_distortion[step],
            output_lod_mean[step],
            output_lod_scale[step],
            output_ray_lod[step],
            remaining_rgb[step],
            remaining_position[step],
            remaining_depth[step],
            remaining_normal[step],
            remaining_f0[step],
            remaining_roughness[step],
            remaining_distortion[step],
            remaining_lod_mean[step],
            remaining_used_lod[step],
            num_hits[step]);

        // float max_weight = 0.0f;

#if MAX_BOUNCES > 0
        for (int ki = 0; ki < TILE_SIZE; ki++)
            for (int kj = 0; kj < TILE_SIZE; kj++) {
                int k = ki * TILE_SIZE + kj;
                int pixel_id = ray_id + ki * params.image_width + kj;
                int c = 0;
                incident_radiance[step][k] = output_rgb[step][k];

                // * Multiply by the BRDF of the previous step
                if (step > 0) {
                    int prev_c = 0;
                    output_rgb[step][k] =
                        output_rgb[step][k] *
                        output_throughput[step - 1][prev_c][k];
                }

                // * Post-process accumulated normal and roughness
                float3 unnormalized_normal = output_normal[step][c][k];
                float3 effective_normal = normalize(unnormalized_normal);
                float effective_roughness =
                    max(output_roughness[step][c][k],
                        eps_min_roughness); // * For stability avoid exactly 0
                                            // roughness
                float3 effective_F0 = output_f0[step][c][k];

                // * Terminate path if the accumulated normal is invalid
                if (length(unnormalized_normal) <
                    reflection_invalid_normal_threshold) {
                    goto forward_pass_end;
                }

                // * Compute the BRDF for this step
                if (step > 0) {
                    output_throughput[step][c][k] *=
                        output_throughput[step - 1][c][k];
                }

                // * Compute reflection ray for the following step
                float3 effective_position =
                    origin[k] + output_depth[step][c][k] * direction[k];
                float3 next_direction = sample_cook_torrance(
                    effective_normal,
                    -direction[k],
                    effective_roughness,
                    make_float2(rnd(seed), rnd(seed)));
                output_throughput[step][c][k] *= cook_torrance_weight(
                    effective_normal,
                    -direction[k],
                    next_direction,
                    effective_roughness,
                    effective_F0);
                float3 next_origin = effective_position +
                                     eps_ray_surface_offset * next_direction;
                reflected_origin = next_origin;
                // reflected_origin = effective_position - next_direction *
                // length(effective_position - tile_origin);

                origin[k] = next_origin;
                direction[k] = next_direction;
                tile_origin = next_origin;       // tmp
                tile_direction = next_direction; // tmp

                params.output_ray_origin[pixel_id + num_pixels * step] =
                    origin[k];
                params.output_ray_direction[pixel_id + num_pixels * step] =
                    direction[k];
            }
#endif
    }

forward_pass_end:

    for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
        // todo disable when m_accumualte is false
        for (int s = 0; s < MAX_BOUNCES + 1; s++) {
            float3 accum_rgb = params.accumulated_rgb[ray_id + num_pixels * s];
            float3 accum_normal =
                params.accumulated_normal[ray_id + num_pixels * s];
            float accum_depth =
                params.accumulated_depth[ray_id + num_pixels * s];
            float3 accum_f0 = params.accumulated_f0[ray_id + num_pixels * s];
            float accum_roughness =
                params.accumulated_roughness[ray_id + num_pixels * s];
            if (!*params.grads_enabled) {
                params.accumulated_rgb[ray_id + num_pixels * s] += make_float3(
                    min(output_rgb[s][k].x, CLAMP_MAX_VALUE),
                    min(output_rgb[s][k].y, CLAMP_MAX_VALUE),
                    min(output_rgb[s][k].z, CLAMP_MAX_VALUE));
                params.accumulated_normal[ray_id + num_pixels * s] +=
                    output_normal[s][0][k];
                params.accumulated_depth[ray_id + num_pixels * s] +=
                    output_depth[s][0][k];
                params.accumulated_f0[ray_id + num_pixels * s] +=
                    output_f0[s][0][k];
                params.accumulated_roughness[ray_id + num_pixels * s] +=
                    output_roughness[s][0][k];
            }
            output_rgb[s][k] = (accum_rgb + output_rgb[s][k]) /
                               float(*params.accumulated_sample_count + 1);
            output_normal[s][0][k] = (accum_normal + output_normal[s][0][k]) /
                                     float(
                                         *params.accumulated_sample_count +
                                         1); // todo would fail with clustering
            output_depth[s][0][k] = (accum_depth + output_depth[s][0][k]) /
                                    float(*params.accumulated_sample_count + 1);
            output_f0[s][0][k] = (accum_f0 + output_f0[s][0][k]) /
                                 float(*params.accumulated_sample_count + 1);
            output_roughness[s][0][k] =
                (accum_roughness + output_roughness[s][0][k]) /
                float(*params.accumulated_sample_count + 1);
        }

        for (int s = 0; s < MAX_BOUNCES + 1; s++) {
            output_rgb[MAX_BOUNCES + 1][k] += output_rgb[s][k];
        }
    }

    for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
        for (int s = 0; s < MAX_BOUNCES + 2; s++) {
            output_rgb_raw[s][k] = output_rgb[s][k];
        }
    }

// * Write image to framebuffer
#pragma unroll
    for (int ki = 0; ki < TILE_SIZE; ki++)
        for (int kj = 0; kj < TILE_SIZE; kj++) {
            int tile_id = ki * TILE_SIZE + kj;

            for (int s = 0; s < *params.num_bounces + 1; s++) {
                int max_c = 0;
                int pixel_id =
                    ray_id + ki * params.image_width + kj + num_pixels * s;
                params.output_rgb[pixel_id] = output_rgb[s][tile_id];
                params.output_t[pixel_id] = output_t[s][tile_id];
                if (!*params.grads_enabled) {
                    params.output_incident_radiance[pixel_id] =
                        incident_radiance[s][tile_id];
                    params.output_normal[pixel_id] =
                        output_normal[s][max_c][tile_id];
                    params.output_position[pixel_id] =
                        output_position[s][max_c][tile_id];
                    params.output_depth[pixel_id] =
                        output_depth[s][max_c][tile_id];
                    params.output_f0[pixel_id] = output_f0[s][max_c][tile_id];
                    params.output_roughness[pixel_id] =
                        output_roughness[s][max_c][tile_id];
#if MAX_BOUNCES > 0
                    params.output_brdf[pixel_id] =
                        output_throughput[s][max_c][tile_id];
#endif
                }
            }

            // Write the final pass
            int pixel_id = ray_id + ki * params.image_width + kj +
                           num_pixels * (MAX_BOUNCES + 1);
            params.output_rgb[pixel_id] = output_rgb[MAX_BOUNCES + 1][tile_id];
        }

    // * Load target images
    float3 target_rgb[TILE_SIZE * TILE_SIZE];
    float3 target_diffuse[TILE_SIZE * TILE_SIZE];
    float3 target_glossy[TILE_SIZE * TILE_SIZE];
    float3 target_position[TILE_SIZE * TILE_SIZE];
    float target_depth[TILE_SIZE * TILE_SIZE];
    float3 target_normal[TILE_SIZE * TILE_SIZE];
    float3 target_f0[TILE_SIZE * TILE_SIZE];
    float target_roughness[TILE_SIZE * TILE_SIZE];

    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_rgb[i * TILE_SIZE + j] =
                params.target_rgb[ray_id + i * params.image_width + j];
        }
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_position[i * TILE_SIZE + j] =
                params.target_position[ray_id + i * params.image_width + j];
        }
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_depth[i * TILE_SIZE + j] =
                params.target_depth[ray_id + i * params.image_width + j];
        }
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_normal[i * TILE_SIZE + j] =
                params.target_normal[ray_id + i * params.image_width + j];
        }
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_f0[i * TILE_SIZE + j] =
                params.target_f0[ray_id + i * params.image_width + j];
        }
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_roughness[i * TILE_SIZE + j] =
                params.target_roughness[ray_id + i * params.image_width + j];
        }
#if MAX_BOUNCES > 0
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_diffuse[i * TILE_SIZE + j] =
                params.target_diffuse[ray_id + i * params.image_width + j];
        }
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_glossy[i * TILE_SIZE + j] =
                params.target_glossy[ray_id + i * params.image_width + j];
        }
#endif

    // last bounce brdf is not used, so these go to MAX_BOUNCES and not
    // MAX_BOUNCES + 1 but start at ray 0
    float3 dL_dthroughput_all_subsequent_steps[TILE_SIZE * TILE_SIZE];
    fill_array(
        dL_dthroughput_all_subsequent_steps,
        TILE_SIZE * TILE_SIZE,
        make_float3(0.0f, 0.0f, 0.0f));
    float3 dL_dthroughput_out[TILE_SIZE * TILE_SIZE];
    fill_array(
        dL_dthroughput_out,
        TILE_SIZE * TILE_SIZE,
        make_float3(0.0f, 0.0f, 0.0f));

    // total for all steps after the current one
    // float3 dL_dsurface_brdf[MAX_BOUNCES][TILE_SIZE*TILE_SIZE]; // last bounce
    // brdf is not used

    float3 dL_dray_origin_next_step = make_float3(0.0f, 0.0f, 0.0f);
    float3 dL_dray_direction_next_step = make_float3(0.0f, 0.0f, 0.0f);

// * Run backward pass
#pragma unroll
    for (int step = total_effective_steps - 1; step >= 0; step--) {
        // * Compute error (difference between prediction and target)
        float3 error[TILE_SIZE * TILE_SIZE];
        for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
            if (step == 0) {
#if MAX_BOUNCES == 0
                error[k] = output_rgb[0][k] - target_rgb[k];
#else
                error[k] = output_rgb[0][k] -
                           target_diffuse[k]; // + (output_rgb[MAX_BOUNCES +
                                              // 1][k] - target_rgb[k]);
#endif
            } else {
                float3 output_glossy = make_float3(0.0f, 0.0f, 0.0f);
                for (int j = 1; j < MAX_BOUNCES + 1; j++) {
                    output_glossy += output_rgb[j][k];
                }
                error[k] =
                    output_glossy -
                    target_glossy[k]; // + (output_rgb[MAX_BOUNCES + 1][k] -
                                      // target_rgb[k]); //
                                      // todo fix densification score weighting,
                                      // not a problem in our config but might
                                      // cause issues for other people if they
                                      // densify longer

                // error[k] = output_rgb[step][k] - target_glossy[k];
            }

            if (step == total_effective_steps - 1) {
                // float3 dL_doutput_roughness = make_float3(0.0f, 0.0f, 0.0f);

                // these flow from dL_dray_origin, dL_dray_direction
                // float3 dL_doutput_normal = make_float3(0.0f, 0.0f, 0.0f);
                // float3 dL_doutput_position = make_float3(0.0f, 0.0f, 0.0f);
            } else {
                float2 lut_values = output_lut_values[step][0][k];
                float3 brdf_of_step =
                    lut_values.x * output_f0[step][0][k] * lut_values.y;

                // cook_torrance_weight(effective_normal, -direction[k],
                // next_direction, effective_roughness, effective_F0);
            }

            // todo debug optimizing from 0 init in a separate file
            // float3 unnormalized_normal = output_normal[step][0][k];
        }

        // Reset to 0 since backward pass increments these
        dL_dray_origin_next_step = make_float3(0.0f, 0.0f, 0.0f);
        dL_dray_direction_next_step = make_float3(0.0f, 0.0f, 0.0f);

        if (num_hits[step] > 0) {

#if ROUGHNESS_DOWNWEIGHT_GRAD == true
            float roughness_downweighting = powf(
                1.0f - output_roughness[max(step - 1, 0)][0][0],
                ROUGHNESS_DOWNWEIGHT_GRAD_POWER);
#else
            float roughness_downweighting = 1.0f;
#endif
            float extra_bounce_weight =
                powf(EXTRA_BOUNCE_WEIGHT, float(max(step - 1, 0)));
            float loss_modulation =
                roughness_downweighting * extra_bounce_weight;

            backward_pass(
                step,
                tile_origin,    ///!!!!!!!!!!!!!!!!!!!!! need to update for each
                                /// ray
                tile_direction, ///!!!!!!!!!!!!!!!!!!!!! need to update for each
                                /// ray
                initial_lod[step],
                lod_by_distance[step],
                ray_id,
                output_rgb_raw[step],
                output_rgb[step],
                output_rgb[MAX_BOUNCES + 1],
                output_t[step],
                //
                output_position[step][0],
                output_depth[step][0],
                output_normal[step][0],
                output_f0[step][0],
                output_roughness[step][0],
                output_distortion[step][0],
                //
                remaining_rgb[step][0],
                remaining_position[step][0],
                remaining_depth[step][0],
                remaining_normal[step][0],
                remaining_f0[step][0],
                remaining_roughness[step][0],
                remaining_distortion[step][0],
                num_hits[step],
                output_throughput[max(step - 1, 0)][0],
                dL_dthroughput_out, // calling this function adds gradients to
                                    // this, its a mess sry
                //
                target_rgb,
                target_diffuse,
                target_glossy,
                target_position,
                target_depth,
                target_normal,
                target_f0,
                target_roughness,
                error,
                loss_modulation,
                step == 0 ? params.diffuse_loss_weight
                          : params.glossy_loss_weight,
                dL_dray_origin_next_step,
                dL_dray_direction_next_step);

            for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
                float2 lut_values = output_lut_values[step][0][k];
                float3 brdf_of_step =
                    lut_values.x * output_f0[step][0][k] * lut_values.y;
                dL_dthroughput_all_subsequent_steps[k] += dL_dthroughput_out[k];
            }
        }

        // todo accumulate dL_dbrdf = sum of dL_dthroughput of next steps
    }

    // Compute grads for brdf
    for (int step = 0; step < MAX_BOUNCES + 1; step++) {
    }
    params.random_seeds[ray_id] = seed;
}
