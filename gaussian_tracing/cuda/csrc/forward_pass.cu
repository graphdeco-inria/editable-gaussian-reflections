#include <math_constants.h>

#pragma inline
__device__ void froward_pass(
    const int step,
    //
    float initial_lod,
    float lod_by_distance,
    //
    const uint32_t &ray_id,
    float3 tile_origin,
    float3 reflected_origin,
    float3 tile_direction,
    float3 (&origin)[TILE_SIZE * TILE_SIZE],
    float3 (&direction)[TILE_SIZE * TILE_SIZE],
    //
    float (&cluster_weights)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    //
    float3 (&output_rgb)[TILE_SIZE * TILE_SIZE],
    float2 (&output_t)[TILE_SIZE * TILE_SIZE],
    //
    float3 (&output_position)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float (&output_depth)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float3 (&output_normal)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float3 (&output_f0)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float (&output_roughness)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float (&output_distortion)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float (&output_lod_mean)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float (&output_lod_scale)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float (&output_ray_lod)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    //
    float3 (&remaining_rgb)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float3 (&remaining_position)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float (&remaining_depth)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float3 (&remaining_normal)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float3 (&remaining_f0)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float (&remaining_roughness)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float (&remaining_distortion)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float (&remaining_lod_mean)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    float (&remaining_ray_lod)[NUM_CLUSTERS][TILE_SIZE * TILE_SIZE],
    //
    int &num_hits) {
    float near_plane = *params.camera_znear; // like 0.5 in kitchen scene
    float far_plane = *params.camera_zfar;   // like 2.5 in kitchen scene
    // float near_plane = 0.0f;
    // float far_plane = 999.9f;
    if (step != 0) {
        near_plane = 0.0f;
    }

    float tmin = near_plane;
    float endpoint = -1.0f;

#if STORAGE_MODE == PER_PIXEL_LINKED_LIST
    uint32_t full_T_uint[TILE_SIZE * TILE_SIZE];
    fill_array(full_T_uint, TILE_SIZE * TILE_SIZE, __float_as_uint(1.0f));

    // * Traverse BVH
    for (uint32_t i = 0; i < NUM_SLABS; i++) {
        float slab_tmin = near_plane + (far_plane - near_plane) / NUM_SLABS * i;
        float slab_tmax =
            near_plane + (far_plane - near_plane) / NUM_SLABS * (i + 1);

        if (output_t[0].y > params.transmittance_threshold) {
            uint32_t uint_initial_lod = __float_as_uint(initial_lod);
            uint32_t uint_lod_by_distance = __float_as_uint(lod_by_distance);
            uint32_t step_uint = (uint32_t)step;
            uint32_t reflected_origin_x = __float_as_uint(reflected_origin.x);
            uint32_t reflected_origin_y = __float_as_uint(reflected_origin.y);
            uint32_t reflected_origin_z = __float_as_uint(reflected_origin.z);
            optixTraverse(
                params.handle,
                tile_origin,
                tile_direction,
                slab_tmin, // tmin
                slab_tmax,
                0.0f, // rayTime
                OptixVisibilityMask(1),
#if USE_POLYCAGE == true
                OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
#else
                OPTIX_RAY_FLAG_NONE,
#endif
                0, // SBT offset
                0, // SBT stride
                0,
                uint_initial_lod,
                uint_lod_by_distance,
                step_uint, // step, replaces slab idx
// i, // slab idx
#if TILE_SIZE == 1
                full_T_uint[0],
                reflected_origin_x,
                reflected_origin_y,
                reflected_origin_z
#elif TILE_SIZE == 2
                printf("dead code\n");
                full_T_uint[0], full_T_uint[1], full_T_uint[2], full_T_uint[3]
#elif TILE_SIZE == 3
                printf("dead code\n");
                full_T_uint[0],
                full_T_uint[1],
                full_T_uint[2],
                full_T_uint[3],
                full_T_uint[4],
                full_T_uint[5],
                full_T_uint[6],
                full_T_uint[7],
                full_T_uint[8]
#else
                           // raise an error
                printf("TILE_SIZE not supported\n");
#endif

            );
            for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
                output_t[k].y = __uint_as_float(full_T_uint[k]);
            }
        }
    }
#endif

// * Initialize registers holding the BUFFER_SIZE nearest gaussians
#if STORAGE_MODE == PER_PIXEL_LINKED_LIST
    register float distances[BUFFER_SIZE];
    register unsigned int idxes[BUFFER_SIZE];
#else
    register unsigned int idxes[BUFFER_SIZE];
    register unsigned int floats[BUFFER_SIZE];
    register unsigned int gaussian_ids[BUFFER_SIZE];
#endif

    const uint3 idx = optixGetLaunchIndex();

#if USE_CLUSTERING == true
    int current_cluster = 0;
#endif

    for (int iteration = 0; iteration < MAX_ITERATIONS && tmin < 99.9f;
         iteration++) {
#if STORAGE_MODE == PER_PIXEL_LINKED_LIST
        fill_array(distances, BUFFER_SIZE, 999.9f);
        fill_array(idxes, BUFFER_SIZE, 999999999u);

        // * Find the BUFFER_SIZE nearest gaussians behind the last batch
        uint32_t hit_idx = params.prev_hit_per_pixel[ray_id];
        while (hit_idx != 999999999u) {
            float curr_distance = params.all_distances[hit_idx];
            uint32_t prev_hit = params.all_prev_hits[hit_idx];

            if (curr_distance > tmin &&
                curr_distance < distances[BUFFER_SIZE - 1]) {
                distances[BUFFER_SIZE - 1] = curr_distance;
                idxes[BUFFER_SIZE - 1] = hit_idx;
            }
#pragma unroll
            for (int i = BUFFER_SIZE - 1; i > 0; i--) {
                if (distances[i] < distances[i - 1]) {
                    // swap i with i-1
                    float tmp_dist = distances[i];
                    int tmp_idx = idxes[i];
                    distances[i] = distances[i - 1];
                    idxes[i] = idxes[i - 1];
                    distances[i - 1] = tmp_dist;
                    idxes[i - 1] = tmp_idx;
                }
            }
            hit_idx = prev_hit;
        }
#else
        for (int ki = 0; ki < TILE_SIZE; ki++)
            for (int kj = 0; kj < TILE_SIZE; kj++) {
                params.full_T[ray_id + ki * params.image_width + kj] =
                    1.0f; // todo this was a tmp solution, doesn't work with
                          // multibounce
            }

        fill_array(floats, BUFFER_SIZE, packFloats(999.9f, 999.9f));
        fill_array(idxes, BUFFER_SIZE, 999999999u);
        fill_array(gaussian_ids, BUFFER_SIZE, NULL_GAUSSIAN_ID);
        optixTraverse(
            params.handle,
            tile_origin,
            tile_direction,
            tmin,     //
            100000.0, // tmax
            0.0f,     // rayTime
            OptixVisibilityMask(1),
#if USE_POLYCAGE == true
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
#else
            OPTIX_RAY_FLAG_NONE,
#endif
            0, // SBT offset
            0, // SBT stride
            0,
            floats[0],
            floats[1],
            floats[2],
            floats[3],
            floats[4],
            floats[5],
            floats[6],
            floats[7],
            floats[8],
            floats[9],
            floats[10],
            floats[11],
            floats[12],
            floats[13],
            floats[14],
            floats[15],
            gaussian_ids[0],
            gaussian_ids[1],
            gaussian_ids[2],
            gaussian_ids[3],
            gaussian_ids[4],
            gaussian_ids[5],
            gaussian_ids[6],
            gaussian_ids[7],
            gaussian_ids[8],
            gaussian_ids[9],
            gaussian_ids[10],
            gaussian_ids[11],
            gaussian_ids[12],
            gaussian_ids[13],
            gaussian_ids[14],
            gaussian_ids[15]);

        for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
            output_t[k].y = params.full_T[ray_id];
        }
#endif

// * Integrate the values
#if STORAGE_MODE == PER_PIXEL_LINKED_LIST
        tmin = max(tmin, distances[0]);
#elif STORAGE_MODE == NO_STORAGE
        tmin = max(tmin, unpackDistance(floats[0]));
#endif

#pragma unroll
        for (int i = 0; i < BUFFER_SIZE; i++) {
#if STORAGE_MODE == PER_PIXEL_LINKED_LIST
            float distance = distances[i];
#elif STORAGE_MODE == NO_STORAGE
            float distance = unpackDistance(floats[i]);
#endif
            tmin = max(distance, tmin); // todo: fails if its not a max, but the
                                        // values should already be sorted??

            if (distance < 99.9f) {

#if SAVE_HIT_STATS == true
                atomicAdd(&params.num_hits_per_pixel[ray_id], 1);
#endif

#if STORAGE_MODE == PER_PIXEL_LINKED_LIST
                uint32_t gaussian_id = params.all_gaussian_ids[idxes[i]];
#if RECOMPUTE_ALPHA_IN_FORWARD_PASS == false && TILE_SIZE == 1
                float gaussval = params.all_gaussvals[idxes[i]];
                float alpha = params.all_alphas[idxes[i]];
#endif
#else
                uint32_t gaussian_id = gaussian_ids[i];
#endif

                float3 gaussian_rgb = READ_RGB(gaussian_id);
#if ATTACH_POSITION == true
                float3 gaussian_position =
                    params.gaussian_position[gaussian_id];
#endif
#if ATTACH_NORMALS == true
                float3 gaussian_normal = params.gaussian_normal[gaussian_id];
#endif
#if ATTACH_F0 == true
                float3 gaussian_f0 = READ_F0(gaussian_id);
#endif
#if ATTACH_ROUGHNESS == true
                float gaussian_roughness = READ_ROUGHNESS(gaussian_id);
#endif

#if RECOMPUTE_ALPHA_IN_FORWARD_PASS == true
                float opacity = READ_OPACITY(gaussian_id);
                float3 scaling = READ_SCALE(gaussian_id);
                float3 mean = READ_MEAN(gaussian_id);
                const float4 *world_to_local =
                    optixGetInstanceInverseTransformFromHandle(
                        optixGetInstanceTraversableFromIAS(
                            params.handle, gaussian_id));
                float exp_power = params.exp_power;
#endif

                num_hits++; //! was incorrect for tiling, review

#if USE_CLUSTERING == true
#if TILE_SIZE != 1
                printf("TILING DOESN'T WORK WITH CLUSTERING YET!\n");
#endif
                float half_chord_length =
                    params.all_half_chord_lengths[idxes[i]];
                if (ray_id == 777) {
                    printf(
                        "endpoint: %f, distance: %f, half_chord_length: %f\n",
                        endpoint,
                        distance,
                        half_chord_length);
                }
                // if ((endpoint > 0.0f) && (distance > endpoint) && (distance -
                // endpoint) <= half_chord_length) {
                //     if (current_cluster < NUM_CLUSTERS - 1) {
                //         current_cluster++; // todo starting for the secondary
                //         ray, ignore the first cluster
                //     }
                // }
                endpoint = max(distance + half_chord_length, endpoint);
                int c = current_cluster;
#else
                int c = 0;
#endif

                int new_idx;
                for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
                    // #if TILE_SIZE > 1 // this is actually slower than just
                    // stopping at the same point for all subpixels
                    //     if (output_t[k].x < params.transmittance_threshold) {
                    //         continue;
                    //     }
                    // #endif

#if RECOMPUTE_ALPHA_IN_FORWARD_PASS == true
#if JITTER == true
                    printf("TILING DOESN'T WORK WITH JITTER YET!\n");
#endif

                    float3 local_origin = make_float3(
                        dot(world_to_local[0], make_float4(origin[k], 1.0)),
                        dot(world_to_local[1], make_float4(origin[k], 1.0)),
                        dot(world_to_local[2], make_float4(origin[k], 1.0)));
                    float3 local_direction = make_float3(
                        dot(make_float3(world_to_local[0]), direction[k]),
                        dot(make_float3(world_to_local[1]), direction[k]),
                        dot(make_float3(world_to_local[2]), direction[k]));

                    float scaling_factor = compute_scaling_factor(
                        opacity, params.alpha_threshold, exp_power);
                    if (BBOX_PADDING > 0.0f) {
                        float3 padded_scaling = scaling * scaling_factor;
                        local_origin = local_origin *
                                       (BBOX_PADDING + padded_scaling) /
                                       padded_scaling;
                        local_direction = local_direction *
                                          (BBOX_PADDING + padded_scaling) /
                                          padded_scaling;
                    }
                    float norm = length(local_direction);
                    local_direction /= norm;
                    auto local_hit_distance_along_ray =
                        dot(-local_origin, local_direction);
                    float3 local_hit_unscaled =
                        local_origin +
                        local_hit_distance_along_ray * local_direction;
                    float sq_dist = dot(local_hit_unscaled, local_hit_unscaled);
                    float gaussval = 0.0f;
                    float alpha = 0.0f;
                    if (sq_dist <= 1.0f) {
                        float3 local_hit = local_hit_unscaled * scaling_factor;
                        gaussval = eval_gaussian(local_hit, exp_power);
                        alpha = compute_alpha(
                            gaussval, opacity, params.alpha_threshold);
                    }
#else
                    float alpha =
                        params.all_alphas[TILE_SIZE * TILE_SIZE * idxes[i] + k];
                    float gaussval = params.all_gaussvals
                                         [TILE_SIZE * TILE_SIZE * idxes[i] + k];
                    float3 local_hit =
                        params.all_local_hits
                            [TILE_SIZE * TILE_SIZE * idxes[i] + k];
#endif

                    float next_T = output_t[k].x * (1.0f - alpha);
                    float weight = output_t[k].x - next_T;
                    output_rgb[k] += gaussian_rgb * weight;

#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
                    // float3 world_hit = tile_origin + tile_direction *
                    // distance; float3 world_hit = READ_MEAN(gaussian_id);
                    // output_position[k][c] += world_hit * weight;
                    // float3 world_hit = READ_MEAN(gaussian_id);
                    output_depth[k][c] += distance * weight;
#else
#if ATTACH_POSITION == true
                    output_position[k][c] += gaussian_position * weight;
#endif
#endif
#if ATTACH_NORMALS == true
                    output_normal[k][c] += gaussian_normal * weight;
#endif
#if ATTACH_F0 == true
                    output_f0[k][c] += gaussian_f0 * weight;
#endif
#if ATTACH_ROUGHNESS == true
                    output_roughness[k][c] += gaussian_roughness * weight;
#endif

                    output_t[k].x = next_T;
                    cluster_weights[c][k] += weight;

#if BACKWARDS_PASS == true
                    // * Store data required for the backward pass
                    if (*params.grads_enabled) {
                        if (k == 0) {
                            new_idx =
                                atomicAdd(params.total_hits_for_backprop, 1);
                            params.all_gaussian_ids_for_backprop[new_idx] =
                                gaussian_id;
                            params.all_prev_hits_for_backprop[new_idx] =
                                params.prev_hit_per_pixel_for_backprop[ray_id];
                            params.prev_hit_per_pixel_for_backprop[ray_id] =
                                new_idx;
                        }
                        params.all_distances_for_backprop
                            [TILE_SIZE * TILE_SIZE * new_idx + k] = distance;
                        params.all_local_hits_for_backprop
                            [TILE_SIZE * TILE_SIZE * new_idx + k] = local_hit;
                        params.all_alphas_for_backprop
                            [TILE_SIZE * TILE_SIZE * new_idx + k] = alpha;
                        params.all_Ts_for_backprop
                            [TILE_SIZE * TILE_SIZE * new_idx + k] =
                            output_t[k].x;
                        params.all_gaussvals_for_backprop
                            [TILE_SIZE * TILE_SIZE * new_idx + k] = gaussval;
                    }
#endif

                    if (output_t[0].x < T_THRESHOLD) {
                        goto finished_integration; // todo review a few options
                                                   // instead of this (continue
                                                   // etc)
                    }

                    // #if REMAINING_COLOR_ESTIMATION != RENDER_BUT_DETACH
                    //     printf("transmittance_threshold %f",
                    //     params.transmittance_threshold); if
                    //     (params.transmittance_threshold > 0.0f) {
                    //         float all_done = true;
                    //         for (int k = 0; k < TILE_SIZE*TILE_SIZE; k++) {
                    //             if (output_t[k].x >=
                    //             params.transmittance_threshold) {
                    //                 all_done = false;
                    //             }
                    //         }
                    //         if (all_done) {
                    //             goto finished_integration;
                    //         }
                    //     }
                    // #endif
                }
            }
        }
    }
finished_integration:

    // #if USE_CLUSTERING == true
    //     for (int k = 0; k < TILE_SIZE*TILE_SIZE; k++) {
    //         for (int c = 0; c < NUM_CLUSTERS; c++) {
    //             float cw = (1.0 - output_t[k].x) / (cluster_weights[c][k] +
    //             1e-12); #if ATTACH_POSITION == true
    //                 output_position[k][c] *= cw;
    //             #endif
    //             #if ATTACH_NORMALS == true
    //                 output_normal[k][c] *= cw; // todo review
    //             #endif
    //             #if ATTACH_F0 == true
    //                 output_f0[k][c] *= cw;
    //             #endif
    //             #if ATTACH_ROUGHNESS == true
    //                 output_roughness[k][c] *= cw;
    //             #endif
    //             #if SAVE_LOD_IMAGES == true
    //                 output_lod_mean[k][c] *= cw;
    //                 output_ray_lod[k][c] *= cw;
    //             #endif
    //         }
    //     }
    // #endif

#if LOG_ALL_HITS == true
    params.prev_hit_per_pixel[ray_id] = 999999999u;
#endif

#if REMAINING_COLOR_ESTIMATION == ASSUME_SAME_AS_FORGROUND

    for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
        float normalization = max((1.0f - output_t[k].x), 1e-12);

        remaining_rgb[k][0] = output_rgb[k] / normalization;
#if ATTACH_POSITION == true
        remaining_position[k][0] = output_position[k][0] / normalization;
#endif
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
        remaining_depth[k][0] = output_depth[k][0] / normalization;
#endif
#if ATTACH_NORMALS == true
        remaining_normal[k][0] = output_normal[k][0] / normalization;
#endif
#if ATTACH_F0 == true
        remaining_f0[k][0] = output_f0[k][0] / normalization;
#endif
#if ATTACH_ROUGHNESS == true
        remaining_roughness[k][0] = output_roughness[k][0] / normalization;
#endif
#if SAVE_LOD_IMAGES == true
        remaining_lod_mean[k][0] = output_lod_mean[k][0] / normalization;
        remaining_ray_lod[k][0] = output_ray_lod[k][0] / normalization;
#endif
    }
#elif REMAINING_COLOR_ESTIMATION == IGNORE_OCCLUSION
    float alpha_sum = 0.0f;

    float3 average_rgb = make_float3(0.0f, 0.0f, 0.0f);
#if ATTACH_POSITION == true || POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
    float3 average_position = make_float3(0.0f, 0.0f, 0.0f);
#endif
#if ATTACH_NORMALS == true
    float3 average_normal = make_float3(0.0f, 0.0f, 0.0f);
#endif
#if ATTACH_F0 == true
    float3 average_f0 = make_float3(0.0f, 0.0f, 0.0f);
#endif
#if ATTACH_ROUGHNESS == true
    float average_roughness = 0.0f;
#endif
#if SAVE_LOD_IMAGES == true
    float average_lod_mean = 0.0f;
    float average_ray_lod = 0.0f;
#endif
    uint32_t hit_idx = params.prev_hit_per_pixel[ray_id];
    while (hit_idx != 999999999) {
        uint32_t prev_hit = params.all_prev_hits[hit_idx];
        float dist = params.all_distances[hit_idx];
        if (dist > tmin) {
            float alpha = params.all_alphas[hit_idx];
            unsigned int gaussian_id = params.all_gaussian_ids[hit_idx];
            float3 color = READ_RGB(gaussian_id);
            average_rgb += color * alpha;
#if ATTACH_POSITION == true
            float3 position = params.gaussian_position[gaussian_id];
            average_position += position * alpha;
#endif
#if ATTACH_NORMALS == true
            float3 normal = params.gaussian_normal[gaussian_id];
            average_normal += normal * alpha;
#endif
#if ATTACH_F0 == true
            float3 f0 = READ_F0(gaussian_id);
            average_f0 += f0 * alpha;
#endif
#if ATTACH_ROUGHNESS == true
            float roughness = READ_ROUGHNESS(gaussian_id);
            average_roughness += roughness * alpha;
#endif
            alpha_sum += alpha;
        }
        hit_idx = prev_hit;
    }
    average_rgb /= max(alpha_sum, 1e-8);
#if ATTACH_POSITION == true || POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
    average_position /= max(alpha_sum, 1e-8);
#endif
#if ATTACH_NORMALS == true
    average_normal /= max(alpha_sum, 1e-8);
#endif
#if ATTACH_F0 == true
    average_f0 /= max(alpha_sum, 1e-8);
#endif
#if ATTACH_ROUGHNESS == true
    average_roughness /= max(alpha_sum, 1e-8);
#endif
    for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
        remaining_rgb[k][0][0] = average_rgb;
    }
#elif REMAINING_COLOR_ESTIMATION == STOCHASTIC_SAMPLING
    auto seed = params.random_seeds[ray_id];

#define NUM_STOCH_SAMPLES 16

    register unsigned int gaussian_id_buffer[NUM_STOCH_SAMPLES];
    fill_array(gaussian_id_buffer, NUM_STOCH_SAMPLES, NULL_GAUSSIAN_ID);
    register float distances_buffer[NUM_STOCH_SAMPLES];
    fill_array(distances_buffer, NUM_STOCH_SAMPLES, 999.9f);

    uint32_t hit_idx = params.prev_hit_per_pixel[ray_id];
    while (hit_idx != 999999999) {
        uint32_t prev_hit = params.all_prev_hits[hit_idx];
        for (int i = 0; i < NUM_STOCH_SAMPLES; i++) {
            float dist = params.all_distances[hit_idx];
            if (dist > tmin && dist < distances_buffer[i]) {
                float u = rnd(seed);
                if (u < params.all_alphas[hit_idx]) {
                    gaussian_id_buffer[i] = params.all_gaussian_ids[hit_idx];
                    distances_buffer[i] = dist;
                }
            }
        }
        hit_idx = prev_hit;
    }

    for (int i = 0; i < NUM_STOCH_SAMPLES; i++) {
        uint32_t gaussian_id = (gaussian_id_buffer[i]);
        float distance = (distances_buffer[i]);
        if (distance >= 89.9f) {
            continue;
        }
        float3 sample_color = READ_RGB(gaussian_id) / NUM_STOCH_SAMPLES;
#if ATTACH_POSITION == true || POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
        float3 sample_position =
            params.gaussian_position[gaussian_id] / NUM_STOCH_SAMPLES;
#endif
#if ATTACH_NORMALS == true
        float3 sample_normal =
            params.gaussian_normal[gaussian_id] / NUM_STOCH_SAMPLES;
#endif
#if ATTACH_F0 == true
        float3 sample_f0 = READ_F0(gaussian_id) / NUM_STOCH_SAMPLES;
#endif
#if ATTACH_ROUGHNESS == true
        float sample_roughness =
            READ_ROUGHNESS(gaussian_id) / NUM_STOCH_SAMPLES;
#endif

        for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
            remaining_rgb[k][0] += sample_color;
#if ATTACH_POSITION == true || POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
            remaining_position[k][0] += sample_position;
#endif
#if ATTACH_NORMALS == true
            remaining_normal[k][0] += sample_normal;
#endif
#if ATTACH_F0 == true
            remaining_f0[k][0] += sample_f0;
#endif
#if ATTACH_ROUGHNESS == true
            remaining_roughness[k][0] += sample_roughness;
#endif
#if SAVE_LOD_IMAGES == true
            printf("unimplemented!\n");
#endif
        }
    }

    params.random_seeds[ray_id] = seed;
#endif

#if REMAINING_COLOR_ESTIMATION != NO_ESTIMATION
    for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
        float remaining_T = output_t[k].x - output_t[k].y;
        output_rgb[k] = output_rgb[k] + remaining_T * remaining_rgb[k][0];
#if ATTACH_POSITION == true
        output_position[k][0] =
            output_position[k][0] + remaining_T * remaining_position[k][0];
#endif
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
        output_depth[k][0] =
            output_depth[k][0] + remaining_T * remaining_depth[k][0];
#endif
#if ATTACH_NORMALS == true
        output_normal[k][0] =
            output_normal[k][0] + remaining_T * remaining_normal[k][0];
#endif
#if ATTACH_F0 == true
        output_f0[k][0] = output_f0[k][0] + remaining_T * remaining_f0[k][0];
#endif
#if ATTACH_ROUGHNESS == true
        output_roughness[k][0] =
            output_roughness[k][0] + remaining_T * remaining_roughness[k][0];
#endif
#if SAVE_LOD_IMAGES == true
        output_lod_mean[k][0] =
            output_lod_mean[k][0] + remaining_T * remaining_lod_mean[k][0];
        output_ray_lod[k][0] =
            output_ray_lod[k][0] + remaining_T * remaining_ray_lod[k][0];
#endif
    }
#endif
}
