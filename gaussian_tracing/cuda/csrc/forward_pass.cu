#include <math_constants.h>

#pragma inline
__device__ void forward_pass(
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

    uint32_t full_T_uint[TILE_SIZE * TILE_SIZE];
    fill_array(full_T_uint, TILE_SIZE * TILE_SIZE, __float_as_uint(1.0f));

    // * Traverse BVH
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
            near_plane, // tmin
            far_plane,
            0.0f, // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            0, // SBT offset
            0, // SBT stride
            0,
            uint_initial_lod,
            uint_lod_by_distance,
            step_uint, // step, replaces slab idx
            full_T_uint[0],
            reflected_origin_x,
            reflected_origin_y,
            reflected_origin_z);
        for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
            output_t[k].y = __uint_as_float(full_T_uint[k]);
        }
    }

    // * Initialize registers holding the BUFFER_SIZE nearest gaussians
    register float distances[BUFFER_SIZE];
    register unsigned int idxes[BUFFER_SIZE];

    const uint3 idx = optixGetLaunchIndex();
    for (int iteration = 0; iteration < MAX_ITERATIONS && tmin < 99.9f;
         iteration++) {
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

        // * Integrate the values
        tmin = max(tmin, distances[0]);

#pragma unroll
        for (int i = 0; i < BUFFER_SIZE; i++) {
            float distance = distances[i];
            tmin = max(distance, tmin); // todo: fails if its not a max, but the
                                        // values should already be sorted??

            if (distance < 99.9f) {
                uint32_t gaussian_id = params.all_gaussian_ids[idxes[i]];
                float gaussval = params.all_gaussvals[idxes[i]];
                float alpha = params.all_alphas[idxes[i]];
                float3 gaussian_rgb = params.gaussian_rgb[gaussian_id];
                float3 gaussian_position =
                    params.gaussian_position[gaussian_id];
                float3 gaussian_normal = params.gaussian_normal[gaussian_id];
                float3 gaussian_f0 =
                    clipped_relu_act(params.gaussian_f0[gaussian_id]);
                float gaussian_roughness =
                    clipped_relu_act(params.gaussian_roughness[gaussian_id]);
                num_hits++; //! was incorrect for tiling, review

                int c = 0;
                int new_idx;
                for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
                    float alpha =
                        params.all_alphas[TILE_SIZE * TILE_SIZE * idxes[i] + k];
                    float gaussval = params.all_gaussvals
                                         [TILE_SIZE * TILE_SIZE * idxes[i] + k];
                    float3 local_hit =
                        params.all_local_hits
                            [TILE_SIZE * TILE_SIZE * idxes[i] + k];

                    float next_T = output_t[k].x * (1.0f - alpha);
                    float weight = output_t[k].x - next_T;
                    output_rgb[k] += gaussian_rgb * weight;

                    output_depth[k][c] += distance * weight;
                    output_normal[k][c] += gaussian_normal * weight;
                    output_f0[k][c] += gaussian_f0 * weight;
                    output_roughness[k][c] += gaussian_roughness * weight;

                    output_t[k].x = next_T;
                    cluster_weights[c][k] += weight;

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

                    if (output_t[0].x < T_THRESHOLD) {
                        goto finished_integration; // todo review a few options
                                                   // instead of this (continue
                                                   // etc)
                    }
                }
            }
        }
    }
finished_integration:

    params.prev_hit_per_pixel[ray_id] = 999999999u;

    for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
        float normalization = max((1.0f - output_t[k].x), 1e-12);

        remaining_rgb[k][0] = output_rgb[k] / normalization;
        remaining_position[k][0] = output_position[k][0] / normalization;
        remaining_depth[k][0] = output_depth[k][0] / normalization;
        remaining_normal[k][0] = output_normal[k][0] / normalization;
        remaining_f0[k][0] = output_f0[k][0] / normalization;
        remaining_roughness[k][0] = output_roughness[k][0] / normalization;
    }

    for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
        float remaining_T = output_t[k].x - output_t[k].y;
        output_rgb[k] = output_rgb[k] + remaining_T * remaining_rgb[k][0];
        output_position[k][0] =
            output_position[k][0] + remaining_T * remaining_position[k][0];
        output_depth[k][0] =
            output_depth[k][0] + remaining_T * remaining_depth[k][0];
        output_normal[k][0] =
            output_normal[k][0] + remaining_T * remaining_normal[k][0];
        output_f0[k][0] = output_f0[k][0] + remaining_T * remaining_f0[k][0];
        output_roughness[k][0] =
            output_roughness[k][0] + remaining_T * remaining_roughness[k][0];
    }
}
