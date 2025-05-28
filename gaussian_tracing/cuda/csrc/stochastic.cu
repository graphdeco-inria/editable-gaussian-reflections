#if LOG_ALL_HITS == true
    // 30 ms baseline

    #if TILE_SIZE > 1
        // uint3 idx2 = optixGetLaunchIndex();
        // uint3 dim2 = optixGetLaunchDimensions();
        // float aspect_ratio = float(dim2.x) / float(dim2.y);
        // float view_size = tan(*params.vertical_fov_radians / 2);

        // register unsigned int gaussian_id_buffer[BUFF_SIZE]; 
        // register float distances_buffer[BUFF_SIZE]; 
        // fill_array(gaussian_id_buffer, BUFF_SIZE, NULL_GAUSSIAN_ID);
        // fill_array(distances_buffer, BUFF_SIZE, 999.9f);

        // for (int i = 0; i < BUFF_SIZE; i++) {
        //         if (curr_distance < distances_buffer[i]) {
        //             float u = rnd(seed);
        //             if (u < curr_alpha) {
        //                 gaussian_id_buffer[i] = curr_gaussian_id;
        //                 distances_buffer[i] = curr_distance;
        //             }
        //         }
        //     }
        //     hit_idx = prev_hit; 
        // }

        // Add their contribution
        // for (int i = 0; i < BUFF_SIZE; i++) {
        //     if (gaussian_id_buffer[i] != NULL_GAUSSIAN_ID) {
        //         output_rgb[0] += READ_RGB(gaussian_id_buffer[i]) / BUFF_SIZE;
        //     }
        // }

        uint32_t hit_idx = params.prev_hit_per_pixel[ray_id];
        while (hit_idx != 999999999) {
            float distance = params.all_distances[hit_idx];
            float alpha = params.all_alphas[hit_idx];
            auto gaussian_id = params.all_gaussian_ids[hit_idx];
            const float4* world_to_local = optixGetInstanceInverseTransformFromHandle(optixGetInstanceTraversableFromIAS(params.handle, gaussian_id));
            
            for (int ki = 0; ki < TILE_SIZE; ki++) { 
                for (int kj = 0; kj < TILE_SIZE; kj++) {
                    float3 world_origin = tile_origin; // assume pinhole
                    float _y = (idx2.y * TILE_SIZE + ki) / (float(dim2.y) * TILE_SIZE); 
                    float _x = (idx2.x * TILE_SIZE + kj) / (float(dim2.x) * TILE_SIZE); 
                    float y = view_size * (1.0f - 2.0f * _y); 
                    float x = aspect_ratio * view_size * (2.0f * _x - 1.0f); 
                    float3 world_direction = normalize(params.camera_rotation_w2c[0] * x + params.camera_rotation_w2c[1] * y - params.camera_rotation_w2c[2]);
                    
                    float3 local_origin = world_origin - READ_MEAN(gaussian_id); // todo pick out from matrix & verify ok
                    float3 local_direction = make_float3(
                        dot(make_float3(world_to_local[0].x, world_to_local[1].x, world_to_local[2].x), world_direction),
                        dot(make_float3(world_to_local[0].y, world_to_local[1].y, world_to_local[2].y), world_direction),
                        dot(make_float3(world_to_local[0].z, world_to_local[1].z, world_to_local[2].z), world_direction)
                    ); // todo verify that it matches the value obtained for the tile direction
                    float norm = length(local_direction);
                    local_direction /= norm;
                    auto local_hit_distance_along_ray = dot(-local_origin, local_direction); 
                    float3 local_hit = local_origin + local_hit_distance_along_ray * local_direction;
                    float sq_dist = dot(local_hit, local_hit);
                    bool hit = sq_dist <= 1.0f;
                    // if (hit) {
                    // }
                    output_rgb[ki * TILE_SIZE + kj] += READ_RGB(gaussian_id) / BUFFER_SIZE;
                    // todo how to compensate for misses!?
                }
            }
            hit_idx = params.all_prev_hits[hit_idx];
        }

        return;
    #endif


    if (false) {
        auto seed = params.random_seeds[ray_id];
        constexpr int BUFF_SIZE = 16*2;

        register unsigned int gaussian_id_buffer[BUFF_SIZE]; 
        register float distances_buffer[BUFF_SIZE]; 
        fill_array(gaussian_id_buffer, BUFF_SIZE, NULL_GAUSSIAN_ID);
        fill_array(distances_buffer, BUFF_SIZE, 999.9f);

        // Sample k gaussians proportionally to their weights
        uint32_t hit_idx = params.prev_hit_per_pixel[ray_id];
        while (hit_idx != 999999999) {
            uint32_t curr_gaussian_id = params.all_gaussian_ids[hit_idx];
            float curr_distance = params.all_distances[hit_idx];
            float curr_alpha = params.all_alphas[hit_idx];
            uint32_t prev_hit = params.all_prev_hits[hit_idx];
            for (int i = 0; i < BUFF_SIZE; i++) {
                if (curr_distance < distances_buffer[i]) {
                    float u = rnd(seed);
                    if (u < curr_alpha) {
                        gaussian_id_buffer[i] = curr_gaussian_id;
                        distances_buffer[i] = curr_distance;
                    }
                }
            }
            hit_idx = prev_hit; 
        }

        // Deduplicate
        for (int i = 0; i < BUFF_SIZE; i++) {
            for (int j = 0; j < i; j++) {
                if (gaussian_id_buffer[i] == gaussian_id_buffer[j]) {
                    gaussian_id_buffer[i] = NULL_GAUSSIAN_ID;
                }
            }
        }

        int num_empty = 0;
        for (int i = 0; i < BUFF_SIZE; i++) {
            if (gaussian_id_buffer[i] == NULL_GAUSSIAN_ID) {
                num_empty++;
            }
        }
        // printf("num_empty: %d\n", num_empty);

        // Compute the exact weight for all gaussians in the buffer
        hit_idx = params.prev_hit_per_pixel[ray_id];
        register float weight_buffer[BUFF_SIZE]; fill_array(weight_buffer, BUFF_SIZE, 1.0f);
        while (hit_idx != 999999999) {
            float distance = params.all_distances[hit_idx];
            float alpha = params.all_alphas[hit_idx];
            auto gaussian_id = params.all_gaussian_ids[hit_idx];
            uint32_t prev_hit = params.all_prev_hits[hit_idx];
            for (int i = 0; i < BUFF_SIZE; i++) {
                if (gaussian_id_buffer[i] != NULL_GAUSSIAN_ID) {
                    if (gaussian_id == gaussian_id_buffer[i]) {
                        weight_buffer[i] *= alpha;
                    } else if (distance < distances_buffer[i]) {
                        weight_buffer[i] *= 1.0f - alpha;
                    }
                }
            }
            hit_idx = prev_hit;
        }


        // Add their contribution
        float weight_sum = 0.0f;
        for (int i = 0; i < BUFF_SIZE; i++) {
            if (gaussian_id_buffer[i] != NULL_GAUSSIAN_ID) {
                output_rgb[0] += READ_RGB(gaussian_id_buffer[i]) / (1.0 - powf(1.0 - weight_buffer[i], BUFF_SIZE)) * weight_buffer[i];
                weight_sum += weight_buffer[i];
            }
        }
        // output_rgb[0] /= weight_sum;

        return; 
    }
    
    if (true) {
        auto seed = params.random_seeds[ray_id];
        constexpr int BUFF_SIZE = 96;

        register unsigned int gaussian_id_buffer[BUFF_SIZE]; 
        register float distances_buffer[BUFF_SIZE]; 
        fill_array(gaussian_id_buffer, BUFF_SIZE, NULL_GAUSSIAN_ID);
        fill_array(distances_buffer, BUFF_SIZE, 999.9f);

        // Sample k gaussians proportionally to their weights
        uint32_t hit_idx = params.prev_hit_per_pixel[ray_id];
        while (hit_idx != 999999999) {
            uint32_t curr_gaussian_id = params.all_gaussian_ids[hit_idx];
            float curr_distance = params.all_distances[hit_idx];
            float curr_alpha = params.all_alphas[hit_idx];
            uint32_t prev_hit = params.all_prev_hits[hit_idx];
            for (int i = 0; i < BUFF_SIZE; i++) {
                if (curr_distance < distances_buffer[i]) {
                    float u = rnd(seed);
                    if (u < curr_alpha) {
                        gaussian_id_buffer[i] = curr_gaussian_id;
                        distances_buffer[i] = curr_distance;
                    }
                }
            }
            hit_idx = prev_hit; 
        }

        // Add their contribution
        for (int i = 0; i < BUFF_SIZE; i++) {
            if (gaussian_id_buffer[i] != NULL_GAUSSIAN_ID) {
                output_rgb[0] += READ_RGB(gaussian_id_buffer[i]) / BUFF_SIZE;
            }
        }
    } else {
        auto seed = params.random_seeds[ray_id];
        constexpr int BUFF_SIZE = 16;

        register unsigned int gaussian_id_buffer[BUFF_SIZE]; 
        register float distances_buffer[BUFF_SIZE]; 
        register float alphas_buffer[BUFF_SIZE]; 
        fill_array(gaussian_id_buffer, BUFF_SIZE, NULL_GAUSSIAN_ID);
        fill_array(distances_buffer, BUFF_SIZE, 999.9f);
        fill_array(alphas_buffer, BUFF_SIZE, 0.0f);

        // Sample k gaussians proportionally to their weights, without replacement
        uint32_t hit_idx = params.prev_hit_per_pixel[ray_id];
        while (hit_idx != 999999999) {
            uint32_t curr_gaussian_id = params.all_gaussian_ids[hit_idx];
            float curr_distance = params.all_distances[hit_idx];
            float curr_alpha = params.all_alphas[hit_idx];
            uint32_t prev_hit = params.all_prev_hits[hit_idx];
            for (int i = 0; i < BUFF_SIZE; i++) {
                if (curr_distance < distances_buffer[i]) {
                    float u = rnd(seed);
                    if (u < curr_alpha) {
                        auto tmp_gaussian_id = gaussian_id_buffer[i];
                        auto tmp_distance = distances_buffer[i];
                        auto tmp_alpha = alphas_buffer[i];

                        gaussian_id_buffer[i] = curr_gaussian_id;
                        distances_buffer[i] = curr_distance;
                        alphas_buffer[i] = curr_alpha;

                        curr_gaussian_id = tmp_gaussian_id;
                        curr_distance = tmp_distance;
                        curr_alpha = tmp_alpha;
                    }
                }
            }
            hit_idx = prev_hit; 
        }

        // Sort the buffer
        for (int i = 0; i < BUFF_SIZE; i++) {
            for (int j = 0; j < i; j++) {
                if (distances_buffer[i] < distances_buffer[j]) {
                    auto tmp_gaussian_id = gaussian_id_buffer[i];
                    auto tmp_distance = distances_buffer[i];
                    auto tmp_alpha = alphas_buffer[i];

                    gaussian_id_buffer[i] = gaussian_id_buffer[j];
                    distances_buffer[i] = distances_buffer[j];
                    alphas_buffer[i] = alphas_buffer[j];

                    gaussian_id_buffer[j] = tmp_gaussian_id;
                    distances_buffer[j] = tmp_distance;
                    alphas_buffer[j] = tmp_alpha;
                }
            }
        }

        // Compute the exact transmittance for all gaussians in the buffer
        hit_idx = params.prev_hit_per_pixel[ray_id];
        register float T_buffer[BUFF_SIZE]; fill_array(T_buffer, BUFF_SIZE, 1.0f);
        while (hit_idx != 999999999) {
            float distance = params.all_distances[hit_idx];
            float alpha = params.all_alphas[hit_idx];
            uint32_t prev_hit = params.all_prev_hits[hit_idx];

            for (int i = 0; i < BUFF_SIZE; i++) {
                if (gaussian_id_buffer[i] != NULL_GAUSSIAN_ID) {
                    if (distance < distances_buffer[i]) { // todo should be <= no?
                        T_buffer[i] *= 1.0f - alpha;
                    }
                }
            }

            hit_idx = prev_hit;
        }

        // Composite all gaussians, estimating the transmittance with that of the nearest gaussian present in the buffer
        hit_idx = params.prev_hit_per_pixel[ray_id];
        float total_weight2 = 0.0f;
        while (hit_idx != 999999999) {
            auto id = params.all_gaussian_ids[hit_idx];
            float distance = params.all_distances[hit_idx];
            float alpha = params.all_alphas[hit_idx];
            uint32_t prev_hit = params.all_prev_hits[hit_idx];

            for (int i = 0; i < BUFF_SIZE - 1; i++) {
                if (distances_buffer[i+1] > distance) { // todo error for gaussians which are exactly present in the buffer
                    if (i == 0 && distances_buffer[i] > distance) {
                        output_rgb[0] += READ_RGB(id) * alpha;
                    } else {
                        output_rgb[0] += READ_RGB(id) * alpha * T_buffer[i];
                        total_weight2 += alpha * T_buffer[i];
                    }
                    break;
                }
            }

            // find
            hit_idx = prev_hit;
        }
    }

    return;
    

#else
    auto seed = params.random_seeds[ray_id];
    
    int total = 0;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        uint32_t gaussian_id = unpackId(gaussian_ids[i]);
        float alpha = unpackAlpha(distances[i]);
        float distance = unpackDistance(distances[i]);
        if (distance >= 89.9f) {  
            continue;
        }
        total++;
        output_rgb[0] += READ_RGB(gaussian_id) / BUFFER_SIZE;
    }
    // output_rgb[0] *= BUFFER_SIZE / total;
    // output_rgb[0] /= 1.0f - params.output_rgbt[ray_id].w;
    return;
#endif