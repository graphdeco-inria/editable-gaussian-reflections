
#pragma inline
__device__ void
forward_pass(const int step, Pixel &pixel, const float3 ray_origin, const float3 ray_direction, int &num_hits) {
    bool grads_enabled = *params.metadata.grads_enabled;
    float transmittance_threshold = *params.config.transmittance_threshold;

    float near_plane = *params.camera.znear;
    if (step != 0) {
        near_plane = 0.0f;
    }
    float far_plane = *params.camera.zfar;

    // * Traverse BVH
    uint32_t step_uint = (uint32_t)step;
    uint32_t total_transmittance_uint = __float_as_uint(1.0f);
    uint32_t grads_enabled_uint = *params.metadata.grads_enabled;
    uint32_t alpha_threshold_uint = __float_as_uint(*params.config.alpha_threshold);
    uint32_t exp_power_uint = __float_as_uint(*params.config.exp_power);
    uint32_t backfacing_max_dist_uint = __float_as_uint(*params.config.backfacing_max_dist);
    uint32_t backfacing_invalid_normal_threshold_uint =
        __float_as_uint(*params.config.backfacing_invalid_normal_threshold);
    uint32_t num_traversed_per_pixel = 0;
    optixTraverse(
        params.bvh_handle,
        ray_origin,
        ray_direction,
        near_plane, // tmin
        far_plane,
        0.0f, // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0, // SBTOffset
        0, // SBTStride
        0, // missSBTIndex
        step_uint,
        total_transmittance_uint,
        grads_enabled_uint,
        alpha_threshold_uint,
        exp_power_uint,
        backfacing_max_dist_uint,
        backfacing_invalid_normal_threshold_uint,
        num_traversed_per_pixel);
    pixel.output_transmittance[step] = 1.0f;
    pixel.output_total_transmittance[step] = __uint_as_float(total_transmittance_uint);
    atomicAdd(&params.stats.num_traversed_per_pixel[pixel.id], num_traversed_per_pixel);

    // * Initialize registers holding the BUFFER_SIZE nearest gaussians
    register float distances[BUFFER_SIZE];
    register unsigned int hit_idxes[BUFFER_SIZE];

    // * Loop over batches from the PPLL
    int total_accumulated = 0;
    float tmin = near_plane;
    for (int iteration = 0; iteration < MAX_ITERATIONS && tmin < far_plane; iteration++) {
        fill_array(distances, BUFFER_SIZE, std::numeric_limits<float>::max());
        fill_array(hit_idxes, BUFFER_SIZE, PerPixelLinkedList::NULL_PTR);

        // * Fill batch with nearest gaussians behind the last one
        for (auto hit_idx : params.ppll_forward.pixel_view(pixel.id)) {
            if (pixel.id == 777 && hit_idx == 832839) {
                printf(" ");
            } //!!!!!!! kludge to fix an undetermined result, this printf
              //! affects compilation and cannot be removed
            float curr_distance = params.ppll_forward.distances[hit_idx];
            if (curr_distance > tmin && curr_distance < distances[BUFFER_SIZE - 1]) {
                distances[BUFFER_SIZE - 1] = curr_distance;
                hit_idxes[BUFFER_SIZE - 1] = hit_idx;
            }
#pragma unroll
            for (int i = BUFFER_SIZE - 1; i > 0; i--) {
                if (distances[i] < distances[i - 1]) {
                    // * Swap i with i-1
                    float tmp_dist = distances[i];
                    int tmp_idx = hit_idxes[i];
                    distances[i] = distances[i - 1];
                    hit_idxes[i] = hit_idxes[i - 1];
                    distances[i - 1] = tmp_dist;
                    hit_idxes[i - 1] = tmp_idx;
                }
            }
        }

        // * Break if all gaussians are processed
        if (hit_idxes[0] == PerPixelLinkedList::NULL_PTR) {
            break;
        }

// * Integrate the batch of gaussians
#pragma unroll
        for (int i = 0; i < BUFFER_SIZE; i++) {
            float distance = distances[i];
            tmin = distance;

            if (distance < far_plane) {
                num_hits++;
                total_accumulated++;

                // * Fetch data from PPLL
                uint32_t gaussian_id = params.ppll_forward.gaussian_ids[hit_idxes[i]];
                float3 local_hit = params.ppll_forward.local_hits[hit_idxes[i]];
                float gaussval = params.ppll_forward.gaussvals[hit_idxes[i]];
                float alpha = params.ppll_forward.alphas[hit_idxes[i]];

                // * Fetch gaussian data
                float3 gaussian_rgb = read_rgb(params, gaussian_id);
                float3 gaussian_normal = read_normal(params, gaussian_id);
                float3 gaussian_f0 = read_f0(params, gaussian_id);
                float gaussian_roughness = read_roughness(params, gaussian_id);

                // * Accumulate values
                float next_transmittance = pixel.output_transmittance[step] * (1.0f - alpha);
                float weight = pixel.output_transmittance[step] - next_transmittance;
                pixel.output_rgb[step] += gaussian_rgb * weight;
                pixel.output_normal[step] += gaussian_normal * weight;
                pixel.output_f0[step] += gaussian_f0 * weight;
                pixel.output_roughness[step] += gaussian_roughness * weight;
                pixel.output_depth[step] += distance * weight;
                pixel.output_transmittance[step] = next_transmittance;

                // * Store data required in backward pass PPLL
                if (grads_enabled) {
                    params.ppll_backward.insert(
                        grads_enabled,
                        pixel.id,
                        gaussian_id,
                        distance,
                        local_hit,
                        gaussval,
                        alpha,
                        pixel.output_transmittance[step]);
                }

                // * Break if transmittance threshold is reached
                if (pixel.output_transmittance[step] < transmittance_threshold) {
                    tmin = far_plane;
                    break;
                }
            }
        }
    }

    // * Update stats
    params.stats.num_accumulated_per_pixel[pixel.id] = total_accumulated;

    // * Approximate the contribution of truncated gaussians & update output
    float remaining_transmittance = pixel.output_transmittance[step] - pixel.output_total_transmittance[step];
    float normalization = max((1.0f - pixel.output_transmittance[step]), *params.config.eps_forward_normalization);
    pixel.remaining_rgb[step] = pixel.output_rgb[step] / normalization;
    pixel.output_rgb[step] = pixel.output_rgb[step] + remaining_transmittance * pixel.remaining_rgb[step];
    pixel.remaining_depth[step] = pixel.output_depth[step] / normalization;
    pixel.output_depth[step] = pixel.output_depth[step] + remaining_transmittance * pixel.remaining_depth[step];
    pixel.remaining_normal[step] = pixel.output_normal[step] / normalization;
    pixel.output_normal[step] = pixel.output_normal[step] + remaining_transmittance * pixel.remaining_normal[step];
    pixel.remaining_f0[step] = pixel.output_f0[step] / normalization;
    pixel.output_f0[step] = pixel.output_f0[step] + remaining_transmittance * pixel.remaining_f0[step];
    pixel.remaining_roughness[step] = pixel.output_roughness[step] / normalization;
    pixel.output_roughness[step] =
        pixel.output_roughness[step] + remaining_transmittance * pixel.remaining_roughness[step];
}