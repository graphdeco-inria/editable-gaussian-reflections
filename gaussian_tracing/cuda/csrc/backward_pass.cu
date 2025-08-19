

__device__ float3 sign(float3 v) {
    return make_float3(
        copysignf(1.0f, v.x), copysignf(1.0f, v.y), copysignf(1.0f, v.z));
}

__device__ float sign(float v) { return copysignf(1.0f, v); }

#if BACKWARDS_PASS == true
#pragma inline
__device__ void backward_pass(
    const int step,
    float3 ray_origin_world,
    float3 ray_direction_world,
    //
    const float initial_blur_level,
    const float blur_by_distance,
    //
    const uint32_t &ray_id,
    //
    const float3 (&output_rgb_raw)[TILE_SIZE * TILE_SIZE],
    const float3 (&output_rgb)[TILE_SIZE * TILE_SIZE],
    const float3 (&final_rgb)[TILE_SIZE * TILE_SIZE],
    const float2 (&output_t)[TILE_SIZE * TILE_SIZE],
    const float3 (&output_position)[TILE_SIZE * TILE_SIZE],
    const float (&output_depth)[TILE_SIZE * TILE_SIZE],
    const float3 (&output_normal)[TILE_SIZE * TILE_SIZE],
    const float3 (&output_f0)[TILE_SIZE * TILE_SIZE],
    const float (&output_roughness)[TILE_SIZE * TILE_SIZE],
    const float (&output_distortion)[TILE_SIZE * TILE_SIZE],
    //
    const float3 (&remaining_rgb)[TILE_SIZE * TILE_SIZE],
    const float3 (&remaining_position)[TILE_SIZE * TILE_SIZE],
    const float (&remaining_depth)[TILE_SIZE * TILE_SIZE],
    const float3 (&remaining_normal)[TILE_SIZE * TILE_SIZE],
    const float3 (&remaining_f0)[TILE_SIZE * TILE_SIZE],
    const float (&remaining_roughness)[TILE_SIZE * TILE_SIZE],
    const float (&remaining_distortion)[TILE_SIZE * TILE_SIZE],
    //
    const int num_hits,
    const float3 throughput[TILE_SIZE * TILE_SIZE],
    float3 dL_dthroughput_out[TILE_SIZE * TILE_SIZE],

    //
    const float3 target_rgb[TILE_SIZE * TILE_SIZE],
    const float3 target_diffuse[TILE_SIZE * TILE_SIZE],
    const float3 target_glossy[TILE_SIZE * TILE_SIZE],
    const float3 target_position[TILE_SIZE * TILE_SIZE],
    const float target_depth[TILE_SIZE * TILE_SIZE],
    const float3 target_normal[TILE_SIZE * TILE_SIZE],
    const float3 target_f0[TILE_SIZE * TILE_SIZE],
    const float target_roughness[TILE_SIZE * TILE_SIZE],
    //
    const float3 error[TILE_SIZE * TILE_SIZE],
    const float loss_modulation,
    const float loss_weight,

    float3 &dL_dray_origin_out,
    float3 &dL_dray_direction_out) {

    float3 backward_prev_gaussian_rgb[TILE_SIZE * TILE_SIZE];
    fill_array(
        backward_prev_gaussian_rgb,
        TILE_SIZE * TILE_SIZE,
        make_float3(0.0f, 0.0f, 0.0f));
    float3 backward_weighted_rgb_deltas[TILE_SIZE * TILE_SIZE];
    fill_array(
        backward_weighted_rgb_deltas,
        TILE_SIZE * TILE_SIZE,
        make_float3(0.0f, 0.0f, 0.0f));
    //
    float3 backward_prev_gaussian_position[TILE_SIZE * TILE_SIZE];
    fill_array(
        backward_prev_gaussian_position,
        TILE_SIZE * TILE_SIZE,
        make_float3(0.0f, 0.0f, 0.0f));
    float3 backward_weighted_position_deltas[TILE_SIZE * TILE_SIZE];
    fill_array(
        backward_weighted_position_deltas,
        TILE_SIZE * TILE_SIZE,
        make_float3(0.0f, 0.0f, 0.0f));
    //
    float backward_prev_gaussian_depth[TILE_SIZE * TILE_SIZE];
    fill_array(backward_prev_gaussian_depth, TILE_SIZE * TILE_SIZE, 0.0f);
    float backward_weighted_depth_deltas[TILE_SIZE * TILE_SIZE];
    fill_array(backward_weighted_depth_deltas, TILE_SIZE * TILE_SIZE, 0.0f);
    //
    float3 backward_prev_gaussian_normal[TILE_SIZE * TILE_SIZE];
    fill_array(
        backward_prev_gaussian_normal,
        TILE_SIZE * TILE_SIZE,
        make_float3(0.0f, 0.0f, 0.0f));
    float3 backward_weighted_normal_deltas[TILE_SIZE * TILE_SIZE];
    fill_array(
        backward_weighted_normal_deltas,
        TILE_SIZE * TILE_SIZE,
        make_float3(0.0f, 0.0f, 0.0f));
    //
    float3 backward_prev_gaussian_f0[TILE_SIZE * TILE_SIZE];
    fill_array(
        backward_prev_gaussian_f0,
        TILE_SIZE * TILE_SIZE,
        make_float3(0.0f, 0.0f, 0.0f));
    float3 backward_weighted_f0_deltas[TILE_SIZE * TILE_SIZE];
    fill_array(
        backward_weighted_f0_deltas,
        TILE_SIZE * TILE_SIZE,
        make_float3(0.0f, 0.0f, 0.0f));
    //
    float backward_prev_gaussian_roughness[TILE_SIZE * TILE_SIZE];
    fill_array(backward_prev_gaussian_roughness, TILE_SIZE * TILE_SIZE, 0.0f);
    float backward_weighted_roughness_deltas[TILE_SIZE * TILE_SIZE];
    fill_array(backward_weighted_roughness_deltas, TILE_SIZE * TILE_SIZE, 0.0f);

    if (*params.grads_enabled) {
        int i = num_hits - 1;

        uint32_t hit_idx = params.prev_hit_per_pixel_for_backprop[ray_id];
        while (hit_idx != 999999999u && i >= 0) {

            uint32_t gaussian_id =
                params.all_gaussian_ids_for_backprop[hit_idx];

            float3 local_hits[TILE_SIZE * TILE_SIZE];
            float curr_Ts[TILE_SIZE * TILE_SIZE];
            float alphas[TILE_SIZE * TILE_SIZE];
            float gaussvals[TILE_SIZE * TILE_SIZE];
            float distances[TILE_SIZE * TILE_SIZE];

#if TILE_SIZE == 1
            local_hits[0] = params.all_local_hits_for_backprop[hit_idx];
            curr_Ts[0] = params.all_Ts_for_backprop[hit_idx];
            alphas[0] = params.all_alphas_for_backprop[hit_idx];
            gaussvals[0] = params.all_gaussvals_for_backprop[hit_idx];
            distances[0] = params.all_distances_for_backprop[hit_idx];
#else
#pragma unroll
            for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
                // #if TILE_SIZE > 1
                //     float curr_T = params.all_Ts_for_backprop[TILE_SIZE *
                //     TILE_SIZE * hit_idx + k]; float alpha =
                //     params.all_alphas_for_backprop[TILE_SIZE * TILE_SIZE *
                //     hit_idx + k]; float gaussval =
                //     params.all_gaussvals_for_backprop[TILE_SIZE
                //     * TILE_SIZE * hit_idx + k]; float3 local_hit =
                //     params.all_local_hits_for_backprop[TILE_SIZE * TILE_SIZE
                //     * hit_idx + k];
                //     // float3 local_hit =
                //     params.all_local_hits_for_backprop[hit_idx];
                // #endif
                curr_Ts[k] = params.all_Ts_for_backprop
                                 [TILE_SIZE * TILE_SIZE * hit_idx + k];
                alphas[k] = params.all_alphas_for_backprop
                                [TILE_SIZE * TILE_SIZE * hit_idx + k];
                gaussvals[k] = params.all_gaussvals_for_backprop
                                   [TILE_SIZE * TILE_SIZE * hit_idx + k];
                local_hits[k] = params.all_local_hits_for_backprop
                                    [TILE_SIZE * TILE_SIZE * hit_idx + k];
                distances[k] = params.all_distances_for_backprop
                                   [TILE_SIZE * TILE_SIZE * hit_idx + k];
            }
#endif

            // float3 gaussian_rgb = READ_RGB(gaussian_id);
            float3 gaussian_rgb_unactivated = params.gaussian_rgb[gaussian_id];

#if ACTIVATION_IN_CUDA == true
#if RELU_INSTEAD_OF_SOFTPLUS == true
            float3 gaussian_rgb = relu_act(gaussian_rgb_unactivated);
#else
            float3 gaussian_rgb = softplus_act(gaussian_rgb_unactivated);
#endif
#else
            float3 gaussian_rgb = gaussian_rgb_unactivated;
#endif

#if ATTACH_POSITION == true
            float3 gaussian_position;
#endif

#if ATTACH_NORMALS == true
            float3 gaussian_normal;
#endif
#if ATTACH_F0 == true
            float3 gaussian_f0;
#endif
#if ATTACH_ROUGHNESS == true
            float gaussian_roughness;
#endif

            if (step == 0) {
#if ATTACH_POSITION == true
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
                gaussian_position =
                    ray_origin_world + distances[0] * ray_direction_world;
#else
                gaussian_position = params.gaussian_position[gaussian_id];
#endif
#endif
#if ATTACH_NORMALS == true
                gaussian_normal = params.gaussian_normal[gaussian_id];
#endif
#if ATTACH_F0 == true
                gaussian_f0 = READ_F0(gaussian_id);
#endif
#if ATTACH_ROUGHNESS == true
                gaussian_roughness = READ_ROUGHNESS(gaussian_id);
#endif
            }

            float opacity = READ_OPACITY(gaussian_id);
            const float4 *world_to_local =
                optixGetInstanceInverseTransformFromHandle(
                    optixGetInstanceTraversableFromIAS(
                        params.handle, gaussian_id));
            const float4 *local_to_world = optixGetInstanceTransformFromHandle(
                optixGetInstanceTraversableFromIAS(params.handle, gaussian_id));
            float3 scaling = READ_SCALE(gaussian_id);
            float4 rotation_unnormalized =
                params.gaussian_rotations[gaussian_id];
#if ACTIVATION_IN_CUDA == true
            float4 rotation = normalize_act(rotation_unnormalized);
#else
            float4 rotation = rotation_unnormalized;
#endif

            float3 dL_drgb_total = make_float3(0.0f, 0.0f, 0.0f);
#if ATTACH_POSITION == true && POSITION_FROM_EXPECTED_TERMINATION_DEPTH == false
            float3 dL_dgaussian_position_total = make_float3(0.0f, 0.0f, 0.0f);
#endif
#if ATTACH_NORMALS == true
            float3 dL_dgaussian_normal_total = make_float3(0.0f, 0.0f, 0.0f);
#endif
#if ATTACH_F0 == true
            float3 dL_dgaussian_f0_total = make_float3(0.0f, 0.0f, 0.0f);
#endif
#if ATTACH_ROUGHNESS == true
            float dL_dgaussian_roughness_total = 0.0f;
#endif
            float dL_dopacity_total = 0.0f;
            float4 dL_drot_total = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 dL_dscale_total = make_float3(0.0f, 0.0f, 0.0f);
            float3 dL_dmean_total = make_float3(0.0f, 0.0f, 0.0f);
#if OPTIMIZE_EXP_POWER == true
            float dL_dexp_powers_total = 0.0f;
#endif
#if EXPECTED_TERMINATION_GRADIENTS == true
            float3 dL_dexpected_termination_total =
                make_float3(0.0f, 0.0f, 0.0f);
#endif

            float weight;
            for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
                // #if TILE_SIZE > 1
                //     float curr_T = params.all_Ts_for_backprop[TILE_SIZE *
                //     TILE_SIZE * hit_idx + k]; float alpha =
                //     params.all_alphas_for_backprop[TILE_SIZE * TILE_SIZE *
                //     hit_idx + k]; float gaussval =
                //     params.all_gaussvals_for_backprop[TILE_SIZE
                //     * TILE_SIZE * hit_idx + k]; float3 local_hit =
                //     params.all_local_hits_for_backprop[TILE_SIZE * TILE_SIZE
                //     * hit_idx + k];
                //     // float3 local_hit =
                //     params.all_local_hits_for_backprop[hit_idx];
                // #endif

                float curr_T = curr_Ts[k];
                float alpha = alphas[k];
                float gaussval = gaussvals[k];
                float3 local_hit = local_hits[k];

// * Loss gradient
#if SKIP_LOSS_AVG_NUM_PIXELS == true
                int num_pixels = 1;
#else
                int num_pixels = params.image_height * params.image_width;
#endif

                // the *2 is the exponent dropping down, and the / 3 is the 3
                // channels being averaged over
                float3 dL_doutput_rgb =
                    2.0f / 3.0f * sign(error[k]) * loss_weight / num_pixels;

                float dL_doutput_depth = 0.0f;
                float3 dL_doutput_position = make_float3(0.0f, 0.0f, 0.0f);
                float3 dL_doutput_normal = make_float3(0.0f, 0.0f, 0.0f);
                float3 dL_doutput_f0 = make_float3(0.0f, 0.0f, 0.0f);
                float dL_doutput_roughness = 0.0f;

                if (step == 0) {
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
                    dL_doutput_depth =
                        2.0f * sign(output_depth[k] - target_depth[k]) /
                        num_pixels *
                        (step == 0 ? params.position_loss_weight : 0.0f);
#else
                    dL_doutput_position =
                        2.0f / 3.0f *
                        sign(output_position[k] - target_position[k]) /
                        num_pixels *
                        (step == 0 ? params.position_loss_weight : 0.0f);
#endif
                    dL_doutput_normal =
                        2.0f / 3.0f *
                        sign(output_normal[k] - target_normal[k]) / num_pixels *
                        (step == 0 ? params.normal_loss_weight : 0.0f);
                    dL_doutput_f0 =
                        2.0f / 3.0f * sign(output_f0[k] - target_f0[k]) /
                        num_pixels * (step == 0 ? params.f0_loss_weight : 0.0f);
                    dL_doutput_roughness =
                        2.0f / 1.0f *
                        sign(output_roughness[k] - target_roughness[k]) /
                        num_pixels *
                        (step == 0 ? params.roughness_loss_weight : 0.0f);
                } else {
                    dL_doutput_rgb = dL_doutput_rgb * throughput[k];
                    dL_dthroughput_out[k] =
                        dL_doutput_rgb *
                        output_rgb[k]; // important that this is +=

                    dL_doutput_rgb *= loss_modulation;
                    //? what about dL_dthroughput_out?
                }

                // * Color gradient
                weight = curr_T / (1.0 - alpha) * alpha;
                float3 dL_drgb = dL_doutput_rgb * weight;
#if ACTIVATION_IN_CUDA == true
                // #if RELU_INSTEAD_OF_SOFTPLUS == true
                dL_drgb = backward_relu_act(dL_drgb, gaussian_rgb);
                // #else
                //     dL_drgb = backward_softplus_act(dL_drgb,
                //     gaussian_rgb_unactivated, gaussian_rgb);
                // #endif
#endif

                float3 dL_dgaussian_position = dL_doutput_position * weight;
                float3 dL_dgaussian_normal = dL_doutput_normal * weight;
                float3 dL_dgaussian_f0 = dL_doutput_f0 * weight;
                float dL_dgaussian_roughness = dL_doutput_roughness * weight;

                dL_drgb_total += dL_drgb;
                if (step == 0) {
#if ATTACH_POSITION == true && POSITION_FROM_EXPECTED_TERMINATION_DEPTH == false
                    dL_dgaussian_position_total += dL_dgaussian_position;
#endif
#if ATTACH_NORMALS == true
                    dL_dgaussian_normal_total += dL_dgaussian_normal;
#endif
// #if CLIPPED_RELU_INSTEAD_OF_SIGMOID == true
#if ATTACH_F0 == true
                    dL_dgaussian_f0_total +=
                        backward_clipped_relu_act(dL_dgaussian_f0, gaussian_f0);
#endif
#if ATTACH_ROUGHNESS == true
                    dL_dgaussian_roughness_total += backward_clipped_relu_act(
                        dL_dgaussian_roughness, gaussian_roughness);
#endif
                    // #else
                    //     #if ATTACH_F0 == true
                    //         dL_dgaussian_f0_total +=
                    //         backward_sigmoid_act(dL_dgaussian_f0,
                    //         gaussian_f0);
                    //     #endif
                    //     #if ATTACH_ROUGHNESS == true
                    //         dL_dgaussian_roughness_total +=
                    //         backward_sigmoid_act(dL_dgaussian_roughness,
                    //         gaussian_roughness);
                    //     #endif
                    // #endif
                }

                // * Alpha gradient
                backward_weighted_rgb_deltas[k] +=
                    (gaussian_rgb - backward_prev_gaussian_rgb[k]) * curr_T;
                backward_prev_gaussian_rgb[k] =
                    gaussian_rgb; // todo re-read from memory instead of storing
                                  // in registers
                if (step == 0) {
#if ATTACH_POSITION == true
                    backward_weighted_position_deltas[k] +=
                        (gaussian_position -
                         backward_prev_gaussian_position[k]) *
                        curr_T;
                    backward_prev_gaussian_position[k] =
                        gaussian_position; // todo re-read from memory instead
                                           // of storing in registers
#endif
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
                    backward_weighted_depth_deltas[k] +=
                        (distances[0] - backward_prev_gaussian_depth[k]) *
                        curr_T;
                    backward_prev_gaussian_depth[k] =
                        distances[0]; // todo re-read from memory instead of
                                      // storing in registers
#endif
#if ATTACH_NORMALS == true
                    backward_weighted_normal_deltas[k] +=
                        (gaussian_normal - backward_prev_gaussian_normal[k]) *
                        curr_T;
                    backward_prev_gaussian_normal[k] =
                        gaussian_normal; // todo re-read from memory instead of
                                         // storing in registers
#endif
#if ATTACH_F0 == true
                    backward_weighted_f0_deltas[k] +=
                        (gaussian_f0 - backward_prev_gaussian_f0[k]) * curr_T;
                    backward_prev_gaussian_f0[k] =
                        gaussian_f0; // todo re-read from memory instead of
                                     // storing in registers
#endif
#if ATTACH_ROUGHNESS == true
                    backward_weighted_roughness_deltas[k] +=
                        (gaussian_roughness -
                         backward_prev_gaussian_roughness[k]) *
                        curr_T;
                    backward_prev_gaussian_roughness[k] =
                        gaussian_roughness; // todo re-read from memory instead
                                            // of storing in registers
#endif
                }

                float dL_dalpha =
                    dot(backward_weighted_rgb_deltas[k] / (1.0f - alpha),
                        dL_doutput_rgb);

#if REMAINING_COLOR_ESTIMATION != NO_ESTIMATION
                dL_dalpha +=
                    -((output_t[k].x - output_t[k].y) / (1.0 - alpha)) *
                    dot(remaining_rgb[k], dL_doutput_rgb);
// #if ATTACH_POSITION == true
//     dL_dalpha += -((output_t[k].x - output_t[k].y) / (1.0 - alpha)) *
//     dot(remaining_position[k], dL_doutput_position);
// #endif
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
                dL_dalpha +=
                    -((output_t[k].x - output_t[k].y) / (1.0 - alpha)) *
                    remaining_depth[k] * dL_doutput_depth;
#endif
#if ATTACH_NORMALS == true
                dL_dalpha +=
                    -((output_t[k].x - output_t[k].y) / (1.0 - alpha)) *
                    dot(remaining_normal[k], dL_doutput_normal);
#endif
#if ATTACH_F0 == true
                dL_dalpha +=
                    -((output_t[k].x - output_t[k].y) / (1.0 - alpha)) *
                    dot(remaining_f0[k], dL_doutput_f0);
#endif
#if ATTACH_ROUGHNESS == true
                dL_dalpha +=
                    -((output_t[k].x - output_t[k].y) / (1.0 - alpha)) *
                    remaining_roughness[k] * dL_doutput_roughness;
#endif
#endif

                if (step == 0) {
// #if ATTACH_POSITION == true
//     dL_dalpha += dot(backward_weighted_position_deltas[k] / (1.0f - alpha),
//     dL_doutput_position);
// #endif
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
                    dL_dalpha += backward_weighted_depth_deltas[k] /
                                 (1.0f - alpha) * dL_doutput_depth;
#endif
#if ATTACH_NORMALS == true
                    dL_dalpha +=
                        dot(backward_weighted_normal_deltas[k] / (1.0f - alpha),
                            dL_doutput_normal);
#endif
#if ATTACH_F0 == true
                    dL_dalpha +=
                        dot(backward_weighted_f0_deltas[k] / (1.0f - alpha),
                            dL_doutput_f0);
#endif
#if ATTACH_ROUGHNESS == true
                    dL_dalpha += backward_weighted_roughness_deltas[k] /
                                 (1.0f - alpha) * dL_doutput_roughness;
#endif
                }

                float windowing = 1.0f;

#if ALLOW_OPACITY_ABOVE_1 == true
                float dL_dopacity = dL_dalpha * gaussval * windowing;
#else
                float dL_dopacity =
                    MAX_ALPHA * dL_dalpha * gaussval * windowing;
#endif
#if ALPHA_RESCALE == true
                dL_dopacity /= 1 - params.alpha_threshold;
#endif
#if ALPHA_SMOOTHING == true
                if (alpha < ALPHA_SMOOTHING_THRESHOLD) {
                    dL_dopacity /= 1 + params.alpha_threshold;
                }
#endif

#if ACTIVATION_IN_CUDA == true
#if ALLOW_OPACITY_ABOVE_1 == true
                dL_dopacity = backward_softplus_act(dL_dopacity, opacity);
#else
                dL_dopacity = backward_sigmoid_act(dL_dopacity, opacity);
#endif
#endif
                dL_dopacity_total += dL_dopacity;

// * Transform gradient
#if ALLOW_OPACITY_ABOVE_1 == true
                float dL_dgaussval = dL_dalpha * opacity *
                                     windowing; // straight-through gradient
#else
                float dL_dgaussval =
                    MAX_ALPHA * dL_dalpha * opacity * windowing;
#endif
#if ALPHA_RESCALE == true
                dL_dgaussval /= 1 + params.alpha_threshold;
#endif
#if ALPHA_SMOOTHING == true
                if (alpha < ALPHA_SMOOTHING_THRESHOLD) {
                    dL_dgaussval /= 1 + params.alpha_threshold;
                }
#endif

#if OPTIMIZE_EXP_POWER == true
                float exp_power = params.gaussian_exp_power[gaussian_id];
#else
                float exp_power = params.exp_power;
#endif
                float sq_norm = dot(local_hit, local_hit);

// * Generalized gaussian exponent gradient
#if OPTIMIZE_EXP_POWER == true
                float dL_dexp_powers = -dL_dgaussval * gaussval *
                                       powf(sq_norm, exp_power) *
                                       logf(sq_norm) / 2.0f;
                dL_dexp_powers_total = dL_dexp_powers;

#if USE_EPANECHNIKOV_KERNEL == true
                printf(
                    "Epanechnikov kernel not supported with optimized exp "
                    "power\n");
#endif
#endif

// * Local hit point gradient
#if SQUARE_KERNEL == true
                float3 dL_dx_local =
                    -local_hit *
                    make_float3(
                        powf(fabs(local_hit.x), 2.0f * exp_power - 2.0f),
                        powf(fabs(local_hit.y), 2.0f * exp_power - 2.0f),
                        powf(fabs(local_hit.z), 2.0f * exp_power - 2.0f)) *
                    gaussval * dL_dgaussval;
#else
#if USE_EPANECHNIKOV_KERNEL == true
                float dL_dsq_norm =
                    2 * exp_power * powf(sq_norm, exp_power - 1.0f);
#else
                float dL_dsq_norm = gaussval * powf(sq_norm, exp_power - 1.0f);
#endif
                float3 dL_dx_local = -local_hit * dL_dsq_norm * dL_dgaussval;
#endif

                // * World hit point gradient
                float scaling_factor = compute_scaling_factor(
                    opacity, params.alpha_threshold, exp_power);
                float3 dL_dx_world = make_float3(
                                         dot(make_float3(
                                                 world_to_local[0].x,
                                                 world_to_local[1].x,
                                                 world_to_local[2].x),
                                             dL_dx_local),
                                         dot(make_float3(
                                                 world_to_local[0].y,
                                                 world_to_local[1].y,
                                                 world_to_local[2].y),
                                             dL_dx_local),
                                         dot(make_float3(
                                                 world_to_local[0].z,
                                                 world_to_local[1].z,
                                                 world_to_local[2].z),
                                             dL_dx_local)) *
                                     scaling_factor;

                // {   // todo pull out changes of basis into a function
                //     float norm = sqrtf(sq_norm) + 1e-12f;
                //     float3 direction_world = ray_direction_world;
                //     float3 direction_local_unnormalized  = make_float3(
                //         dot(make_float3(world_to_local[0]),
                //         ray_direction_world),
                //         dot(make_float3(world_to_local[1]),
                //         ray_direction_world),
                //         dot(make_float3(world_to_local[2]),
                //         ray_direction_world)
                //     ) / (scaling_factor + EPS_SCALE_GRAD); //!!!!!!! review
                //     scaling factor, factor on dL_dx_world is also sus since
                //     it is used below float3 direction_local =
                //     normalize(direction_local_unnormalized); float3
                //     origin_local = make_float3(
                //         dot(make_float3(world_to_local[0]),
                //         ray_origin_world),
                //         dot(make_float3(world_to_local[1]),
                //         ray_origin_world),
                //         dot(make_float3(world_to_local[2]), ray_origin_world)
                //     ) / (scaling_factor + EPS_SCALE_GRAD);
                //     float dL_dt_world = dot(dL_dx_world, direction_world);
                //     float dL_dt_local = dL_dt_world / norm;
                //     float3 dL_dorigin_local = dL_dt_local * -direction_local;
                //     float3 dL_dorigin_world = make_float3(
                //         dot(make_float3(world_to_local[0].x,
                //         world_to_local[1].x, world_to_local[2].x),
                //         dL_dorigin_local),
                //         dot(make_float3(world_to_local[0].y,
                //         world_to_local[1].y, world_to_local[2].y),
                //         dL_dorigin_local),
                //         dot(make_float3(world_to_local[0].z,
                //         world_to_local[1].z, world_to_local[2].z),
                //         dL_dorigin_local)
                //     ) + dL_dx_world;
                //     float3 dL_ddirection_local = dL_dt_local * -origin_local;
                //     float3 dL_direction_local_unnormalized =
                //     dL_ddirection_local * (dL_ddirection_local/norm -
                //     dot(direction_local_unnormalized, dL_ddirection_local) *
                //     direction_local_unnormalized / powf(norm, 3.0f)); float3
                //     dL_ddirection_world = make_float3(
                //         dot(make_float3(world_to_local[0].x,
                //         world_to_local[1].x, world_to_local[2].x),
                //         dL_direction_local_unnormalized),
                //         dot(make_float3(world_to_local[0].y,
                //         world_to_local[1].y, world_to_local[2].y),
                //         dL_direction_local_unnormalized),
                //         dot(make_float3(world_to_local[0].z,
                //         world_to_local[1].z, world_to_local[2].z),
                //         dL_direction_local_unnormalized)
                //     );

                //     dL_dray_origin_out += dL_dorigin_world;
                //     dL_dray_direction_out += dL_ddirection_world;
                // }

                // * Local to world matrix gradient
                float3 dL_dl2w_0 = -dL_dx_world.x * local_hit;
                float3 dL_dl2w_1 = -dL_dx_world.y * local_hit;
                float3 dL_dl2w_2 = -dL_dx_world.z * local_hit;

                // * Mean gradient
                dL_dmean_total -= dL_dx_world;

                // * Scaling gradient
                float3 rot_0 =
                    make_float3(local_to_world[0]) /
                    (scaling * scaling_factor + ANTIALIASING + EPS_SCALE_GRAD);
                float3 rot_1 =
                    make_float3(local_to_world[1]) /
                    (scaling * scaling_factor + ANTIALIASING + EPS_SCALE_GRAD);
                float3 rot_2 =
                    make_float3(local_to_world[2]) /
                    (scaling * scaling_factor + ANTIALIASING + EPS_SCALE_GRAD);
                float3 dL_dscale =
                    dL_dl2w_0 * rot_0 + dL_dl2w_1 * rot_1 + dL_dl2w_2 * rot_2;

#if ACTIVATION_IN_CUDA == true
                dL_dscale = backward_exp_act(dL_dscale, scaling);
#endif
                dL_dscale_total += dL_dscale;

                // * Rotation matrix gradient
                float3 dL_drot_0 = dL_dl2w_0 * (scaling + ANTIALIASING);
                float3 dL_drot_1 = dL_dl2w_1 * (scaling + ANTIALIASING);
                float3 dL_drot_2 = dL_dl2w_2 * (scaling + ANTIALIASING);

                // * Rotation quaternion gradient
                float r = rotation.x;
                float x = rotation.y;
                float y = rotation.z;
                float z = rotation.w;
                float dL_dr =
                    (2.f * x * (dL_drot_2.y - dL_drot_1.z) +
                     2.f * y * (dL_drot_0.z - dL_drot_2.x) +
                     2.f * z * (dL_drot_1.x - dL_drot_0.y));
                float dL_dx =
                    (-4.f * x * (dL_drot_1.y + dL_drot_2.z) +
                     2.f * y * (dL_drot_0.y + dL_drot_1.x) +
                     2.f * z * (dL_drot_0.z + dL_drot_2.x) +
                     2.f * r * (dL_drot_2.y - dL_drot_1.z));
                float dL_dy =
                    (2.f * x * (dL_drot_0.y + dL_drot_1.x) -
                     4.f * y * (dL_drot_0.x + dL_drot_2.z) +
                     2.f * z * (dL_drot_1.z + dL_drot_2.y) +
                     2.f * r * (dL_drot_0.z - dL_drot_2.x));
                float dL_dz =
                    (2.f * x * (dL_drot_0.z + dL_drot_2.x) +
                     2.f * y * (dL_drot_1.z + dL_drot_2.y) -
                     4.f * z * (dL_drot_0.x + dL_drot_1.y) +
                     2.f * r * (dL_drot_1.x - dL_drot_0.y));
                float4 dL_drot = make_float4(dL_dr, dL_dx, dL_dy, dL_dz);
#if ACTIVATION_IN_CUDA == true
                dL_drot = backward_normalize_act(
                    dL_drot, rotation_unnormalized, rotation);
#endif
                dL_drot_total += dL_drot;
            }

#if USE_GRADIENT_SCALING == true
#if TILE_SIZE > 1
            printf(
                "Error: gradient scaling not supported with tile size > 1\n");
#endif

#if GRADIENT_SCALING_UNCLAMPED == true
            float grad_dist_weight = powf(distances[0], 2.0f);
#else
            float grad_dist_weight =
                max(min(powf(distances[0], 2.0f), 1.0f), 0.0f);
#endif
#else
            float grad_dist_weight = 1.0f;
#endif

            // * Flush to memory
            if (i < DETACH_AFTER * BUFFER_SIZE) {
                atomicAdd3(
                    &params.dL_drgb[gaussian_id],
                    dL_drgb_total * grad_dist_weight * GLOBAL_GRADIENT_SCALE);
                atomicAdd(
                    &params.dL_dopacity[gaussian_id],
                    dL_dopacity_total * grad_dist_weight *
                        GLOBAL_GRADIENT_SCALE);
                if (step == 0) {
#if ATTACH_NORMALS == true
                    atomicAdd3(
                        &params.dL_dgaussian_normal[gaussian_id],
                        dL_dgaussian_normal_total * grad_dist_weight *
                            GLOBAL_GRADIENT_SCALE);
#endif
// #if ATTACH_POSITION == true && POSITION_FROM_EXPECTED_TERMINATION_DEPTH ==
// false
//     atomicAdd3(&params.dL_dgaussian_position[gaussian_id],
//     dL_dgaussian_position_total * grad_dist_weight * GLOBAL_GRADIENT_SCALE);
// #endif
#if ATTACH_F0 == true
                    atomicAdd3(
                        &params.dL_dgaussian_f0[gaussian_id],
                        dL_dgaussian_f0_total * grad_dist_weight *
                            GLOBAL_GRADIENT_SCALE);
#endif
#if ATTACH_ROUGHNESS == true
                    atomicAdd(
                        &params.dL_dgaussian_roughness[gaussian_id],
                        dL_dgaussian_roughness_total * grad_dist_weight *
                            GLOBAL_GRADIENT_SCALE);
#endif
                }
                atomicAdd4(
                    &params.dL_drotations[gaussian_id],
                    dL_drot_total * grad_dist_weight * GLOBAL_GRADIENT_SCALE);
                atomicAdd3(
                    &params.dL_dmeans[gaussian_id],
                    dL_dmean_total * grad_dist_weight * GLOBAL_GRADIENT_SCALE);

                //

                atomicAdd(
                    &params.gaussian_total_weight[gaussian_id],
                    weight); // todo assumes tile size 1
                if (step == 0) {
                    atomicAdd3(
                        &params.densification_gradient_diffuse[gaussian_id],
                        dL_dmean_total * grad_dist_weight *
                            GLOBAL_GRADIENT_SCALE);
                } else {
                    atomicAdd3(
                        &params.densification_gradient_glossy[gaussian_id],
                        dL_dmean_total / params.glossy_loss_weight *
                            grad_dist_weight / MAX_BOUNCES *
                            GLOBAL_GRADIENT_SCALE);
                }

                atomicAdd3(
                    &params.dL_dscales[gaussian_id],
                    dL_dscale_total * grad_dist_weight * GLOBAL_GRADIENT_SCALE);
#if OPTIMIZE_EXP_POWER == true
                atomicAdd(
                    &params.dL_dexp_powers[gaussian_id],
                    dL_dexp_powers_total * grad_dist_weight *
                        GLOBAL_GRADIENT_SCALE);
#endif
            }

            i--;
            hit_idx = params.all_prev_hits_for_backprop[hit_idx];
        }

        params.prev_hit_per_pixel_for_backprop[ray_id] =
            hit_idx; // * Update the starting point for the next step
    }
}
#endif