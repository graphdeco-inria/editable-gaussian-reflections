
#pragma inline
__device__ void backward_pass(
    const int step, Pixel &pixel, float3 ray_origin, float3 ray_direction, float3 throughput, const int num_hits) {
    // * Preload config parameters
    const float alpha_threshold = *params.config.alpha_threshold;
    const float exp_power = *params.config.exp_power;
    const float eps_scale_grad = *params.config.eps_scale_grad;
    int num_bounces = min(*params.config.num_bounces, MAX_BOUNCES);

#if ROUGHNESS_DOWNWEIGHT_GRAD == true
    float roughness_downweighting =
        powf(1.0f - pixel.output_roughness[max(step - 1, 0)], ROUGHNESS_DOWNWEIGHT_GRAD_POWER);
#else
    float roughness_downweighting = 1.0f;
#endif

    // * Init variables used to flow gradient from back to front
    float3 prev_gaussian_scale = make_float3(0.0f, 0.0f, 0.0f);
    float3 weighted_scale_deltas = make_float3(0.0f, 0.0f, 0.0f);
    float3 prev_gaussian_mean = make_float3(0.0f, 0.0f, 0.0f);
    float3 weighted_mean_deltas = make_float3(0.0f, 0.0f, 0.0f);
    float4 prev_gaussian_rotation = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 weighted_rotation_deltas = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 prev_gaussian_rgb = make_float3(0.0f, 0.0f, 0.0f);
    float3 weighted_rgb_deltas = make_float3(0.0f, 0.0f, 0.0f);
    float3 prev_gaussian_normal = make_float3(0.0f, 0.0f, 0.0f);
    float3 weighted_normal_deltas = make_float3(0.0f, 0.0f, 0.0f);
    float3 prev_gaussian_f0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 weighted_f0_deltas = make_float3(0.0f, 0.0f, 0.0f);
    float prev_gaussian_roughness = 0.0f;
    float weighted_roughness_deltas = 0.0f;
    float prev_gaussian_depth = 0.0f;
    float weighted_depth_deltas = 0.0f;

    uint32_t last_hit_idx;
    int i = 0;
    for (auto hit_idx : params.ppll_backward.pixel_view(pixel.id)) {
        last_hit_idx = hit_idx;
        if (i == num_hits) {
            // * Update the starting point for the next step
            params.ppll_backward.head_per_pixel[pixel.id] = last_hit_idx;
            return;
        }
        i++;

        // * Read all PPLL data
        uint32_t gaussian_id = params.ppll_backward.gaussian_ids[hit_idx];
        float3 local_hit = params.ppll_backward.local_hits[hit_idx];
        float transmittance = params.ppll_backward.transmittances[hit_idx];
        float alpha = params.ppll_backward.alphas[hit_idx];
        float gaussval = params.ppll_backward.gaussvals[hit_idx];
        float distance = params.ppll_backward.distances[hit_idx];

        // * Read all gaussian data
        float3 gaussian_rgb;
        float3 gaussian_normal;
        float3 gaussian_f0;
        float gaussian_roughness;
        if (step == 0) {
            gaussian_rgb = read_rgb(params, gaussian_id);
            gaussian_normal = read_normal(params, gaussian_id);
            gaussian_f0 = read_f0(params, gaussian_id);
            gaussian_roughness = read_roughness(params, gaussian_id);
        } else {
            gaussian_rgb = read_rgb(params, gaussian_id);
        }
        float opacity = read_opacity(params, gaussian_id);
        float3 scaling = read_scale(params, gaussian_id);
        float4 rotation_unnormalized = params.gaussians.rotation[gaussian_id];
        float4 rotation = normalize_act(rotation_unnormalized);
        float gaussian_depth = distance;

        // * Fetch the transform matrices
        const float4 *world_to_local = optixGetInstanceInverseTransformFromHandle(
            optixGetInstanceTraversableFromIAS(params.bvh_handle, gaussian_id));
        const float4 *local_to_world =
            optixGetInstanceTransformFromHandle(optixGetInstanceTraversableFromIAS(params.bvh_handle, gaussian_id));

        // * Output buffer gradient
        int num_pixels = 1; // * Deliberately avoid averaging over all pixels (seemed more stable)
        float3 dL_doutput_rgb = make_float3(0.0f, 0.0f, 0.0f);
        float3 dL_doutput_diffuse = make_float3(0.0f, 0.0f, 0.0f);
        float3 dL_doutput_specular = make_float3(0.0f, 0.0f, 0.0f);
        float dL_doutput_depth = 0.0f;
        float3 dL_doutput_normal = make_float3(0.0f, 0.0f, 0.0f);
        float3 dL_doutput_f0 = make_float3(0.0f, 0.0f, 0.0f);
        float dL_doutput_roughness = 0.0f;
        if (step == 0) {
            dL_doutput_rgb = 2.0f / 3.0f * sign(pixel.output_rgb[0] - pixel.target_diffuse) *
                             *params.config.loss_weight_diffuse / num_pixels;
            dL_doutput_depth = 2.0f / 1.0f * sign(pixel.output_depth[0] - pixel.target_depth) *
                               *params.config.loss_weight_depth / num_pixels;
            dL_doutput_normal = 2.0f / 3.0f * sign(pixel.output_normal[0] - pixel.target_normal) *
                                *params.config.loss_weight_normal / num_pixels;
            dL_doutput_f0 =
                2.0f / 3.0f * sign(pixel.output_f0[0] - pixel.target_f0) * *params.config.loss_weight_f0 / num_pixels;
            dL_doutput_roughness = 2.0f / 1.0f * sign(pixel.output_roughness[0] - pixel.target_roughness) *
                                   *params.config.loss_weight_roughness / num_pixels;
        } else {
            float3 output_specular = make_float3(0.0f, 0.0f, 0.0f);
            for (int j = 1; j < num_bounces + 1; j++) {
                output_specular += pixel.output_rgb[j];
            }
            dL_doutput_rgb = 2.0f / 3.0f * sign(output_specular - pixel.target_specular) *
                             *params.config.loss_weight_specular / num_pixels * roughness_downweighting;
            dL_doutput_rgb *= throughput;
        }

        // * Color gradient
        float weight = transmittance / (1.0 - alpha) * alpha;
        float3 dL_dgaussian_rgb = backward_act_for_rgb(dL_doutput_rgb * weight, gaussian_rgb);
        float3 dL_dgaussian_normal = backward_act_for_normal(dL_doutput_normal * weight, gaussian_normal);
        float3 dL_dgaussian_f0 = backward_act_for_f0(dL_doutput_f0 * weight, gaussian_f0);
        float dL_dgaussian_roughness = backward_act_for_roughness(dL_doutput_roughness * weight, gaussian_roughness);

        // * Flow gradients from previous gaussian behind this one
        if (step == 0) {
            weighted_rgb_deltas += (gaussian_rgb - prev_gaussian_rgb) * transmittance;
            prev_gaussian_rgb = gaussian_rgb;
            weighted_normal_deltas += (gaussian_normal - prev_gaussian_normal) * transmittance;
            prev_gaussian_normal = gaussian_normal;
            weighted_f0_deltas += (gaussian_f0 - prev_gaussian_f0) * transmittance;
            prev_gaussian_f0 = gaussian_f0;
            weighted_roughness_deltas += (gaussian_roughness - prev_gaussian_roughness) * transmittance;
            prev_gaussian_roughness = gaussian_roughness;
            weighted_depth_deltas += (gaussian_depth - prev_gaussian_depth) * transmittance;
            prev_gaussian_depth = gaussian_depth;
        } else {
            weighted_rgb_deltas += (gaussian_rgb - prev_gaussian_rgb) * transmittance;
            prev_gaussian_rgb = gaussian_rgb;
        }

        // * Alpha gradient
        float dL_dalpha = 0.0f;
        float tmp1 = 1.0f / (1.0f - alpha);
        dL_dalpha += dot(weighted_rgb_deltas * tmp1, dL_doutput_rgb);
        dL_dalpha += dot(weighted_normal_deltas * tmp1, dL_doutput_normal);
        dL_dalpha += dot(weighted_f0_deltas * tmp1, dL_doutput_f0);
        dL_dalpha += (weighted_roughness_deltas * tmp1) * dL_doutput_roughness;
        dL_dalpha += (weighted_depth_deltas * tmp1) * dL_doutput_depth;

        float tmp2 = -((pixel.output_transmittance[step] - pixel.output_total_transmittance[step]) / (1.0f - alpha));
        dL_dalpha += tmp2 * dot(pixel.remaining_rgb[step], dL_doutput_rgb);
        dL_dalpha += tmp2 * dot(pixel.remaining_normal[step], dL_doutput_normal);
        dL_dalpha += tmp2 * dot(pixel.remaining_f0[step], dL_doutput_f0);
        dL_dalpha += tmp2 * (pixel.remaining_roughness[step] * dL_doutput_roughness);
        dL_dalpha += tmp2 * (pixel.remaining_depth[step] * dL_doutput_depth);

        // * Opacity gradient
        float dL_dgaussian_opacity = MAX_ALPHA * dL_dalpha * gaussval;
        dL_dgaussian_opacity = backward_sigmoid_act(dL_dgaussian_opacity, opacity);

        // * Transform gradient
        float dL_dgaussval = MAX_ALPHA * dL_dalpha * opacity;
        float sq_norm = dot(local_hit, local_hit);
        float dL_dsq_norm = gaussval * powf(sq_norm, exp_power - 1.0f);
        float3 dL_dx_local = -local_hit * dL_dsq_norm * dL_dgaussval;

        // * World hit point gradient
        float scaling_factor = compute_scaling_factor(opacity, alpha_threshold, exp_power);
        float3 dL_dx_world =
            make_float3(
                dot(make_float3(world_to_local[0].x, world_to_local[1].x, world_to_local[2].x), dL_dx_local),
                dot(make_float3(world_to_local[0].y, world_to_local[1].y, world_to_local[2].y), dL_dx_local),
                dot(make_float3(world_to_local[0].z, world_to_local[1].z, world_to_local[2].z), dL_dx_local)) *
            scaling_factor;

        // * Local to world matrix gradient
        float3 dL_dl2w_0 = -dL_dx_world.x * local_hit;
        float3 dL_dl2w_1 = -dL_dx_world.y * local_hit;
        float3 dL_dl2w_2 = -dL_dx_world.z * local_hit;

        // * Mean gradient
        float3 dL_dgaussian_mean = -dL_dx_world;

        // * Scaling gradient
        float3 rot_0 = make_float3(local_to_world[0]) / (scaling * scaling_factor + eps_scale_grad);
        float3 rot_1 = make_float3(local_to_world[1]) / (scaling * scaling_factor + eps_scale_grad);
        float3 rot_2 = make_float3(local_to_world[2]) / (scaling * scaling_factor + eps_scale_grad);
        float3 dL_dgaussian_scale =
            backward_exp_act(dL_dl2w_0 * rot_0 + dL_dl2w_1 * rot_1 + dL_dl2w_2 * rot_2, scaling);

        // * Rotation matrix gradient
        float3 dL_drot_0 = dL_dl2w_0 * scaling;
        float3 dL_drot_1 = dL_dl2w_1 * scaling;
        float3 dL_drot_2 = dL_dl2w_2 * scaling;

        // * Rotation quaternion gradient
        float r = rotation.x;
        float x = rotation.y;
        float y = rotation.z;
        float z = rotation.w;
        float dL_dr =
            (2.f * x * (dL_drot_2.y - dL_drot_1.z) + 2.f * y * (dL_drot_0.z - dL_drot_2.x) +
             2.f * z * (dL_drot_1.x - dL_drot_0.y));
        float dL_dx =
            (-4.f * x * (dL_drot_1.y + dL_drot_2.z) + 2.f * y * (dL_drot_0.y + dL_drot_1.x) +
             2.f * z * (dL_drot_0.z + dL_drot_2.x) + 2.f * r * (dL_drot_2.y - dL_drot_1.z));
        float dL_dy =
            (2.f * x * (dL_drot_0.y + dL_drot_1.x) - 4.f * y * (dL_drot_0.x + dL_drot_2.z) +
             2.f * z * (dL_drot_1.z + dL_drot_2.y) + 2.f * r * (dL_drot_0.z - dL_drot_2.x));
        float dL_dz =
            (2.f * x * (dL_drot_0.z + dL_drot_2.x) + 2.f * y * (dL_drot_1.z + dL_drot_2.y) -
             4.f * z * (dL_drot_0.x + dL_drot_1.y) + 2.f * r * (dL_drot_1.x - dL_drot_0.y));
        float4 dL_dgaussian_rotation =
            backward_normalize_act(make_float4(dL_dr, dL_dx, dL_dy, dL_dz), rotation_unnormalized, rotation);

        // * Flush to memory
        atomicAddX(&params.gaussians.dL_dopacity[gaussian_id], dL_dgaussian_opacity);
        atomicAddX(&params.gaussians.dL_dscale[gaussian_id], dL_dgaussian_scale);
        atomicAddX(&params.gaussians.dL_dmean[gaussian_id], dL_dgaussian_mean);
        atomicAddX(&params.gaussians.dL_drotation[gaussian_id], dL_dgaussian_rotation);
        atomicAddX(&params.gaussians.dL_drgb[gaussian_id], dL_dgaussian_rgb);
        if (step == 0) {
            atomicAddX(&params.gaussians.dL_dnormal[gaussian_id], dL_dgaussian_normal);
            atomicAddX(&params.gaussians.dL_df0[gaussian_id], dL_dgaussian_f0);
            atomicAddX(&params.gaussians.dL_droughness[gaussian_id], dL_dgaussian_roughness);
        }
        atomicAdd(&params.gaussians.total_weight[gaussian_id], weight);
    }
}