#include "params.h"

#include "utils/common.h"
#include "utils/helpers.cu"

#include "backward_pass.cu"
#include "forward_pass.cu"

extern "C" __global__ void __intersection__gaussian() {
    // * Fetch config
    bool grads_enabled = (bool)optixGetPayload_2();
    float alpha_threshold = __uint_as_float(optixGetPayload_3());
    float exp_power = __uint_as_float(optixGetPayload_4());
    float backfacing_max_dist = __uint_as_float(optixGetPayload_5());
    float backfacing_invalid_normal_threshold = __uint_as_float(optixGetPayload_6());
    uint32_t num_traversed_per_pixel = optixGetPayload_7();

    // * Fetch ray data
    float3 local_origin = optixGetObjectRayOrigin();
    float3 local_direction = optixGetObjectRayDirection();

    // * Load gaussian data
    const uint32_t gaussian_id = optixGetInstanceIndex();
    float opacity = read_opacity(params, gaussian_id);
    float3 scale = read_scale(params, gaussian_id);

    // * Compute pixel index
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
    uint32_t pixel_id = idx.y * params.image_width + idx.x;

    // * Update stats
    optixSetPayload_7(num_traversed_per_pixel + 1);

    // * Reject gaussians behind ray
    if (dot(local_origin, local_direction) > 0.0) {
        return;
    }

    // * Compute the hit point along the ray
    float norm = length(local_direction);
    local_direction /= norm;
    float local_hit_distance_along_ray = dot(-local_origin, local_direction);
    float world_distance = local_hit_distance_along_ray / norm;
    float3 local_hit_unscaled = local_origin + local_hit_distance_along_ray * local_direction;

    // * Clip the gaussian at the alpha threshold
    float sq_dist = dot(local_hit_unscaled, local_hit_unscaled);
    if (sq_dist > 1.0f) {
        return;
    }

    // * Reject backfacing normals when not a primary ray
    int step = optixGetPayload_0();
    if (step != 0 && world_distance < backfacing_max_dist) {
        float3 gaussian_normal = read_normal(params, gaussian_id);
        if (length(gaussian_normal) > backfacing_invalid_normal_threshold &&
            dot(gaussian_normal, local_direction) > 0.0f) {
            return;
        }
    }

    // * Compute alpha value
    float3 local_hit =
        local_hit_unscaled * compute_scaling_factor(opacity, alpha_threshold, exp_power);
    float gaussval = eval_gaussian(local_hit, exp_power);
    float alpha = compute_alpha(gaussval, opacity, alpha_threshold);

    // * Compute the exact total transmittance for the ray
    float full_T = __uint_as_float(optixGetPayload_1());
    full_T *= 1.0 - alpha;
    optixSetPayload_1(__float_as_uint(full_T));

    // * Log all hits to per-pixel linked list
    params.ppll_forward.insert(
        grads_enabled, pixel_id, gaussian_id, world_distance, local_hit, gaussval, alpha);
}

extern "C" __global__ void __raygen__rg() {
    // * Compute pixel index
    int num_pixels = params.image_width * params.image_height;
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
    uint32_t pixel_id = idx.y * params.image_width + idx.x;
    float eps_ray_surface_offset = *params.config.eps_ray_surface_offset;
    float eps_min_roughness = *params.config.eps_min_roughness;
    float reflection_invalid_normal_threshold = *params.config.reflection_invalid_normal_threshold;
    int num_bounces = min(*params.config.num_bounces, MAX_BOUNCES);

    // * Update random seed based on iteration
    unsigned int seed = tea<4>(pixel_id, *params.metadata.total_num_calls);

    // * Compute the ray coordinates
    float3 ray_origin = *params.camera.origin;
    float3 ray_direction = params.camera.compute_primary_ray_direction(
        *params.config.jitter_primary_rays, idx, dim, seed);

    // * Init output buffers for all bounces (steps)
    Pixel pixel(pixel_id);
    int num_hits[MAX_BOUNCES + 1];
    fill_array(num_hits, MAX_BOUNCES + 1, 0);

    // * Track total steps taken to account for early termination
    int total_effective_steps = 0;

    // *** Forward pass
    for (int step = 0; step < num_bounces + 1; step++) {
        total_effective_steps++;

        // * Compute the indicent radiance
        forward_pass(step, pixel, ray_origin, ray_direction, num_hits[step]);
        params.ppll_forward.reset(pixel.id);

        // * Multiply step color by the throughput of the previous step
        if (step > 0) {
            pixel.output_rgb[step] = pixel.output_rgb[step] * pixel.output_throughput[step - 1];
        }

        // * Post-process accumulated normal and roughness
        float3 unnormalized_normal = pixel.output_normal[step];
        float3 effective_normal = normalize(unnormalized_normal);
        float effective_roughness =
            max(pixel.output_roughness[step],
                eps_min_roughness); // * For stability avoid exactly 0 roughness

        // * Terminate path if the accumulated normal is invalid
        if (length(unnormalized_normal) < reflection_invalid_normal_threshold) {
            break;
        }

        // * Compute reflection ray for the following step
        float3 effective_position = ray_origin + pixel.output_depth[step] * ray_direction;
        float3 next_direction = sample_vndf(
            effective_normal,
            -ray_direction,
            effective_roughness,
            make_float2(rnd(seed), rnd(seed)));
        float3 next_origin = effective_position + eps_ray_surface_offset * next_direction;

        // * Update throughput in a cumulative product
        float3 effective_F0 = pixel.output_f0[step];
        if (step > 0) {
            pixel.output_throughput[step] = pixel.output_throughput[step - 1];
        }

        pixel.output_throughput[step] *= principled_specular(effective_normal, -ray_direction, next_direction, effective_F0, effective_roughness) / fmaxf(sample_vndf_pdf(effective_normal, -ray_direction, next_direction, effective_roughness), 1e-12f); 

        // * Update ray and log it for debugging
        ray_origin = next_origin;
        ray_direction = next_direction;
        pixel.output_ray_origin[step] = ray_origin;
        pixel.output_ray_direction[step] = ray_direction;
    }

    // * Total all rgb passes into the final image (on which denoising is
    // applied)
    for (int step = 0; step < num_bounces + 1; step++) {
        pixel.output_final += pixel.output_rgb[step];
    }

    // *** Backward pass
    if (*params.metadata.grads_enabled) {
        params.framebuffer.fetch_targets(pixel);
        for (int step = total_effective_steps - 1; step >= 0; step--) {
            if (num_hits[step] > 0) {
                float3 throughput =
                    step == 0 ? make_float3(1.0f) : pixel.output_throughput[step - 1];
                backward_pass(step, pixel, ray_origin, ray_direction, throughput, num_hits[step]);
            }
        }
    } else {
        // * Only write outputs when gradients are disabled since its a bit slow
        if (*params.config.accumulate_samples) {
            params.framebuffer.update_accumulators(pixel);
        }
        params.framebuffer.write_outputs(pixel);
    }

    // * Update the initial seed so the next call has a different one
    params.metadata.random_seeds[pixel_id] = seed;
}
