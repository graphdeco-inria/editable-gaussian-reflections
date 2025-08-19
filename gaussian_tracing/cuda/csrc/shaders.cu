#include "params.h"
#include "utils/vec_math.h"
#include <iostream>
#include <random>

#include "utils/activations.cu"
#include "utils/ggx_brdf.h"
#include "utils/kernel.cu"
#include "utils/misc.cu"
#include "utils/random.h"

#if USE_GT_BRDF == false
__device__ float2 bilinear_sample_LUT(float2 uv) {
    uv = clamp(uv, make_float2(0.0f, 0.0f), make_float2(1.0f, 1.0f));
    float2 uv_scaled = uv * (LUT_SIZE - 1);
    int x0 = floor(uv_scaled.x);
    int y0 = floor(uv_scaled.y);
    int x1 = min(x0 + 1, LUT_SIZE - 1);
    int y1 = min(y0 + 1, LUT_SIZE - 1);
    float2 uv_fract = uv_scaled - make_float2(x0, y0);
    float2 value00 = params.lut[y0 * LUT_SIZE + x0];
    float2 value01 = params.lut[y1 * LUT_SIZE + x0];
    float2 value10 = params.lut[y0 * LUT_SIZE + x1];
    float2 value11 = params.lut[y1 * LUT_SIZE + x1];
    float2 value0 = value00 + uv_fract.y * (value01 - value00);
    float2 value1 = value10 + uv_fract.y * (value11 - value10);
    return value0 + uv_fract.x * (value1 - value0);
}
#endif

#if USE_POLYCAGE == true
extern "C" __global__ void __anyhit__ah() {
#else
extern "C" __global__ void __intersection__gaussian() {
#endif
#if USE_POLYCAGE == true
#define RETURN                                                                 \
    optixIgnoreIntersection();                                                 \
    return;
#else
#define RETURN return;
#endif

#if DEBUG_SINGLE_EMPTY_RAYTRACE == true
    RETURN
#endif

    // * Intersect the ray with the gaussian's maximal value
    const uint32_t gaussian_id = optixGetInstanceIndex();
    float3 local_origin = optixGetObjectRayOrigin();
    float3 local_direction = optixGetObjectRayDirection();

    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
    uint32_t ray_id =
        (idx.y * TILE_SIZE) * params.image_width + idx.x * TILE_SIZE;

#if SAVE_HIT_STATS == true
    atomicAdd(&params.num_traversed_per_pixel[ray_id], 1);
#endif

#if FUSED_MESH == true
    {
        // todo fix the gaussian id
        // int hit_idx = atomicAdd(params.total_hits, 1);
        // params.all_gaussian_ids[hit_idx] = gaussian_id;
        // params.all_distances[hit_idx] = sorting_distance;
        // params.all_prev_hits[hit_idx] = params.prev_hit_per_pixel[ray_id];
        // params.prev_hit_per_pixel[ray_id] = hit_idx;
        params.output_rgb[ray_id] = READ_RGB(gaussian_id);
    }
    RETURN
#endif

    float opacity = READ_OPACITY(gaussian_id);
    float3 scale = READ_SCALE(gaussian_id);
    if (BBOX_PADDING > 0.0f) {
#if OPTIMIZE_EXP_POWER == true
        float p = params.gaussian_exp_power[gaussian_id];
#else
        float p = params.exp_power;
#endif
        float3 scaling =
            scale * compute_scaling_factor(opacity, params.alpha_threshold, p);
        scaling = scaling + ANTIALIASING;
        local_origin = local_origin * (BBOX_PADDING + scaling) / scaling;
        local_direction = local_direction * (BBOX_PADDING + scaling) / scaling;
    }

    float norm = length(local_direction);
    local_direction /= norm;
    float local_hit_distance_along_ray = dot(-local_origin, local_direction);

// * Compute the hit point along the ray
#if GLOBAL_SORT == true
    float m[12];
    optixGetObjectToWorldTransformMatrix(m);
    float3 gaussian_mean = make_float3(m[3], m[7], m[11]);
    float3 reflected_origin = make_float3(
        __uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5()),
        __uint_as_float(optixGetPayload_6()));
    float sorting_distance =
        dot(gaussian_mean - reflected_origin, optixGetWorldRayDirection());
#else
    float sorting_distance = local_hit_distance_along_ray / norm;
#endif
    float3 local_hit_unscaled =
        (local_origin + local_hit_distance_along_ray * local_direction) *
        BB_SHRINKAGE;

    // * Clip the gaussian at params.alpha_threshold, taking bounding box scale
    // into account todo: if end up implementing tiling, it is possible to
    // optimize further considering the worse ray of the tile, or by using
    // capsules
    float sq_dist;
    if (BBOX_PADDING > 0.0f) {
        // Clip against the full bbox after padding its size
        float3 local_origin_fullbox = optixGetObjectRayOrigin();
        float3 local_direction_fullbox = optixGetObjectRayDirection();
        local_direction_fullbox /= length(local_direction_fullbox);
        float local_hit_distance_along_ray_fullbox =
            dot(-local_origin_fullbox, local_direction_fullbox);
        float3 local_hit_unscaled_fullbox =
            local_origin_fullbox +
            local_hit_distance_along_ray_fullbox * local_direction_fullbox;
        sq_dist = dot(local_hit_unscaled_fullbox, local_hit_unscaled_fullbox);
    } else {
        sq_dist = dot(local_hit_unscaled, local_hit_unscaled);
    }
#if SQUARE_KERNEL == false
    if (sq_dist > 1.0f) {
        RETURN
    }
#endif

#if STRICT_REJECT_GAUSSIANS_BEHIND_RAY == true
    if (dot(optixGetObjectRayOrigin(), optixGetObjectRayDirection()) > 0.0) {
        RETURN
    }
#endif

#if STORAGE_MODE == NO_STORAGE
    float tmin = optixGetRayTmin();
    if (sorting_distance <= tmin) {
        RETURN
    }
#endif

#if SKIP_BACKFACING_NORMALS == true && ATTACH_NORMALS == true
    int step = optixGetPayload_2();
    if (step != 0 && sorting_distance < SKIP_BACKFACING_MAX_DIST) {
        float3 gaussian_normal = params.gaussian_normal[gaussian_id];
        if (length(gaussian_normal) >
                SKIP_BACKFACING_REFLECTION_VALID_NORMAL_MIN_NORM &&
            dot(gaussian_normal, local_direction) > 0.0f) {
            RETURN
        }
    }
#endif

#if TILE_SIZE > 1
#if STORAGE_MODE == NO_STORAGE
    printf("Error! NO_STORAGE mode not supported with TILE_SIZE > 1\n");
#endif
#if LOG_ALL_HITS != true
    printf("Error! Only LOG_ALL_HITS mode supported with TILE_SIZE > 1\n");
#endif
    if (BBOX_PADDING != 0.0f) {
        printf("Error! BBOX_PADDING not supported yet!\n");
    }

    // * Log the shared info for all subpixels
    int hit_idx = atomicAdd(params.total_hits, 1);
    params.all_gaussian_ids[hit_idx] = gaussian_id;
    params.all_distances[hit_idx] = sorting_distance;
#if NUM_SLABS > 1
    params.all_slab_idx[hit_idx] = optixGetPayload_2();
#endif
    params.all_prev_hits[hit_idx] = params.prev_hit_per_pixel[ray_id];
    params.prev_hit_per_pixel[ray_id] = hit_idx;

// * Compute all alpha values
#if OPTIMIZE_EXP_POWER == true
    float exp_power =
        params
            .gaussian_exp_power[gaussian_id]; // todo reuse existing read above
#else
    float exp_power = params.exp_power;
#endif

    float view_size = tan(*params.vertical_fov_radians / 2);
    float aspect_ratio = float(dim.x) / float(dim.y);

    for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
        int ki = k / TILE_SIZE;
        int kj = k % TILE_SIZE;
        float _y = (idx.y * TILE_SIZE + ki) / (float(dim.y) * TILE_SIZE) +
                   0.5f / (float(dim.y) * TILE_SIZE);
        float _x = (idx.x * TILE_SIZE + kj) / (float(dim.x) * TILE_SIZE) +
                   0.5f / (float(dim.x) * TILE_SIZE);
        float y = view_size * (1.0f - 2.0f * _y);
        float x = aspect_ratio * view_size * (2.0f * _x - 1.0f);
        float3 subpixel_direction = normalize(
            params.camera_rotation_w2c[0] * x +
            params.camera_rotation_w2c[1] * y - params.camera_rotation_w2c[2]);

        // use a direct api call
        float3 local_subpixel_direction =
            optixTransformVectorFromWorldToObjectSpace(subpixel_direction);
        float scaling_factor =
            compute_scaling_factor(opacity, params.alpha_threshold, exp_power);

        if (BBOX_PADDING > 0.0f) {
            float3 scaling = scale * scaling_factor;
            float3 padded_scaling = scaling * scaling_factor;
            padded_scaling = padded_scaling + ANTIALIASING;
            local_subpixel_direction = local_subpixel_direction *
                                       (BBOX_PADDING + padded_scaling) /
                                       padded_scaling;
        }

        float norm = length(local_subpixel_direction);
        local_subpixel_direction /= norm;

        auto local_hit_distance_along_ray =
            dot(-local_origin, local_subpixel_direction);
        float3 local_hit_unscaled =
            local_origin +
            local_hit_distance_along_ray * local_subpixel_direction;

        float sq_dist = dot(local_hit_unscaled, local_hit_unscaled);
        float gaussval = 0.0f;
        float alpha = 0.0f;
        float3 local_hit = make_float3(0.0f, 0.0f, 0.0f);
        if (sq_dist <= 1.0f) {
            local_hit = local_hit_unscaled * scaling_factor;
            gaussval = eval_gaussian(local_hit, exp_power);
            alpha = compute_alpha(gaussval, opacity, params.alpha_threshold);
        }

#if SQUARE_KERNEL == true && TILE_SIZE == 1
        if (alpha < ALPHA_THRESHOLD) {
            RETURN
        }
#endif

        // * Log the subpixels
        params.all_alphas[TILE_SIZE * TILE_SIZE * hit_idx + k] = alpha;
#if BACKWARDS_PASS == true
        if (*params
                 .grads_enabled) { // todo check impact of this on forward pass
            params.all_gaussvals[TILE_SIZE * TILE_SIZE * hit_idx + k] =
                gaussval;
            params.all_local_hits[TILE_SIZE * TILE_SIZE * hit_idx + k] =
                local_hit;
        }
#endif

#if TILE_SIZE == 2
        switch (k) {
        case 0: {
            float full_T = __uint_as_float(optixGetPayload_3());
            full_T *= 1.0 - alpha;
            optixSetPayload_1(__float_as_uint(full_T));
            break;
        }
        case 1: {
            float full_T = __uint_as_float(optixGetPayload_4());
            full_T *= 1.0 - alpha;
            optixSetPayload_2(__float_as_uint(full_T));
            break;
        }
        case 2: {
            float full_T = __uint_as_float(optixGetPayload_5());
            full_T *= 1.0 - alpha;
            optixSetPayload_3(__float_as_uint(full_T));
            break;
        }
        case 3: {
            float full_T = __uint_as_float(optixGetPayload_6());
            full_T *= 1.0 - alpha;
            optixSetPayload_4(__float_as_uint(full_T));
            break;
        }
        }
#elif TILE_SIZE == 3
        switch (k) {
        case 0: {
            float full_T = __uint_as_float(optixGetPayload_3());
            full_T *= 1.0 - alpha;
            optixSetPayload_1(__float_as_uint(full_T));
            break;
        }
        case 1: {
            float full_T = __uint_as_float(optixGetPayload_4());
            full_T *= 1.0 - alpha;
            optixSetPayload_2(__float_as_uint(full_T));
            break;
        }
        case 2: {
            float full_T = __uint_as_float(optixGetPayload_5());
            full_T *= 1.0 - alpha;
            optixSetPayload_3(__float_as_uint(full_T));
            break;
        }
        case 3: {
            float full_T = __uint_as_float(optixGetPayload_6());
            full_T *= 1.0 - alpha;
            optixSetPayload_4(__float_as_uint(full_T));
            break;
        }
        case 4: {
            float full_T = __uint_as_float(optixGetPayload_7());
            full_T *= 1.0 - alpha;
            optixSetPayload_5(__float_as_uint(full_T));
            break;
        }
        case 5: {
            float full_T = __uint_as_float(optixGetPayload_8());
            full_T *= 1.0 - alpha;
            optixSetPayload_6(__float_as_uint(full_T));
            break;
        }
        case 6: {
            float full_T = __uint_as_float(optixGetPayload_9());
            full_T *= 1.0 - alpha;
            optixSetPayload_7(__float_as_uint(full_T));
            break;
        }
        case 7: {
            float full_T = __uint_as_float(optixGetPayload_10());
            full_T *= 1.0 - alpha;
            optixSetPayload_8(__float_as_uint(full_T));
            break;
        }
        case 8: {
            float full_T = __uint_as_float(optixGetPayload_11());
            full_T *= 1.0 - alpha;
            optixSetPayload_9(__float_as_uint(full_T));
            break;
        }
        }
#endif
    }
#else
#if OPTIMIZE_EXP_POWER == true
    float p = params.gaussian_exp_power[gaussian_id];
#else
    float p = params.exp_power;
#endif
    float3 local_hit =
        local_hit_unscaled *
        compute_scaling_factor(opacity, params.alpha_threshold, p);
    float gaussval = eval_gaussian(local_hit, p);
    float alpha = compute_alpha(gaussval, opacity, params.alpha_threshold);

// * Compute the total transmittance for the ray accurately
#if STORAGE_MODE == NO_STORAGE
    params.full_T[ray_id] *= 1.0 - alpha;
#else
    float full_T = __uint_as_float(optixGetPayload_3());
    full_T *= 1.0 - alpha;
    optixSetPayload_3(__float_as_uint(full_T));
#endif

#if USE_CLUSTERING == true
    float half_chord_length =
        sqrtf(1.0f - dot(local_hit, local_hit)) / local_hit_distance_along_ray;
#endif

// * Log all hits if required by configuration
#if LOG_ALL_HITS == true
    int hit_idx = atomicAdd(params.total_hits, 1);
    // if (alpha >= params.alpha_threshold) { //? why was there a check again
    // here?
    params.all_gaussian_ids[hit_idx] = gaussian_id;
    params.all_distances[hit_idx] = sorting_distance;
#if USE_CLUSTERING == true
    params.all_half_chord_lengths[hit_idx] = half_chord_length;
#endif
#if RECOMPUTE_ALPHA_IN_FORWARD_PASS == false
    params.all_alphas[hit_idx] = alpha;
#if BACKWARDS_PASS == true
    if (*params.grads_enabled) { // todo check impact of this on forward pass
        params.all_gaussvals[hit_idx] = gaussval;
        params.all_local_hits[hit_idx] = local_hit;
    }
#endif
#endif
#if NUM_SLABS > 1
    params.all_slab_idx[hit_idx] = optixGetPayload_3();
#endif
    params.all_prev_hits[hit_idx] = params.prev_hit_per_pixel[ray_id];
    params.prev_hit_per_pixel[ray_id] = hit_idx;
    // }
#endif
#endif

// * Read payload
#if STORAGE_MODE != PER_PIXEL_LINKED_LIST
    register unsigned int floats[BUFFER_SIZE] = {
        optixGetPayload_0(),
        optixGetPayload_1(),
        optixGetPayload_2(),
        optixGetPayload_3(),
        optixGetPayload_4(),
        optixGetPayload_5(),
        optixGetPayload_6(),
        optixGetPayload_7(),
        optixGetPayload_8(),
        optixGetPayload_9(),
        optixGetPayload_10(),
        optixGetPayload_11(),
        optixGetPayload_12(),
        optixGetPayload_13(),
        optixGetPayload_14(),
        optixGetPayload_15()};
    register unsigned int gaussian_ids[BUFFER_SIZE] = {
        optixGetPayload_16(),
        optixGetPayload_17(),
        optixGetPayload_18(),
        optixGetPayload_19(),
        optixGetPayload_20(),
        optixGetPayload_21(),
        optixGetPayload_22(),
        optixGetPayload_23(),
        optixGetPayload_24(),
        optixGetPayload_25(),
        optixGetPayload_26(),
        optixGetPayload_27(),
        optixGetPayload_28(),
        optixGetPayload_29(),
        optixGetPayload_30(),
        optixGetPayload_31()};
#endif

// * Insert into the hit buffer
#if STOCHASTIC == true
    for (int i = 0; i < BUFFER_SIZE - 1; i++) {
        float u = rnd(seed);
        if (sorting_distance < unpackDistance(floats[i]) && u < alpha) {
            gaussian_ids[i] = packId(gaussian_id, 0);
            floats[i] = packFloats(sorting_distance, alpha);
        }
#if BACKWARDS_PASS == true
        // sort the array gain
        for (int j = 0; j < i; j++) {
            if (sorting_distance < unpackDistance(floats[j])) {
                auto tmp = floats[i];
                floats[i] = floats[j];
                floats[j] = tmp;
                auto tmp2 = gaussian_ids[i];
                gaussian_ids[i] = gaussian_ids[j];
                gaussian_ids[j] = tmp2;
            }
        }
#endif
    }
    params.random_seeds[ray_id] = seed;
#elif STORAGE_MODE != PER_PIXEL_LINKED_LIST
    if (sorting_distance < unpackDistance(floats[BUFFER_SIZE - 1])) {
        floats[BUFFER_SIZE - 1] = packFloats(sorting_distance, alpha);
        gaussian_ids[BUFFER_SIZE - 1] = packId(gaussian_id, 0);
    }
#pragma unroll
    for (int i = BUFFER_SIZE - 1; i > 0; i--) {
        if (unpackDistance(floats[i]) < unpackDistance(floats[i - 1])) {
            unsigned int tmp_dist = floats[i];
            unsigned int tmp_id = gaussian_ids[i];
            floats[i] = floats[i - 1];
            gaussian_ids[i] = gaussian_ids[i - 1];
            floats[i - 1] = tmp_dist;
            gaussian_ids[i - 1] = tmp_id;
        }
    }
#endif

// * Write payload
#if STORAGE_MODE != PER_PIXEL_LINKED_LIST
    optixSetPayload_0(floats[0]);
    optixSetPayload_1(floats[1]);
    optixSetPayload_2(floats[2]);
    optixSetPayload_3(floats[3]);
    optixSetPayload_4(floats[4]);
    optixSetPayload_5(floats[5]);
    optixSetPayload_6(floats[6]);
    optixSetPayload_7(floats[7]);
    optixSetPayload_8(floats[8]);
    optixSetPayload_9(floats[9]);
    optixSetPayload_10(floats[10]);
    optixSetPayload_11(floats[11]);
    optixSetPayload_12(floats[12]);
    optixSetPayload_13(floats[13]);
    optixSetPayload_14(floats[14]);
    optixSetPayload_15(floats[15]);
    optixSetPayload_16(gaussian_ids[0]);
    optixSetPayload_17(gaussian_ids[1]);
    optixSetPayload_18(gaussian_ids[2]);
    optixSetPayload_19(gaussian_ids[3]);
    optixSetPayload_20(gaussian_ids[4]);
    optixSetPayload_21(gaussian_ids[5]);
    optixSetPayload_22(gaussian_ids[6]);
    optixSetPayload_23(gaussian_ids[7]);
    optixSetPayload_24(gaussian_ids[8]);
    optixSetPayload_25(gaussian_ids[9]);
    optixSetPayload_26(gaussian_ids[10]);
    optixSetPayload_27(gaussian_ids[11]);
    optixSetPayload_28(gaussian_ids[12]);
    optixSetPayload_29(gaussian_ids[13]);
    optixSetPayload_30(gaussian_ids[14]);
    optixSetPayload_31(gaussian_ids[15]);
#endif

    RETURN
}

#if USE_POLYCAGE == true
extern "C" __global__ void __intersection__gaussian() {}
#else
#endif

#include "backward_pass.cu"
#include "forward_pass.cu"

extern "C" __global__ void __raygen__rg() {
    int num_pixels = params.image_width * params.image_height;
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
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
#if JITTER == true && MAX_BOUNCES > 0
    const float2 jitter_offset =
        make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
    // const float2 jitter_offset = *params.grads_enabled ?
    // make_float2(rnd(seed)
    // - 0.5f, rnd(seed) - 0.5f) : make_float2(0.0f, 0.0f);
#else
    const float2 jitter_offset = make_float2(0.0f, 0.0f);
#endif
    float2 idxf = make_float2(idx.x, idx.y) + jitter_offset;

    float view_size = tan(*params.vertical_fov_radians / 2);
    float aspect_ratio = float(dim.x) / float(dim.y);

#if ORTHO_CAM == true
    float x = 2.0f * ((idxf.x * TILE_SIZE + float(TILE_SIZE) / 2) /
                      (float(dim.x) * TILE_SIZE)) -
              1.0f;
    float y = 1.0f - 2.0f * ((idxf.y * TILE_SIZE + float(TILE_SIZE) / 2) /
                             (float(dim.y) * TILE_SIZE));
    float z = 1.0f;

    tile_origin =
        *params.camera_position_world + x * params.camera_rotation_w2c[0] +
        y * params.camera_rotation_w2c[1] + z * params.camera_rotation_w2c[2];
    tile_direction = -params.camera_rotation_w2c[2];
#else
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
#endif

    for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
        origin[k] = tile_origin;
#if TILE_SIZE > 1
        int ki = k / TILE_SIZE;
        int kj = k % TILE_SIZE;
        float _y = (idx.y * TILE_SIZE + ki) / (float(dim.y) * TILE_SIZE) +
                   0.5f / (float(dim.y) * TILE_SIZE);
        float _x = (idx.x * TILE_SIZE + kj) / (float(dim.x) * TILE_SIZE) +
                   0.5f / (float(dim.x) * TILE_SIZE);
        float y = view_size * (1.0f - 2.0f * _y);
        float x = aspect_ratio * view_size * (2.0f * _x - 1.0f);
        direction[k] = normalize(
            params.camera_rotation_w2c[0] * x +
            params.camera_rotation_w2c[1] * y - params.camera_rotation_w2c[2]);
        origin[k] = tile_origin;
#else
        direction[k] = tile_direction;
#endif
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
                make_float3(INIT_F0, INIT_F0, INIT_F0));
    float output_roughness[MAX_BOUNCES + 1][NUM_CLUSTERS]
                          [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                output_roughness[i][c], TILE_SIZE * TILE_SIZE, INIT_ROUGHNESS);
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
                make_float3(INIT_F0, INIT_F0, INIT_F0));
    float remaining_roughness[MAX_BOUNCES + 1][NUM_CLUSTERS]
                             [TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < MAX_BOUNCES + 1; i++)
        for (int c = 0; c < NUM_CLUSTERS; c++)
            fill_array(
                remaining_roughness[i][c],
                TILE_SIZE * TILE_SIZE,
                INIT_ROUGHNESS);
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
        froward_pass(
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

        float max_weight = 0.0f;
#if USE_CLUSTERING == true
        for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
            // randomly select a cluster based on the weights
            // cluster_weights[step][c][k] and F0
        }
#endif

        // #if REFLECTIONS_FROM_GT_GLOSSY_IRRADIANCE
        //     step = MAX_BOUNCES + 1;
        //     for (int ki = 0; ki < TILE_SIZE; ki++) for (int kj = 0; kj <
        //     TILE_SIZE; kj++) {
        //         int pixel_id = ray_id + ki * params.image_width + kj;
        //         output_rgb[1][0][pixel_id] =
        //         params.target_glossy_irradiance[pixel_id];
        //     }
        // #endif

#if MAX_BOUNCES > 0
        for (int ki = 0; ki < TILE_SIZE; ki++)
            for (int kj = 0; kj < TILE_SIZE; kj++) {
                int k = ki * TILE_SIZE + kj;
                int pixel_id = ray_id + ki * params.image_width + kj;

#if USE_CLUSTERING == true
                // int c = selected_clusters[step][k];
                int c = 0;
#else
                int c = 0;
#endif

                incident_radiance[step][k] = output_rgb[step][k];

                // * Multiply by the BRDF of the previous step
                if (step > 0) {
                    // #if USE_CLUSTERING == true
                    //     int prev_c = selected_clusters[step - 1][k];
                    // #else
                    int prev_c = 0;
                    // #endif
                    output_rgb[step][k] =
                        output_rgb[step][k] *
                        output_throughput[step - 1][prev_c][k];
                    // #if USE_CLUSTERING == true
                    //     float ww = cluster_weights[step - 1][prev_c][k]; // !
                    //     tmp and todo fails for multibounce
                    //     output_rgb[step][k] *= ww;
                    // #endif
                }

#if SURFACE_BRDF_FROM_GT_ROUGHNESS == true
                float effective_roughness = params.target_roughness[pixel_id];
#else
                float effective_roughness = output_roughness[step][c][k];
#endif
                effective_roughness = max(effective_roughness, MIN_ROUGHNESS);

#if REFLECTION_RAY_FROM_GT_NORMAL == true
                float3 unnormalized_normal = params.target_normal[pixel_id];
#else
                float3 unnormalized_normal = output_normal[step][c][k];
#endif
#if NORMALIZE_NORMAL_MAP == true
                float3 effective_normal = normalize(unnormalized_normal);
#else
                float3 effective_normal = unnormalized_normal;
#endif

#if SURFACE_BRDF_FROM_GT_F0 == true
                float3 effective_F0 = params.target_f0[pixel_id];
#else
                float3 effective_F0 = output_f0[step][c][k];
#endif

                // * Compute the BRDF for this step
                {
#if USE_GT_BRDF == true
                    output_throughput[step][c][k] =
                        params.target_brdf[pixel_id];
#else
#if SAMPLE_FROM_BRDF == false
                    // USE LUT
                    float n_dot_v = dot(-direction[k], effective_normal);
                    float2 uv = make_float2(n_dot_v, effective_roughness);
                    float2 lut_values = bilinear_sample_LUT(uv);
                    output_throughput[step][c][k] *=
                        lut_values.x * effective_F0 + lut_values.y;
#endif

                    if (step > 0) {
                        output_throughput[step][c][k] *=
                            output_throughput[step - 1][c][k];
                    }

#endif
                }

                // * Compute reflection ray for the following step
                {
#if REFLECTION_RAY_FROM_GT_POSITION == true
                    float3 effective_position =
                        params.target_position[pixel_id]; // todo tiling
#else
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
                    float3 effective_position =
                        origin[k] + output_depth[step][c][k] * direction[k];
#else
                    float3 effective_position = output_position[step][c][k];
#if PROJECT_POSITION_TO_RAY == true
                    effective_position =
                        origin[k] +
                        dot(direction[k], effective_position - origin[k]) *
                            direction[k];
#endif
#endif
#endif

                    if (REFLECTION_VALID_NORMAL_MIN_NORM > 0.0f) {
                        if (length(unnormalized_normal) <
                            REFLECTION_VALID_NORMAL_MIN_NORM) {
                            goto forward_pass_end;
                        }
                    }

#if SAMPLE_FROM_BRDF == true
                    float3 next_direction = sample_cook_torrance(
                        effective_normal,
                        -direction[k],
                        effective_roughness,
                        make_float2(rnd(seed), rnd(seed)));

#if USE_GT_BRDF == false
#if INCLUDE_BRDF_WEIGHT == true
#if USE_LUT == true
                    float n_dot_v = dot(-direction[k], effective_normal);
                    float2 uv = make_float2(n_dot_v, effective_roughness);
                    float2 lut_values = bilinear_sample_LUT(uv);
                    output_lut_values[step][c][k] = lut_values;
                    output_throughput[step][c][k] *=
                        lut_values.x * effective_F0 + lut_values.y;
#else
                    output_throughput[step][c][k] *= cook_torrance_weight(
                        effective_normal,
                        -direction[k],
                        next_direction,
                        effective_roughness,
                        effective_F0);
#endif
#else
                    if (effective_F0.x == 0.0f && effective_F0.y == 0.0f &&
                        effective_F0.z == 0.0f) {
                        output_throughput[step][c][k] *= 0.0f;
                    }
#endif
#endif
#else
                    float3 next_direction =
                        reflect(direction[k], effective_normal);
#endif

                    float3 next_origin =
                        effective_position + SURFACE_EPS * next_direction;
                    reflected_origin = next_origin;
                    // reflected_origin = effective_position - next_direction *
                    // length(effective_position - tile_origin);

                    origin[k] = next_origin;
                    direction[k] = next_direction;
                    tile_origin = next_origin;       // tmp
                    tile_direction = next_direction; // tmp

#if SAVE_RAY_IMAGES == true
                    params.output_ray_origin[pixel_id + num_pixels * step] =
                        origin[k];
                    params.output_ray_direction[pixel_id + num_pixels * step] =
                        direction[k];
#endif
                }
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
                // todo refactor
                // #if USE_CLUSTERING == true
                //     int max_c = 0;
                //     float max_weight = 0.0f;
                //     for (int c = 0; c < NUM_CLUSTERS; c++) {
                //         float weight = cluster_weights[s][c][tile_id];
                //         if (weight > max_weight) {
                //             max_weight = weight;
                //             max_c = c;
                //         }
                //     }
                // #else
                int max_c = 0;
                // #endif

                int pixel_id =
                    ray_id + ki * params.image_width + kj + num_pixels * s;
                params.output_rgb[pixel_id] = output_rgb[s][tile_id];
                params.output_t[pixel_id] = output_t[s][tile_id];
                if (!*params.grads_enabled) {
#if SAVE_ALL_MAPS == true
                    params.output_incident_radiance[pixel_id] =
                        incident_radiance[s][tile_id];
#if ATTACH_NORMALS == true
                    params.output_normal[pixel_id] =
                        output_normal[s][max_c][tile_id];
#endif
#if ATTACH_POSITION == true || POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
                    params.output_position[pixel_id] =
                        output_position[s][max_c][tile_id];
#endif
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
                    params.output_depth[pixel_id] =
                        output_depth[s][max_c][tile_id];
#endif
#if ATTACH_F0 == true
                    params.output_f0[pixel_id] = output_f0[s][max_c][tile_id];
#endif
#if ATTACH_ROUGHNESS == true
                    params.output_roughness[pixel_id] =
                        output_roughness[s][max_c][tile_id];
#endif
#if MAX_BOUNCES > 0
                    params.output_brdf[pixel_id] =
                        output_throughput[s][max_c][tile_id];
#endif
#endif
#if RENDER_DISTORTION == true
                    params.output_distortion[pixel_id] =
                        output_distortion[s][max_c][tile_id];
#endif
#if SAVE_LOD_IMAGES == true
                    params.output_lod_mean[pixel_id] =
                        output_lod_mean[s][max_c][tile_id];
                    params.output_lod_scale[pixel_id] =
                        output_lod_scale[s][max_c][tile_id];
                    params.output_ray_lod[pixel_id] =
                        output_ray_lod[s][max_c][tile_id];
#endif
                }
            }

            // Write the final pass
            int pixel_id = ray_id + ki * params.image_width + kj +
                           num_pixels * (MAX_BOUNCES + 1);
            params.output_rgb[pixel_id] = output_rgb[MAX_BOUNCES + 1][tile_id];
        }

#if BACKWARDS_PASS == true

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
#if ATTACH_POSITION == true
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_position[i * TILE_SIZE + j] =
                params.target_position[ray_id + i * params.image_width + j];
        }
#endif
#if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_depth[i * TILE_SIZE + j] =
                params.target_depth[ray_id + i * params.image_width + j];
        }
#endif
#if ATTACH_NORMALS == true
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_normal[i * TILE_SIZE + j] =
                params.target_normal[ray_id + i * params.image_width + j];
        }
#endif
#if ATTACH_F0 == true
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_f0[i * TILE_SIZE + j] =
                params.target_f0[ray_id + i * params.image_width + j];
        }
#endif
#if ATTACH_ROUGHNESS == true
    for (int i = 0; i < TILE_SIZE; i++)
        for (int j = 0; j < TILE_SIZE; j++) {
            target_roughness[i * TILE_SIZE + j] =
                params.target_roughness[ray_id + i * params.image_width + j];
        }
#endif
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
            float3 unnormalized_normal = output_normal[step][0][k];
        }

        // Reset to 0 since backward pass increments these
        dL_dray_origin_next_step = make_float3(0.0f, 0.0f, 0.0f);
        dL_dray_direction_next_step = make_float3(0.0f, 0.0f, 0.0f);

        if (num_hits[step] > 0) {

#if ROUGHNESS_DOWNWEIGHT_GRAD == true
            float roughness_weight = powf(
                1.0f - output_roughness[max(step - 1, 0)][0][0],
                ROUGHNESS_DOWNWEIGHT_GRAD_POWER);
#else
            float roughness_weight = 1.0f;
#endif

#if DOWNWEIGHT_EXTRA_BOUNCES == true
            float extra_bounce_weight =
                powf(EXTRA_BOUNCE_WEIGHT, float(max(step - 1, 0)));
#else
            float extra_bounce_weight = 1.0f;
#endif

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
                roughness_weight * extra_bounce_weight,
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

#endif

    params.random_seeds[ray_id] = seed;
}
