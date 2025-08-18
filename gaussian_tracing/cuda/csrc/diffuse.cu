// todo separate out the first trace from the bounces in the code before all of
// this

float3 output_rgb = make_float3(0.0f, 0.0f, 0.0f);

for (int i = 0; i < 6; i++) {
    float3 remaining_rgb = make_float3(0.0f, 0.0f, 0.0f);
    float3 remaining_position = make_float3(0.0f, 0.0f, 0.0f);
    float3 remaining_normal = make_float3(0.0f, 0.0f, 0.0f);
    float3 remaining_f0 = make_float3(0.0f, 0.0f, 0.0f);
    float remaining_roughness = 0.0f;

    float2 output_t = make_float2(1.0f, 1.0f);
    float3 output_position = make_float3(0.0f, 0.0f, 0.0f);
    float3 output_normal = make_float3(0.0f, 0.0f, 0.0f);
    float3 output_f0 = make_float3(0.0f, 0.0f, 0.0f);
    float output_roughness = 0.0f;
    float3 output_surface_brdf = make_float3(1.0f, 1.0f, 1.0f);

    // float3 tile_origin = // todo set to origin from the first raytrace
    // float3 tile_direction = // todo select from a predetermined list of
    // directions and rotate so z = the surface normal

    froward_pass(
        step,
        ray_id,
        tile_origin,
        tile_direction,
        origin,
        direction,
        output_rgb,
        output_t,
        output_position,
        output_normal,
        output_f0,
        output_roughness,
        remaining_rgb,
        remaining_position,
        remaining_normal,
        remaining_f0,
        remaining_roughness,
        num_hits);

    // todo multiply by the same brdf here
}

// todo would need to pass in dummy values, here, but wigthin the code, aim for
// target diffuse after for (int step = 6; step >= 0; step--) {
//     backward_pass(
//         step,
//         ray_id, tile_origin, tile_direction,
//         origin, direction,
//         output_rgb, output_t,  output_position, output_normal, output_f0,
//         output_roughness,
//         remaining_rgb, remaining_position, remaining_normal, remaining_f0,
//         remaining_roughness,
//         num_hits, output_surface_brdf
//     );
// }