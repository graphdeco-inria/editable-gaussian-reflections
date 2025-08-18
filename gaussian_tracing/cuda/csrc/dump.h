

struct __align__(64) Dump {
    int idx = -2;

    // One entry per step:

    float3 origin[MAX_BOUNCES + 1];
    float3 direction[MAX_BOUNCES + 1];
    float4 rgbt[MAX_BOUNCES + 1];
    float3 position[MAX_BOUNCES + 1];
    float3 normal[MAX_BOUNCES + 1];
    float3 f0[MAX_BOUNCES + 1];
    float roughness[MAX_BOUNCES + 1];
    float distortion[MAX_BOUNCES + 1];
    float lod_mean[MAX_BOUNCES + 1];
    float lod_scale[MAX_BOUNCES + 1];
    float ray_lod[MAX_BOUNCES + 1];

    float full_T[MAX_BOUNCES + 1];
    float3 remaining_rgb[MAX_BOUNCES + 1];

    // One entry per hit:

    int step[MAX_DUMPED_HITS];
    float distances[MAX_DUMPED_HITS];
    uint32_t gaussian_ids[MAX_DUMPED_HITS];

    float3 hit_point_world[MAX_DUMPED_HITS];
    float3 hit_point_local[MAX_DUMPED_HITS];
    float gaussval[MAX_DUMPED_HITS];
    float alpha[MAX_DUMPED_HITS];
    float T[MAX_DUMPED_HITS];

    float4 xforms_0[MAX_DUMPED_HITS];
    float4 xforms_1[MAX_DUMPED_HITS];
    float4 xforms_2[MAX_DUMPED_HITS];

    float4 inv_xforms_0[MAX_DUMPED_HITS];
    float4 inv_xforms_1[MAX_DUMPED_HITS];
    float4 inv_xforms_2[MAX_DUMPED_HITS];

    float3 dL_drgb[MAX_DUMPED_HITS];
    float dL_dopacity[MAX_DUMPED_HITS];

    float dL_dgaussian_lod_mean[MAX_DUMPED_HITS];
    float dL_dgaussian_lod_scale[MAX_DUMPED_HITS];

    float3 backward_weighted_rgb_deltas[MAX_DUMPED_HITS];
    float3 backward_prev_gaussian_rgb[MAX_DUMPED_HITS];
    float backward_T[MAX_DUMPED_HITS];

    float dL_dalpha[MAX_DUMPED_HITS];
    float dL_dgaussval[MAX_DUMPED_HITS];
    float3 dL_dx_local[MAX_DUMPED_HITS];
    float3 dL_dx_world[MAX_DUMPED_HITS];

    float4 dL_dxform_0[MAX_DUMPED_HITS];
    float4 dL_dxform_1[MAX_DUMPED_HITS];
    float4 dL_dxform_2[MAX_DUMPED_HITS];

    float3 dL_drot_0[MAX_DUMPED_HITS];
    float3 dL_drot_1[MAX_DUMPED_HITS];
    float3 dL_drot_2[MAX_DUMPED_HITS];

    float3 dL_dmeans[MAX_DUMPED_HITS];
    float3 dL_dscales[MAX_DUMPED_HITS];
    float4 dL_drotations[MAX_DUMPED_HITS];

    float dL_dexp_powers[MAX_DUMPED_HITS];
};