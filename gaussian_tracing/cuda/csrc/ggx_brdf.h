#ifndef GGX_BRDF_CUH
#define GGX_BRDF_CUH

#include <cmath>
#include <cuda_runtime.h>


// New version below

#define BRDF_EPS 1e-8

__device__ float D_GGX(float3 N, float3 H, float alpha) {
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float a2 = alpha * alpha;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI * (denom * denom) + BRDF_EPS);
}

__device__ void D_GGX_derivatives(float3 N, float3 H, float alpha, float& dD_dalpha, float3& dD_dN, float3& dD_dH) {
    float3 Nn = normalize(N);
    float3 Hn = normalize(H);
    float x = fmaxf(dot(Nn, Hn), 0.0f); 
    float a2 = alpha * alpha;
    float s_minus1 = a2 - 1.0f;
    float x2 = x * x;
    float d = x2 * s_minus1 + 1.0f;
    float d3 = d * d * d;

    dD_dalpha = (2.0f * alpha * (d - 2.0f * a2 * x2)) / (M_PI * d3);

    float dD_dx = (-4.0f * a2 * x * s_minus1) / (M_PI * d3);
    dD_dN = dD_dx * Hn;
    dD_dH = dD_dx * Nn;
}

__device__ float G1(float3 N, float3 W, float alpha) {
    float NdotW = fmaxf(dot(N, W), 0.0f);
    float a = alpha;
    float k = (a * a) / 2.0f;
    return NdotW / (NdotW * (1.0f - k) + k + BRDF_EPS); 
}

__device__ void G1_derivatives(float3 N, float3 W, float alpha,
    float& dG1_dalpha, float3& dG1_dN, float3& dG1_dW,
    float& G1_out) {
    float3 Nn = normalize(N);
    float3 Wn = normalize(W);
    float x = fmaxf(dot(Nn, Wn), 0.0f); 
    float a2 = alpha * alpha;
    float k = a2 * 0.5f;
    float d = x * (1.0f - k) + k + BRDF_EPS;
    float d2 = d * d;

    float G = x / d;
    G1_out = G;

    float dk_dalpha = alpha;
    float dd_dalpha = (1.0f - x) * dk_dalpha;
    dG1_dalpha = -x * dd_dalpha / d2;

    float dd_dx = 1.0f - k;
    float dG_dx = (d - x * dd_dx) / d2;

    dG1_dN = dG_dx * Wn;
    dG1_dW = dG_dx * Nn;
}

__device__ float G_Smith(float3 N, float3 V, float3 L, float alpha) {
    return G1(N, V, alpha) * G1(N, L, alpha);
}

__device__ void G_Smith_derivatives(float3 N, float3 V, float3 L, float alpha,
    float& dG_dalpha, float3& dG_dN,
    float3& dG_dV, float3& dG_dL) {
    float dG_dalpha_v, dG_dalpha_l;
    float3 dG_dN_v, dG_dN_l;
    float3 dG_dV_v, dG_dl_l;

    float Gv, Gl;
    
    G1_derivatives(N, V, alpha, dG_dalpha_v, dG_dN_v, dG_dV_v, Gv);
    G1_derivatives(N, L, alpha, dG_dalpha_l, dG_dN_l, dG_dl_l, Gl);

    dG_dalpha = dG_dalpha_v * Gl + Gv * dG_dalpha_l;

    dG_dN = dG_dN_v * Gl + dG_dN_l * Gv;
    dG_dV = dG_dV_v * Gl;
    dG_dL = dG_dl_l * Gv;
}

__device__ float3 fresnel_schlick(float3 F0, float cosTheta) {
    return F0 + (1.0f - F0) * powf(1.0f - cosTheta, 5.0f);
}

__device__ void fresnel_schlick_backward(
    float3 dl_dF, // ∂L/∂F (incoming gradient)
    float3 F0, float cosTheta,
    float3& dl_dF0, // output accumulators for F0
    float& dl_dcosTheta, // scalar output for cosTheta
    float3& F_out // forward Fresnel result
) {
    float one_minus_cos = 1.0f - cosTheta;
    float one_minus_cos2 = one_minus_cos * one_minus_cos;
    float one_minus_cos4 = one_minus_cos2 * one_minus_cos2;
    float one_minus_cos5 = one_minus_cos4 * one_minus_cos;

    float3 one_minus_F0 = make_float3(1.0f - F0.x, 1.0f - F0.y, 1.0f - F0.z);

    F_out = F0 + one_minus_F0 * one_minus_cos5;

    // Scalar, same for all channels
    float dF_dF0_scalar = 1.0f - one_minus_cos5;
    dl_dF0 = dl_dF * dF_dF0_scalar;

    // ∂F/∂cosTheta
    float3 dF_dcosTheta = make_float3(
        -5.0f * one_minus_F0.x * one_minus_cos4,
        -5.0f * one_minus_F0.y * one_minus_cos4,
        -5.0f * one_minus_F0.z * one_minus_cos4
    );
    
    dl_dcosTheta = dot(dl_dF, dF_dcosTheta);
}

__device__ float3 cook_torrance_brdf(float3 N, float3 V, float3 L, float roughness, float3 F0) {
    if (F0.x == 0.0f && F0.y == 0.0f && F0.z == 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float3 H = normalize(V + L);
    float alpha = roughness * roughness;

    float D = D_GGX(N, H, alpha);
    float G = G_Smith(N, V, L, alpha); 
    float cosTheta = fmaxf(dot(L, H), 0.0f);
    float3 F = fresnel_schlick(F0, cosTheta);

    float NdotL = fmaxf(dot(N, L), 0.0f);
    float NdotV = fmaxf(dot(N, V), 0.0f);

    float denom = 4.0f * NdotL * NdotV + BRDF_EPS;

    return (D * G * F) / denom;
}

__device__ float3 cook_torrance_weight(float3 N, float3 V, float3 L, float roughness, float3 F0) {
    if (F0.x == 0.0f && F0.y == 0.0f && F0.z == 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    float3 H = normalize(V + L);

    float NdotH = fmaxf(dot(N, H), 0.0f);
    float VdotH = fmaxf(dot(V, H), 0.0f);
    float NdotV = fmaxf(dot(N, V), 0.0f);

    float alpha = roughness * roughness;
    float G = G_Smith(N, V, L, alpha);
    float3 F = fresnel_schlick(F0, VdotH);

    return F * G * VdotH / (NdotH * NdotV + BRDF_EPS);
}

__device__ float3 sample_cook_torrance(float3 N, float3 V, float roughness, float2 uniform_samples) {

    // Walter's trick
    float alpha = roughness * roughness;
    float phi = 2.0f * M_PI * uniform_samples.x;
    float y = uniform_samples.y;
    float cosTheta = sqrtf((1.0f - y) / (1.0f + (alpha * alpha - 1.0f) * y));
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    float3 H_local = make_float3(
        sinTheta * cosf(phi),
        sinTheta * sinf(phi),
        cosTheta
    );

    // Transform H to world space
    float3 up = fabsf(N.z) < 0.99f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 T = normalize(cross(N.z < 0.999f ? make_float3(0,0,1) : make_float3(1,0,0), N));
    float3 B = cross(N, T);
    float3 H = H_local.x * T + H_local.y * B + H_local.z * N;

    return reflect(-V, H);
}

#endif