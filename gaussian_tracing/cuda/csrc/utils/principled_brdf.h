#pragma once

#include <cmath>
#include <cuda_runtime.h>

__device__ float3 fresnel_schlick(float3 F0, float cosTheta) {
    return F0 + (1.0f - F0) * powf(1.0f - cosTheta, 5.0f);
}

__device__ float3 exact_fresnel(const float3 wi,
    const float3 H,
    const float3 f0) {
    auto fresnel_dielectric_cos = [] __device__ (float cosi, float3 eta) {
        float c = fabsf(cosi);
        float3 g2 = eta * eta - 1.0f + c * c;
        float3 g  = make_float3(sqrtf(g2.x), sqrtf(g2.y), sqrtf(g2.z));
        float3 A  = (g - c) / (g + c);
        float3 B  = (c * (g + c) - 1.0f) / (c * (g - c) + 1.0f);
        float3 result = 0.5f * A * A * (1.0f + B * B);
        if (g2.x <= 0.0f) result.x = 1.0f;
        if (g2.y <= 0.0f) result.y = 1.0f;
        if (g2.z <= 0.0f) result.z = 1.0f;
        return result;
    };

    auto inverse_lerp = [] __device__ (float3 a, float b, float3 x) {
        return (x - a) / (b - a);
    };

    auto lerp3 = [] __device__ (const float3& a, const float3& b, float3 t) {
        return a + (b - a) * t;
    };

    float3 ior = (2.0f / (1.0f - make_float3(sqrtf(f0.x), sqrtf(f0.y), sqrtf(f0.z)))) - 1.0f; 
    const float3 real_F  = fresnel_dielectric_cos(dot(wi, H), ior);
    const float3 real_F0 = fresnel_dielectric_cos(1.0f, ior);
    return lerp3(f0, make_float3(1.0f), inverse_lerp(real_F0, 1.0f, real_F));
}

__device__ inline float D_GGX(float3 N, float3 H, float alpha) {
    float a2 = alpha * alpha;                 
    float Nh = fmaxf(dot(N, H), 0.0f);
    if (Nh <= 0.0f) return 0.0f;
    float Nh2   = Nh * Nh;
    float denom = fmaf((a2 - 1.0f), Nh2, 1.0f); // 1 + (a^2-1)*(N·H)^2
    return a2 / (M_PI * denom * denom);
}

__device__ inline float lambda_GGX(float3 N, float3 W, float alpha) {
    float c = dot(N, W);                       
    if (c <= 0.0f) return 0.0f;                
    float a2 = alpha * alpha;
    float c2 = c * c;
    float s2 = fmaf((1.0f - a2), c2, a2);  // a^2 + (1-a^2)*c^2
    float s  = sqrtf(s2);
    return 0.5f * (s / c - 1.0f);
}

__device__ inline float G_Smith(float3 N, float3 V, float3 L, float alpha) {
    float lv = lambda_GGX(N, V, alpha);
    float ll = lambda_GGX(N, L, alpha);
    return 1.0f / (1.0f + lv + ll);
}

__device__ float3 principled_specular(float3 N, float3 V, float3 L, float3 f0, float roughness) {
    float alpha = roughness * roughness;
    if (alpha * alpha < 5e-7f) {
        return make_float3(0.0f);
    }

    float3 H = normalize(V + L);

    float D = D_GGX(N, H, alpha);
    float3 F = exact_fresnel(V, H, f0);
    float G = G_Smith(N, V, L, alpha); 

    float NoL = dot(N, L);
    if (NoL <= 0.0f) return make_float3(0.0f);
    float NoV = dot(N, V);
    if (NoV <= 0.0f) return make_float3(0.0f);

    return D * F * G / (4.0f * NoV);
}

__device__ inline float3 safe_normalize(const float3& v) {
    float l2 = fmaf(v.x, v.x, fmaf(v.y, v.y, v.z * v.z));
    float inv = rsqrtf(fmaxf(l2, 1e-30f));
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

__device__ float safe_sqrtf(float a)
{
  return (a > 0.0) ? sqrt(a) : 0.0;
}

__device__ float3 sample_vndf(float3 N, float3 V, float roughness, float2 u)
{
    // inline tangent frame
    float3 T, B;
    float3 nn = safe_normalize(N);
    if (nn.z < -0.999999f) {
        T = make_float3(0, -1, 0);
        B = make_float3(-1, 0, 0);
    } else {
        float a = 1.0f / fmaxf(1.0f + nn.z, 1e-30f);
        B = safe_normalize(make_float3(-nn.x * nn.y * a, 1.0f - nn.y * nn.y * a, -nn.y));
        T = safe_normalize(make_float3(1.0f - nn.x * nn.x * a, -nn.x * nn.y * a, -nn.x));
    }

    // view in local frame
    float3 Vlocal = safe_normalize(make_float3(dot(V, T), dot(V, B), dot(V, nn)));

    // Heitz hemisphere flip
    bool flip = (Vlocal.z < 0.0f);
    if (flip) Vlocal.z = -Vlocal.z;

    // stretch view (GGX)
    float alpha = roughness * roughness;
    float3 Vh = safe_normalize(make_float3(alpha * Vlocal.x, alpha * Vlocal.y, Vlocal.z));

    // basis around Vh
    float lensq = fmaf(Vh.x, Vh.x, Vh.y * Vh.y);
    float3 T1, T2;
    if (lensq > 1e-7f) {
        T1 = make_float3(-Vh.y, Vh.x, 0.0f) * rsqrtf(lensq);
        T2 = cross(Vh, T1);
    } else {
        T1 = make_float3(1.0f, 0.0f, 0.0f);
        T2 = make_float3(0.0f, 1.0f, 0.0f);
    }

    // disk sample and Heitz warp
    float r   = sqrtf(fminf(u.x, 1.0f));
    float phi = 2.0f * M_PIf * u.y;
    float t1  = r * cosf(phi);
    float t2  = r * sinf(phi);
    float s   = 0.5f * (1.0f + Vh.z);
    float tmp = fmaxf(0.0f, 1.0f - t1 * t1);
    t2 = (1.0f - s) * safe_sqrtf(tmp) + s * t2;

    // reproject onto hemisphere (stretched domain)
    float3 Hs = t1 * T1 + t2 * T2 + safe_sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // flip back if we flipped V
    if (flip) Hs.z = -Hs.z;

    // un-stretch and normalize
    float3 Hlocal = safe_normalize(make_float3(alpha * Hs.x, alpha * Hs.y, fmaxf(0.0f, Hs.z)));

    // to world and reflect
    float3 H = make_float3(Hlocal.x * T.x + Hlocal.y * B.x + Hlocal.z * nn.x,
                           Hlocal.x * T.y + Hlocal.y * B.y + Hlocal.z * nn.y,
                           Hlocal.x * T.z + Hlocal.y * B.z + Hlocal.z * nn.z);

    // * Normalize seems required here otherwise small errors lead to errors down the road
    return normalize(reflect(-V, H)); 
}

__device__ float G1(float3 N, float3 W, float alpha) {
    float NdotW = dot(N, W);
    if (NdotW <= 0.0f) return 0.0f;
    return 1.0f / (1.0f + lambda_GGX(N, W, alpha));
}

__device__ float sample_vndf_pdf( // VNDF PDF, solid angle
    float3 N, float3 V, float3 L, float roughness)
{
    float NoV = fminf(dot(N, V), 1.0f);
    if (NoV <= 0.0f) return 0.0f;

    float3 H = safe_normalize(V + L);
    float NoH = fminf(dot(N, H), 1.0f);
    float VoH = fminf(dot(V, H), 1.0f);
    if (NoH <= 0.0f || VoH <= 0.0f) return 0.0f;

    // Filament trick: D = r^2 / (PI * (|N×H|^2 + (r*NoH)^2)^2)
    float a = roughness * roughness;
    float3 NxH = cross(N, H);
    const float t   = fmaf(a * NoH, a * NoH, dot(NxH, NxH)); // |N×H|^2 + (a*NoH)^2
    const float inv = 1.0f / fmaxf(t, 1e-20f);
    float D = (a * a) * (inv * inv) * (1.0f / M_PIf);

    float G1v = G1(N, V, a);
    return D * G1v / (4.0f * NoV);
}