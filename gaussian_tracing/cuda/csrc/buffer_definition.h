
#pragma once

// * Gaussian data

#define GAUSSIAN_SHAPE_ATTRIBUTES(X)                                           \
    X(opacity, 1)                                                              \
    X(scale, 3)                                                                \
    X(mean, 3)                                                                 \
    X(rotation, 4)

#define GAUSSIAN_ATTACHED_LAYERS(X)                                            \
    X(rgb, 3)                                                                  \
    X(normal, 3)                                                               \
    X(f0, 3)                                                                   \
    X(roughness, 1)

#define GAUSSIAN_PRODUCED_LAYERS(X)                                            \
    GAUSSIAN_ATTACHED_LAYERS(X)                                                \
    X(depth, 1)

#define ALL_GAUSSIAN_ATTRIBUTES(X)                                             \
    GAUSSIAN_ATTACHED_LAYERS(X)                                                \
    GAUSSIAN_SHAPE_ATTRIBUTES(X)

#define ALL_GAUSSIAN_ACTIVATIONS(X)                                            \
    X(opacity, sigmoid)                                                        \
    X(scale, exp)                                                              \
    X(mean, identity)                                                          \
    X(rotation, normalize)                                                     \
    X(rgb, relu)                                                               \
    X(normal, identity)                                                        \
    X(f0, clipped_relu)                                                        \
    X(roughness, clipped_relu)

// * Outputs and targets

#define PRODUCED_LAYERS(X)                                                     \
    X(rgb, 3, make_float3(0.0f, 0.0f, 0.0f))                                   \
    X(depth, 1, 999999.9f)                                                     \
    X(normal, 3, make_float3(0.0f, 0.0f, 0.0f))                                \
    X(f0, 3, make_float3(0.0f, 0.0f, 0.0f))                                    \
    X(roughness, 1, 0.0f)

#define OUTPUT_LAYERS(X)                                                       \
    PRODUCED_LAYERS(X)                                                         \
    X(transmittance, 1, 1.0f)                                                  \
    X(total_transmittance, 1, 1.0f)

#define ALL_OUTPUT_BUFFERS(X)                                                  \
    OUTPUT_LAYERS(X)                                                           \
    X(brdf, 3, make_float3(0.0f, 0.0f, 0.0f))                                  \
    X(ray_origin, 3, make_float3(0.0f, 0.0f, 0.0f))                            \
    X(ray_direction, 3, make_float3(0.0f, 0.0f, 0.0f))

#define ALL_ACCUMULATION_BUFFERS(X)                                            \
    X(rgb, 3)                                                                  \
    X(transmittance, 1)                                                        \
    X(total_transmittance, 1)                                                  \
    X(depth, 1)                                                                \
    X(normal, 3)                                                               \
    X(f0, 3)                                                                   \
    X(roughness, 1)

#define GEOMETRY_BRDF_TARGET_BUFFERS(X)                                        \
    X(depth, 1)                                                                \
    X(normal, 3)                                                               \
    X(f0, 3)                                                                   \
    X(roughness, 1)

#define ALL_TARGET_BUFFERS(X)                                                  \
    X(diffuse, 3)                                                              \
    X(glossy, 3)                                                               \
    GEOMETRY_BRDF_TARGET_BUFFERS(X)

// * Helpers

#define FLOAT_OF_SIZE(k) FLOAT_OF_SIZE_IMPL(k)
#define FLOAT_OF_SIZE_IMPL(k) FLOAT_TYPE_##k
#define FLOAT_TYPE_1 float
#define FLOAT_TYPE_2 float2
#define FLOAT_TYPE_3 float3
#define FLOAT_TYPE_4 float4

#define ZERO_OF_SIZE(k) ZERO_OF_SIZE_IMPL(k)
#define ZERO_OF_SIZE_IMPL(k) ZERO_##k
#define ZERO_1 0.0f
#define ZERO_2 make_float2(0.0f, 0.0f)
#define ZERO_3 make_float3(0.0f, 0.0f, 0.0f)
#define ZERO_4 make_float4(0.0f, 0.0f, 0.0f, 0.0f)
