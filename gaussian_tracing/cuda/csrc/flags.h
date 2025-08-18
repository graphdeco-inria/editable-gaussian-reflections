#pragma once

// -----------------------
// ----- Main feature flags
// -----------------------

#define MAX_BOUNCES 2
#define USE_GT_BRDF false
#define SURFACE_BRDF_FROM_GT_F0 false
#define SURFACE_BRDF_FROM_GT_ROUGHNESS false
#define SURFACE_BRDF_FROM_GT_NORMAL false
#define REFLECTION_RAY_FROM_GT_POSITION false
#define REFLECTION_RAY_FROM_GT_NORMAL false

#define POSITION_FROM_EXPECTED_TERMINATION_DEPTH true

#define USE_CLUSTERING false
#define NUM_CLUSTERS 2

#define BACKWARDS_PASS true
#define NUM_CHUNKS 1

#define CLAMP_MAX_VALUE 9999999999.0f

// -----------------------
// ----- Current WIP
// -----------------------

#define STRICT_REJECT_GAUSSIANS_BEHIND_RAY true

#define ROUGHNESS_DOWNWEIGHT_GRAD true
#define ROUGHNESS_DOWNWEIGHT_GRAD_POWER 3.0f
#define DOWNWEIGHT_EXTRA_BOUNCES true
#define EXTRA_BOUNCE_WEIGHT 0.01f

#define USE_LUT false

#define USE_EPANECHNIKOV_KERNEL false
#define INCLUDE_BRDF_WEIGHT true

#define SAMPLE_FROM_BRDF true
#define DENOISER true

#define USE_RUSSIAN_ROULETTE false
#define ROULETTE_POWER 0.10f

#define SKIP_BACKFACING_NORMALS true
#define SKIP_BACKFACING_MAX_DIST 0.1f
#define SKIP_BACKFACING_REFLECTION_VALID_NORMAL_MIN_NORM 0.9f
#define SURFACE_EPS 0.01f

#define MIN_ROUGHNESS 0.01f

// --------------------- stuff to adjust below

#define REFLECTION_VALID_NORMAL_MIN_NORM                                       \
    0.7f // higher leads to fewer invalid rays, but to black outlines

#define IGNORE_0_VOLUME_GAUSSIANS true
#define IGNORE_0_VOLUME_MINSIZE 0.0f

// --------------------------------- finalized values below

#define TONEMAP false
#define STRAIGHT_THROUGH_TONEMAPPING_GRAD false
#define INVERT_TONEMAP_TARGET false

#define NORMALIZE_NORMAL_MAP true
#define PROJECT_POSITION_TO_RAY true

#define INIT_F0 0.0f
#define INIT_ROUGHNESS 0.0f

#define SKIP_LOSS_AVG_NUM_PIXELS true
#define GLOBAL_GRADIENT_SCALE 1.0f

#define JITTER true
#define GLOBAL_SORT false

// --------------------------- let go for now

#define USE_LEVEL_OF_DETAIL false
#define USE_LEVEL_OF_DETAIL_MASKING false
#define LOD_KERNEL_EXPONENT 4.0f

#define USE_GRADIENT_SCALING false
#define GRADIENT_SCALING_UNCLAMPED false

#define INDEPENDENT_GLOSSY_LOSS false
#define SEMI_INDEPENDENT_GLOSSY_LOSS false

#define ADD_LOD_MEAN_TO_SCALE false

// -----------------------
// ----- Rendering mode
// -----------------------

// * STORAGE_MODE choices:
#define NO_STORAGE 0
#define PER_PIXEL_LINKED_LIST 1

#define STORAGE_MODE PER_PIXEL_LINKED_LIST

// * REMAINING_COLOR_ESTIMATION choices:
#define ASSUME_SAME_AS_FOREGROUND 0
#define IGNORE_OCCLUSION 1
#define STOCHASTIC_SAMPLING 2
#define RENDER_BUT_DETACH 3
#define NO_ESTIMATION 4

#define REMAINING_COLOR_ESTIMATION ASSUME_SAME_AS_FOREGROUND

// -----------------------
// ---- Peformance options
// -----------------------

#define MAX_ITERATIONS                                                         \
    99 // n.b. ignored when REMAINING_COLOR_ESTIMATION == RENDER_BUT_DETACH
#define BUFFER_SIZE 16 // n.b. forced to 16 when STORAGE_MODE == NO_STORAGE

// -----------------------
// ----- Quality/peformance tradeoff
// -----------------------

#define T_THRESHOLD 0.01f
#define EXP_POWER 3
#define ALPHA_THRESHOLD 0.005f

// -----------------------
// ----- Image logging
// -----------------------

#define SAVE_ALL_MAPS true   //! kind of slow now? includes too many things
#define SAVE_RAY_IMAGES true // !!!!!
#define SAVE_LUT_IMAGES false
#define SAVE_LOD_IMAGES false
#define SAVE_HIT_STATS false

// n.b. writing all maps to memory can several ms per frame and should not be
// done at inference

// -----------------------
// ----- WIP feature flags
// -----------------------

#define RENDER_DISTORTION false
#define DISTORTION_NEAR_PLANE 0.2f
#define DISTORTION_FAR_PLANE 1000.0f
#define USE_DISTORTION_LOSS false
#define DISTORTION_LOSS_WEIGHT 1.0f

#define REFLECTIONS_FROM_GT_GLOSSY_IRRADIANCE false

#define SELF_SUPERVISED_POSITION_GRADS                                         \
    false //  interaction between using gt normals and self-supervised grads?
          //  need to disable the grads?
#define SELF_SUPERVISED_NORMAL_GRADS                                           \
    false //  interaction between using gt normals and self-supervised grads?
          //  need to disable the grads?
#define SELF_SUPERVISED_F0_GRADS                                               \
    false //  interaction between using gt normals and self-supervised grads?
          //  need to disable the grads?
#define SELF_SUPERVISED_ROUGHNESS_GRADS                                        \
    false //  interaction between using gt normals and self-supervised grads?
          //  need to disable the grads?

// -----------------------
// ----- Debug options
// -----------------------

#define ENABLE_DEBUG_DUMP false
#define DEBUG_DUMP_PIXEL_ID 777

#define DYN_CLAMPING true
#define DETACH_AFTER 9999

#define DEBUG_ASSUME_KNOWING_TMAX false
#define DEBUG_ASSUME_KNOWING_TMIN false
#define DEBUG_VIEW_ELLIPSOIDS false

#define DEBUG_CHEAP_TMAX_ESTIMATE false
#define CHEAP_TMAX_DOWNSAMPLING 6
#define DEBUG_SINGLE_EMPTY_RAYTRACE false
#define DEBUG_SECOND_BOUNCE false

#define DEBUG_DISABLE_CLIPPING false

// -----------------------
// ----- Legacy options
// -----------------------

#define SQUARE_KERNEL false
#define USE_POLYCAGE false
#define ALPHA_SMOOTHING false
#define ALPHA_SMOOTHING_THRESHOLD 0.1f
#define ALPHA_RESCALE false
#define USE_MASKING false
#define STOCHASTIC false
#define BB_SHRINKAGE 1.0f

// -----------------------
// ----- Incomplete options
// -----------------------

#define TILE_SIZE 1
#define OPTIMIZE_EXP_POWER false
#define GRADS_ON_REMAINING_GAUSSIANS false
#define COMPACTION false
#define FUSED_MESH false
#define TRI_SOUP_FOR_MESHES false
#define OPACITY_MODULATION false
#define ORTHO_CAM false
#define ANTIALIASING 0.0f
#define MEASURE_LOSS false
#define ALLOW_OPACITY_ABOVE_1 false
#define NUM_SLABS 1

// -----------------------
// ----- Numerical stability & memory
// -----------------------

#define MIN_SCALING_FACTOR 0.0f
#define MAX_ALPHA 0.9999f
#define EPS_SCALE_GRAD 1e-12f

// -----------------------
// ----- Memory consumption
// -----------------------

#define PPLL_STORAGE_SIZE 400000000
#define PPLL_STORAGE_SIZE_BACKWARD 300000000

// -----------------------
// ----- Final or automatic options, don't edit
// -----------------------

#define MAX_DUMPED_HITS (MAX_ITERATIONS * BUFFER_SIZE * (MAX_BOUNCES + 1))

#define LUT_SIZE 512

#define ACTIVATION_IN_CUDA true
#define RELU_INSTEAD_OF_SOFTPLUS true
#define CLIPPED_RELU_INSTEAD_OF_SIGMOID true

#ifdef EXTRA_H_AVAILABLE
#include "extra.h"
#endif

#if STORAGE_MODE == NO_STORAGE
#define BUFFER_SIZE 16
#endif

#if REMAINING_COLOR_ESTIMATION == RENDER_BUT_DETACH
#define DETACH_AFTER MAX_ITERATIONS
#define MAX_ITERATIONS 9999
#endif

#if STORAGE_MODE == PER_PIXEL_LINKED_LIST
#define LOG_ALL_HITS true
#else
#define LOG_ALL_HITS false
#endif

#if ENABLE_DEBUG_DUMP == true
#define BACKWARDS_PASS true
#endif

#if TILE_SIZE > 1
#define RECOMPUTE_ALPHA_IN_FORWARD_PASS false
// #define BBOX_PADDING 0.005f
#define BBOX_PADDING 0.0f
#else
#define RECOMPUTE_ALPHA_IN_FORWARD_PASS false
#define BBOX_PADDING 0.0f
#endif

#if STORAGE_MODE == NO_STORAGE
#define RECOMPUTE_ALPHA_IN_FORWARD_PASS true
#endif

#if DEBUG_VIEW_ELLIPSOIDS == true
#define RECOMPUTE_ALPHA_IN_FORWARD_PASS true
#define DYN_CLAMPING false
#endif

#if FUSED_MESH == true
#define USE_POLYCAGE 1
#endif

#if MAX_BOUNCES == 0
#define DENOISER false
#define SAVE_RAY_IMAGES false
#define SAVE_LUT_IMAGES false
#define SURFACE_BRDF_FROM_GT_F0 false
#define SURFACE_BRDF_FROM_GT_ROUGHNESS false
#define SURFACE_BRDF_FROM_GT_NORMAL false
#define REFLECTION_RAY_FROM_GT_POSITION false
#define REFLECTION_RAY_FROM_GT_NORMAL false
#define POSITION_FROM_EXPECTED_TERMINATION_DEPTH false
#endif

#define USE_GT_DIFFUSE_IRRADIANCE                                              \
    false // n.b. this is a placeholder for future work

// Specify which values get attached to gaussians
#define ATTACH_POSITION false
#define ATTACH_NORMALS false
#define ATTACH_F0 false
#define ATTACH_ROUGHNESS false
//

#if MAX_BOUNCES > 0
#define ATTACH_POSITION true
#define ATTACH_NORMALS true
#define ATTACH_F0 true
#define ATTACH_ROUGHNESS true
#endif

#if USE_DISTORTION_LOSS == true
#define RENDER_DISTORTION true
#endif
#if USE_LEVEL_OF_DETAIL == false
#define SAVE_LOD_IMAGES false
#endif

#if USE_CLUSTERING == 0
#define NUM_CLUSTERS 1
#endif

// --------------------------------------------------------------------

// Preset configurations

#if CONFIG == 0 // DEFAULT
#define T_THRESHOLD 0.1f
#define EXP_POWER 3
#define ALPHA_THRESHOLD 0.02f
#define BB_SHRINKAGE 0.8f
#elif CONFIG == 1 // HIGH_SPEED
#define T_THRESHOLD 0.2f
#define EXP_POWER 5
#define ALPHA_THRESHOLD 0.1f
#define BB_SHRINKAGE 0.7f
#elif CONFIG == 2 // HIGH_QUALITY
#define T_THRESHOLD 0.05f
#define EXP_POWER 2
#define ALPHA_THRESHOLD 0.01f
#elif CONFIG == 3 // MATCH_3DGS
#define T_THRESHOLD 0.001f
#define EXP_POWER 1
#define ALPHA_THRESHOLD 0.001f
#define GLOBAL_SORT true
#define BBOX_PADDING 0.002f
#define ANTIALIAS 0.001f
#define REMAINING_COLOR_ESTIMATION NO_ESTIMATION
#define PPLL_STORAGE_SIZE 500000000
#elif CONFIG == 4          // MATCH_3DRT
#define T_THRESHOLD 0.001f // (& switch to 0.03 at inference)
#define EXP_POWER 2
#define ALPHA_THRESHOLD 0.01f
#define REMAINING_COLOR_ESTIMATION                                             \
    NO_ESTIMATION  // using our estimate propbably changes nothing but can only
                   // improve psnr
#elif CONFIG == 11 // TMP
// #define BBOX_PADDING 0.002f
#elif CONFIG != -1
#error "Invalid CONFIG value"
#endif