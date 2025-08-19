#pragma once

// -----------------------
// ----- Main feature flags
// -----------------------

#define MAX_BOUNCES 2
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

#define INCLUDE_BRDF_WEIGHT true

#define SAMPLE_FROM_BRDF true
#define DENOISER true

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

#define NORMALIZE_NORMAL_MAP true

#define INIT_F0 0.0f
#define INIT_ROUGHNESS 0.0f

#define SKIP_LOSS_AVG_NUM_PIXELS true
#define GLOBAL_GRADIENT_SCALE 1.0f

#define JITTER true

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

#define SAVE_HIT_STATS false

// n.b. writing all maps to memory can several ms per frame and should not be
// done at inference

// -----------------------
// ----- Debug options
// -----------------------

#define DETACH_AFTER 9999
#define USE_POLYCAGE false
#define TILE_SIZE 1
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

#define LUT_SIZE 512
#define LOG_ALL_HITS true
#define RECOMPUTE_ALPHA_IN_FORWARD_PASS false
#define BBOX_PADDING 0.0f
#define NUM_CLUSTERS 1
