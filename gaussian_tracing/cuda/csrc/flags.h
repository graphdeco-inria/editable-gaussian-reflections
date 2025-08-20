#pragma once

// * Set the max number of bounces for the path tracer, can be reduced in config
// but this might be slower
#define MAX_BOUNCES 2

// * Essential for stability, can't remove
#define MAX_ALPHA 0.9999f

// * Hacky loss downweighting, would like this removed but it improves results
// in the roughness scene
#define ROUGHNESS_DOWNWEIGHT_GRAD true
#define ROUGHNESS_DOWNWEIGHT_GRAD_POWER 3.0f

// * Performance flags, safe to ignore
#define BUFFER_SIZE 16
#define MAX_ITERATIONS 99

#define T_THRESHOLD 0.01f
#define EXP_POWER 3
#define ALPHA_THRESHOLD 0.005f

#define PPLL_STORAGE_SIZE 400000000
#define PPLL_STORAGE_SIZE_BACKWARD 300000000

// -----------------------
// ----- To remove
// -----------------------

#define NUM_CHUNKS 1
#define CLAMP_MAX_VALUE 9999999999.0f
#define EXTRA_BOUNCE_WEIGHT 0.01f
#define SKIP_BACKFACING_MAX_DIST 0.1f
#define SKIP_BACKFACING_REFLECTION_VALID_NORMAL_MIN_NORM 0.9f
#define SURFACE_EPS 0.01f
#define MIN_ROUGHNESS 0.01f
#define REFLECTION_VALID_NORMAL_MIN_NORM 0.7f
#define DETACH_AFTER 9999
#define TILE_SIZE 1
#define NUM_SLABS 1
#define MIN_SCALING_FACTOR 0.0f
#define EPS_SCALE_GRAD 1e-12f
#define LUT_SIZE 512
#define NUM_CLUSTERS 1
