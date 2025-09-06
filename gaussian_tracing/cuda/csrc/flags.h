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
#define EXTRA_BOUNCE_WEIGHT 0.01f

// * Performance flags, safe to ignore
#define BUFFER_SIZE 16
#define MAX_ITERATIONS 99

#define PPLL_STORAGE_SIZE 400000000
#define PPLL_STORAGE_SIZE_BACKWARD 300000000

// -----------------------
// ----- To remove
// -----------------------

#define CLAMP_MAX_VALUE 9999999999.0f
#define DETACH_AFTER 9999
#define TILE_SIZE 1
#define EPS_SCALE_GRAD 1e-12f
#define NUM_CLUSTERS 1
