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
