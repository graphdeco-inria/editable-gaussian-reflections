#pragma once

#include "flags.h"

#include <cuda.h>
#include <iostream>
#include <optix.h>
#include <random>

#include "core/all.h"

struct Params {
    uint32_t image_width;
    uint32_t image_height;

    Camera camera;
    Config config;
    Framebuffer framebuffer;
    Gaussians gaussians;
    Metadata metadata;
    Stats stats;

    PerPixelLinkedList ppll_forward;
    PerPixelLinkedList ppll_backward;

    OptixTraversableHandle bvh_handle;
};

extern "C" {
__constant__ Params params;
}
