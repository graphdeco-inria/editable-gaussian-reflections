
#include <optix_denoiser_tiling.h>
#include <torch/extension.h>

using namespace at;

static OptixImage2D createOptixImage2D(
    unsigned int width, unsigned int height, CUdeviceptr tensor_data = 0) {
    OptixImage2D oi;

    if (tensor_data != 0) {
        oi.data = tensor_data;
    } else {
        const uint64_t frame_byte_size = width * height * sizeof(float3);
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&oi.data), frame_byte_size));
    }
    oi.width = width;
    oi.height = height;
    oi.rowStrideInBytes = width * sizeof(float3);
    oi.pixelStrideInBytes = sizeof(float3);
    oi.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    return oi;
}
