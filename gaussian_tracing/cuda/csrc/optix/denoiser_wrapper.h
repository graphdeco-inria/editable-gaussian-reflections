#include "../params.h"

#include <optix_denoiser_tiling.h>
#include <torch/extension.h>

using namespace at;

class DenoiserWrapper {
    OptixDenoiser optix_denoiser = nullptr;
    uint32_t scratch_size = 0;
    CUdeviceptr hdrIntensity = 0;
    CUdeviceptr hdrAverageColor = 0;
    CUdeviceptr scratch = 0;
    CUdeviceptr state = 0;
    uint32_t state_size = 0;
    unsigned int overlap = 0;
    OptixDenoiserParams params_denoiser = {};
    OptixDenoiserGuideLayer guideLayer = {};
    std::vector<OptixDenoiserLayer> layers;

  public:
    DenoiserWrapper(
        OptixDeviceContext context_,
        const Params &params_on_host,
        const Tensor m_output_rgb,
        const Tensor m_output_normal) {
        OptixDenoiserOptions options = {};
        options.guideAlbedo = 0;
        options.guideNormal = 1;
        options.denoiseAlpha = (OptixDenoiserAlphaMode)0;
        OptixDenoiserModelKind modelKind;
        modelKind = OPTIX_DENOISER_MODEL_KIND_HDR;
        OPTIX_CHECK(optixDenoiserCreate(
            context_, modelKind, &options, &optix_denoiser));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&hdrIntensity), sizeof(float)));

        OptixDenoiserSizes denoiser_sizes;
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(
            optix_denoiser,
            params_on_host.image_width,
            params_on_host.image_height,
            &denoiser_sizes));
        scratch_size = static_cast<uint32_t>(
            denoiser_sizes.withoutOverlapScratchSizeInBytes);
        overlap = 0;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&hdrAverageColor), 3 * sizeof(float)));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&scratch), scratch_size));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&state),
            denoiser_sizes.stateSizeInBytes));

        // * Denoise inplace
        OptixDenoiserLayer layer = {};
        layer.input = createOptixImage2D(
            params_on_host.image_width,
            params_on_host.image_height,
            reinterpret_cast<CUdeviceptr>(
                m_output_rgb.index({MAX_BOUNCES + 1}).data_ptr()));
        layer.output = createOptixImage2D(
            params_on_host.image_width,
            params_on_host.image_height,
            reinterpret_cast<CUdeviceptr>(
                m_output_rgb.index({MAX_BOUNCES + 1}).data_ptr()));
        layers.push_back(layer);
        guideLayer.normal = createOptixImage2D(
            params_on_host.image_width,
            params_on_host.image_height,
            reinterpret_cast<CUdeviceptr>(
                m_output_normal.index({0}).data_ptr()));

        state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);
        OPTIX_CHECK(optixDenoiserSetup(
            optix_denoiser,
            nullptr, // * CUDA stream
            params_on_host.image_width,
            params_on_host.image_height,
            state,
            state_size,
            scratch,
            scratch_size));
        params_denoiser.hdrIntensity = hdrIntensity;
        params_denoiser.hdrAverageColor = hdrAverageColor;
        params_denoiser.blendFactor = 0.0f;
        params_denoiser.temporalModeUsePreviousLayers = 0;
    }

    void run() {
        OPTIX_CHECK(optixDenoiserComputeIntensity(
            optix_denoiser,
            nullptr, // * CUDA stream
            &layers[0].input,
            hdrIntensity,
            scratch,
            scratch_size));
        OPTIX_CHECK(optixDenoiserInvoke(
            optix_denoiser,
            nullptr, // * CUDA stream
            &params_denoiser,
            state,
            state_size,
            &guideLayer,
            layers.data(),
            static_cast<unsigned int>(layers.size()),
            0, // * Input offset X
            0, // * Input offset y
            scratch,
            scratch_size));
    }

    ~DenoiserWrapper() {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(hdrIntensity)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(hdrAverageColor)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(scratch)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state)));
        OPTIX_CHECK(optixDenoiserDestroy(optix_denoiser));
    }

  private:
    static OptixImage2D createOptixImage2D(
        unsigned int width, unsigned int height, CUdeviceptr tensor_data) {
        OptixImage2D oi;
        oi.data = tensor_data;
        oi.width = width;
        oi.height = height;
        oi.rowStrideInBytes = width * sizeof(float3);
        oi.pixelStrideInBytes = sizeof(float3);
        oi.format = OPTIX_PIXEL_FORMAT_FLOAT3;
        return oi;
    }
};
