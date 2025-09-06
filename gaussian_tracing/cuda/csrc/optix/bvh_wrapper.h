
#include "../core/all.h"
#include "../params.h"

void populateBVH(
    OptixInstance *instances,
    OptixTraversableHandle gasHandle,
    int num_gaussians,
    float3 *scales,
    float4 *rotations,
    float3 *means,
    float *opacity,
    float alpha_threshold,
    float exp_power,
    float global_scale_factor);

struct BVHWrapper {
    OptixTraversableHandle tlas_handle; // * Pass this handle when launching the
                                        // ray tracing pipeline

    BVHWrapper(
        OptixDeviceContext context_,
        Tensor m_gaussian_means,
        Tensor m_gaussian_scales,
        Tensor m_gaussian_rotations,
        Tensor m_gaussian_opacity,
        const ConfigDataHolder &config_,
        const Params &params_on_host_)
        : context(context_), config(config_), params_on_host(params_on_host_) {
        build_blas();
        build_tlas(
            m_gaussian_means,
            m_gaussian_scales,
            m_gaussian_rotations,
            m_gaussian_opacity);
        cudaDeviceSynchronize();
    }

    void rebuild(
        Tensor m_gaussian_means,
        Tensor m_gaussian_scales,
        Tensor m_gaussian_rotations,
        Tensor m_gaussian_opacity) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(device_instances)));
        CUDA_CHECK(
            cudaFree(reinterpret_cast<void *>(device_temp_tlas_buffer_sizes)));
        CUDA_CHECK(
            cudaFree(reinterpret_cast<void *>(device_tlas_output_buffer)));

        build_tlas(
            m_gaussian_means,
            m_gaussian_scales,
            m_gaussian_rotations,
            m_gaussian_opacity);

        cudaDeviceSynchronize();
    }

    void update(
        Tensor m_gaussian_means,
        Tensor m_gaussian_scales,
        Tensor m_gaussian_rotations,
        Tensor m_gaussian_opacity) {
        // * Update Transforms
        populateBVH(
            reinterpret_cast<OptixInstance *>(device_instances),
            blas_handle,
            m_gaussian_means.sizes()[0],
            reinterpret_cast<float3 *>(m_gaussian_scales.data_ptr()),
            reinterpret_cast<float4 *>(m_gaussian_rotations.data_ptr()),
            reinterpret_cast<float3 *>(m_gaussian_means.data_ptr()),
            reinterpret_cast<float *>(m_gaussian_opacity.data_ptr()),
            config.alpha_threshold.item<float>(),
            config.exp_power.item<float>(),
            config.global_scale_factor.item<float>());

        // * Update TLAS
        OptixAccelBuildOptions accel_options_tlas = {};
        accel_options_tlas.buildFlags =
            build_flags | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
        accel_options_tlas.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(
            context,
            0,
            &accel_options_tlas,
            &tlas_input,
            1,
            device_temp_tlas_buffer_sizes,
            tlas_buffer_sizes.tempSizeInBytes,
            device_tlas_output_buffer,
            tlas_buffer_sizes.outputSizeInBytes,
            &tlas_handle,
            nullptr,
            0));
    }

  private:
    // * Input fields
    OptixDeviceContext context;
    const ConfigDataHolder &config;
    const Params &params_on_host;

    // * Optix stuff
    uint32_t aabb_input_flags[2] = {OPTIX_GEOMETRY_FLAG_NONE};
    unsigned int build_flags =
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    OptixBuildInput blas_input = {};
    OptixBuildInput tlas_input = {};
    OptixTraversableHandle blas_handle;
    CUdeviceptr device_tlas_output_buffer;
    CUdeviceptr device_temp_tlas_buffer_sizes;
    CUdeviceptr device_blas_output_buffer;
    CUdeviceptr device_instances;
    OptixAccelBufferSizes tlas_buffer_sizes;
    CUdeviceptr device_aabb_buffer;
    Tensor unit_bbox_tensor = torch::tensor(
        {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, torch::device(torch::kCUDA));

    void build_blas() {
        OptixAccelBuildOptions accel_options_blas = {};
        accel_options_blas.buildFlags = build_flags;
        accel_options_blas.operation = OPTIX_BUILD_OPERATION_BUILD;

        device_aabb_buffer =
            reinterpret_cast<CUdeviceptr>(unit_bbox_tensor.data_ptr());
        blas_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        blas_input.customPrimitiveArray.aabbBuffers = &device_aabb_buffer;
        blas_input.customPrimitiveArray.numPrimitives = 1;
        blas_input.customPrimitiveArray.flags = aabb_input_flags;
        blas_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBufferSizes blas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context, &accel_options_blas, &blas_input, 1, &blas_buffer_sizes));

        CUdeviceptr d_temp_blas_buffer_sizes;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_temp_blas_buffer_sizes),
            blas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&device_blas_output_buffer),
            blas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(
            context,
            0,
            &accel_options_blas,
            &blas_input,
            1,
            d_temp_blas_buffer_sizes,
            blas_buffer_sizes.tempSizeInBytes,
            device_blas_output_buffer,
            blas_buffer_sizes.outputSizeInBytes,
            &blas_handle,
            nullptr,
            0));

        CUDA_CHECK(
            cudaFree(reinterpret_cast<void *>(d_temp_blas_buffer_sizes)));
    }

    void build_tlas(
        Tensor m_gaussian_means,
        Tensor m_gaussian_scales,
        Tensor m_gaussian_rotations,
        Tensor m_gaussian_opacity) {
        auto num_gaussians = m_gaussian_means.sizes()[0];

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&device_instances),
            sizeof(OptixInstance) * num_gaussians));
        populateBVH(
            reinterpret_cast<OptixInstance *>(device_instances),
            blas_handle,
            num_gaussians,
            reinterpret_cast<float3 *>(m_gaussian_scales.data_ptr()),
            reinterpret_cast<float4 *>(m_gaussian_rotations.data_ptr()),
            reinterpret_cast<float3 *>(m_gaussian_means.data_ptr()),
            reinterpret_cast<float *>(m_gaussian_opacity.data_ptr()),
            config.alpha_threshold.item<float>(),
            config.exp_power.item<float>(),
            config.global_scale_factor.item<float>());

        OptixAccelBuildOptions accel_options_tlas = {};
        accel_options_tlas.buildFlags =
            build_flags | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
        accel_options_tlas.operation = OPTIX_BUILD_OPERATION_BUILD;

        tlas_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        tlas_input.instanceArray.instances = device_instances;
        tlas_input.instanceArray.numInstances = num_gaussians;

        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context, &accel_options_tlas, &tlas_input, 1, &tlas_buffer_sizes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&device_temp_tlas_buffer_sizes),
            tlas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&device_tlas_output_buffer),
            tlas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(
            context,
            0,
            &accel_options_tlas,
            &tlas_input,
            1,
            device_temp_tlas_buffer_sizes,
            tlas_buffer_sizes.tempSizeInBytes,
            device_tlas_output_buffer,
            tlas_buffer_sizes.outputSizeInBytes,
            &tlas_handle,
            nullptr,
            0));
    }

  public:
    ~BVHWrapper() {
        cudaFree(reinterpret_cast<void *>(device_instances));
        cudaFree(reinterpret_cast<void *>(device_temp_tlas_buffer_sizes));
        cudaFree(reinterpret_cast<void *>(device_tlas_output_buffer));
        cudaFree(reinterpret_cast<void *>(device_blas_output_buffer));
    }
};