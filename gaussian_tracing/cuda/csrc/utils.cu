#include "glm/glm.hpp"
#include "cuda_fp16.h"


__device__ __inline__ void atomicAdd4(float4* address, float4 val) {
	atomicAdd(&address->x, val.x);
	atomicAdd(&address->y, val.y);
	atomicAdd(&address->z, val.z);
	atomicAdd(&address->w, val.w);
}

__device__ __inline__ void atomicAdd3(float3* address, float3 val) {
	atomicAdd(&address->x, val.x);
	atomicAdd(&address->y, val.y);
	atomicAdd(&address->z, val.z);
}

__device__ int lcg_hash(int x) {
    return (1103515245 * x + 12345);  
}

// __device__ float3 computeCov2D(const float3 _scale, float mod, const float4 _rot, const float3 _mean)
// {
//     glm::vec3 scale = glm::vec3(_scale.x, _scale.y, _scale.z);
//     glm::vec4 rot = glm::vec4(_rot.x, _rot.y, _rot.z, _rot.w);
//     glm::vec3 mean = glm::vec3(_mean.x, _mean.y, _mean.z);

//     const float viewmatrix[16] = {
//         1.0000, 0.0023, -0.0039, 0.0000,
//         0.0045, -0.5093, 0.8606, 0.0000,
//         -0.0000, -0.8606, -0.5093, 0.0000,
//         -0.0000, 0.0000, 1.9218, 1.0000
//     };

//     // float focal_x, float focal_y, float tan_fovx, float tan_fovy,
//     // const float* viewmatrix
//     // todo hard-code focal_x, focal_y, tan_fovx, tan_fovy based on the results from the 3dgs code

// 	// Create scaling matrix
// 	glm::mat3 S = glm::mat3(1.0f);
// 	S[0][0] = mod * scale.x;
// 	S[1][1] = mod * scale.y;
// 	S[2][2] = mod * scale.z;

// 	// Normalize quaternion to get valid rotation
// 	glm::vec4 q = rot;// / glm::length(rot); //??
// 	float r = q.x;
// 	float x = q.y;
// 	float y = q.z;
// 	float z = q.w;

// 	// Compute rotation matrix from quaternion
// 	glm::mat3 R = glm::mat3(
// 		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
// 		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
// 		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
// 	);

// 	glm::mat3 M = S * R;

// 	// Compute 3D world covariance matrix Sigma
// 	glm::mat3 Sigma = glm::transpose(M) * M;

//     float cov3D[6];
// 	// Covariance is symmetric, only store upper right
// 	cov3D[0] = Sigma[0][0];
// 	cov3D[1] = Sigma[0][1];
// 	cov3D[2] = Sigma[0][2];
// 	cov3D[3] = Sigma[1][1];
// 	cov3D[4] = Sigma[1][2];
// 	cov3D[5] = Sigma[2][2];

// 	// The following models the steps outlined by equations 29
// 	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
// 	// Additionally considers aspect / scaling of viewport.
// 	// Transposes used to account for row-/column-major conventions.
// 	// float3 t = transformPoint4x3(mean, viewmatrix);
//     float3 t = {
// 		viewmatrix[0] * mean.x + viewmatrix[4] * mean.y + viewmatrix[8] * mean.z + viewmatrix[12],
// 		viewmatrix[1] * mean.x + viewmatrix[5] * mean.y + viewmatrix[9] * mean.z + viewmatrix[13],
// 		viewmatrix[2] * mean.x + viewmatrix[6] * mean.y + viewmatrix[10] * mean.z + viewmatrix[14],
// 	}; //?

// 	// const float limx = 1.3f * tan_fovx;
// 	// const float limy = 1.3f * tan_fovy;
// 	// const float txtz = t.x / t.z;
// 	// const float tytz = t.y / t.z;
// 	// t.x = min(limx, max(-limx, txtz)) * t.z;
// 	// t.y = min(limy, max(-limy, tytz)) * t.z;

// 	float size_x = 2048.0 / 2;
// 	float size_y = 2048.0 / 2; 

// 	glm::mat3 J = glm::mat3(
// 		size_y, 0.0f, 0.0, //-t.x
// 		0.0f, size_y, 0.0, //-t.y
// 		0, 0, 0);

// 	glm::mat3 W = glm::mat3(
// 		viewmatrix[0], viewmatrix[4], viewmatrix[8],
// 		viewmatrix[1], viewmatrix[5], viewmatrix[9],
// 		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

// 	glm::mat3 T = W * J;

// 	glm::mat3 Vrk = glm::mat3(
// 		cov3D[0], cov3D[1], cov3D[2],
// 		cov3D[1], cov3D[3], cov3D[4],
// 		cov3D[2], cov3D[4], cov3D[5]);

// 	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

// 	// Apply low-pass filter: every Gaussian should be at least
// 	// one pixel wide/high. Discard 3rd row and column.
// 	cov[0][0] += 0.3f; 
// 	cov[1][1] += 0.3f;
// 	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
// }


__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max)
{
	rect_min = {
		(unsigned int) max((int)0, (int)((p.x - max_radius) / 16)),
		(unsigned int)max((int)0, (int)((p.y - max_radius) / 16))
	};
	rect_max = {
		(unsigned int)max((int)0, (int)((p.x + max_radius + 16 - 1) / 16)),
		(unsigned int)max((int)0, (int)((p.y + max_radius + 16 - 1) / 16))
	};
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

template<typename T>
__device__ void fill_array(T* arr, uint32_t size, T val) {
	for (int i = 0; i < size; i++) {
		arr[i] = val;
	}
}

//!!
// #if STORAGE_MODE != PER_PIXEL_LINKED_LIST
// 	__device__ unsigned int packFloats(const float& distance, const float& alpha) {
// 		__half2 packed_halves = __halves2half2(__float2half(distance), __float2half(alpha));
// 		return *reinterpret_cast<const unsigned int*>(&packed_halves);
// 	}

// 	__device__ float unpackDistance(const unsigned int& packed_bits) {
// 		__half2 packed_halves = *reinterpret_cast<const __half2*>(&packed_bits);
// 		return __half2float(packed_halves.x);
// 	}

// 	__device__ float unpackAlpha(const unsigned int& packed_bits) {
// 		__half2 packed_halves = *reinterpret_cast<const __half2*>(&packed_bits);
// 		return __half2float(packed_halves.y);
// 	}
// #else
	__device__ unsigned int packFloats(const float& distance, const float& alpha) {
		return __float_as_uint(distance);
	}

	__device__ float unpackDistance(const float& packed) {
		return __uint_as_float(packed);
	}

	__device__ float unpackAlpha(const float& packed) {
		return __uint_as_float(packed);
	}
// #endif


#ifdef SORT_BY_COUNTING
	__device__ unsigned int packId(const uint32_t gaussian_id, uint32_t count) {
		// store the last 4 bits of count into the 4 most significant bits of gaussian_id
		return (gaussian_id << 4) | (count & 0xF);
	}

	__device__ uint32_t unpackId(const uint32_t& packed) {
		return packed >> 4;
	}

	__device__ uint32_t unpackCount(const uint32_t& packed) {
		return packed & 0xF;	
	}
#else
	__device__ unsigned int packId(const uint32_t gaussian_id, uint32_t count) {
		// store the last 4 bits of count into the 4 most significant bits of gaussian_id
		return gaussian_id;
	}

	__device__ uint32_t unpackId(const uint32_t& packed) {
		return packed;
	}

	__device__ uint32_t unpackCount(const uint32_t& packed) {
		return 0;
	}
#endif


