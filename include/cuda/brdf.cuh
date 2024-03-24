#pragma once

#include "vector_math.cuh"

__forceinline__ __host__ __device__
float3 shlick_fresnel(float3 Ks, float3 wi, float3 h)
{
	float k = powf(1 - dot(wi, h), 5.0f);
	return Ks + (1 - Ks) * k;
}

// Orthogonal basis (coordinate frame) at a surface point
struct Basis {
	float3 u;
	float3 v;
	float3 w;

	__forceinline__ __host__ __device__
	float3 operator()(float3 a) {
		return a.x * u + a.y * v + a.z * w;
	}
};

// Ground-glass-unknown material
struct GGX {
	float3 Kd;
	float3 Ks;
	float alpha;
};