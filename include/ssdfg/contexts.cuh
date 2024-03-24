#pragma once

#include <array>

#include <optix/optix.h>

#include <oak/mesh.hpp>
#include <oak/camera.hpp>
#include <oak/transform.hpp>

#include "cuda/interop.cuh"

struct SilhouetteRenderContext {
	// Interoperable render buffers
	LinkedFramebuffer visibility_lfb;
	LinkedFramebuffer depth_lfb;
	LinkedFramebuffer boundary_lfb;
	LinkedFramebuffer render_lfb;

	LinkedFramebuffer sdf_lfb;
	LinkedFramebuffer barycentrics_lfb;
	LinkedFramebuffer gradients_lfb;

	// Raytracing items
	OptixShaderBindingTable sbt;
	OptixTraversableHandle gas;
	CUdeviceptr device_packet;

	// CUDA computation buffers
	struct {
		float *visibility;
		float *depth;
		int32_t *primitives;
		float2 *barycentrics;
		float3 *render;
	} render_targets;

	struct {
		float *boundary;
		float *sdf;
		float2 *gradients;
	} image_space;

	struct {
		uint3 *triangles;
		float3 *vgradients;
		uint *counts;
	} mesh;

	vk::Extent2D extent;

	// Refreshing the current mesh
	// TODO: return a new one?
	void update(const OptixDeviceContext &, const Mesh &);

	// Rendering
	void render(const OptixPipeline &, const Camera &, const Transform &);

	// Constructing
	static SilhouetteRenderContext from(const DeviceResourceContext &, const OptixDeviceContext &, const std::array <OptixProgramGroup, 3> &, const Mesh &, const vk::Extent2D &);
};