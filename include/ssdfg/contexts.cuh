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
	LinkedFramebuffer sdf_lfb;
	LinkedFramebuffer gradients_lfb;

	// Raytracing items
	OptixShaderBindingTable sbt;
	OptixTraversableHandle gas;
	CUdeviceptr device_packet;

	// CUDA computation buffers
	float *visibility;
	float *depth;
	float *boundary;
	float *sdf;
	float2 *gradients;

	vk::Extent2D extent;

	// Rendering
	void render(const OptixPipeline &, const Camera &, const Transform &);

	// Constructing
	static SilhouetteRenderContext from(const DeviceResourceContext &, const OptixDeviceContext &, const std::array <OptixProgramGroup, 3> &, const Mesh &, const vk::Extent2D &);
};