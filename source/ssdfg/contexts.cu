#include "cuda/optix/acceleration.cuh"
#include "cuda/optix/util.cuh"
#include "cuda/vector_math.cuh"
#include "shaders/optix/ssdfg.cuh"
#include "ssdfg/contexts.cuh"
#include "ssdfg/kernels.cuh"

SilhouetteRenderContext SilhouetteRenderContext::from
(
	const DeviceResourceContext &drc,
	const OptixDeviceContext &optix_context,
	const std::array <OptixProgramGroup, 3> &program_groups,
	const Mesh &mesh,
	const vk::Extent2D &extent
)
{
	// Construct the acceleration structure
	OptixTraversableHandle gas = gas_from_mesh(optix_context, mesh);

	// Make the shader binding table records
	Record <float> raygen_record;
	optixSbtRecordPackHeader(program_groups[0], &raygen_record);
	CUdeviceptr raygen_record_sbt = cuda_element_buffer(raygen_record);

	Record <float> closesthit_record;
	optixSbtRecordPackHeader(program_groups[1], &closesthit_record);
	CUdeviceptr closesthit_record_sbt = cuda_element_buffer(closesthit_record);

	Record <float> miss_record;
	optixSbtRecordPackHeader(program_groups[2], &miss_record);
	CUdeviceptr miss_record_sbt = cuda_element_buffer(miss_record);

	// Create the shader binding table
	OptixShaderBindingTable sbt {};

	sbt.raygenRecord = raygen_record_sbt;

	sbt.hitgroupRecordBase = closesthit_record_sbt;
	sbt.hitgroupRecordCount = 1;
	sbt.hitgroupRecordStrideInBytes = sizeof(Record <float>);

	sbt.missRecordBase = miss_record_sbt;
	sbt.missRecordCount = 1;
	sbt.missRecordStrideInBytes = sizeof(Record <float>);

	// Configure a blitting presenter
	const littlevk::ImageCreateInfo image_info {
		.width = extent.width,
		.height = extent.height,
		.format = drc.swapchain.format,
		.usage = vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst,
		.aspect = vk::ImageAspectFlagBits::eColor,
		.external = true
	};

	auto sampler = littlevk::SamplerCompiler(drc.device, drc.dal);
	auto visibility_lfb = LinkedFramebuffer::from(drc, image_info, sampler);
	auto depth_lfb = LinkedFramebuffer::from(drc, image_info, sampler);
	auto boundary_lfb = LinkedFramebuffer::from(drc, image_info, sampler);
	auto render_lfb = LinkedFramebuffer::from(drc, image_info, sampler);

	auto sdf_lfb = LinkedFramebuffer::from(drc, image_info, sampler);
	auto barycentrics_lfb = LinkedFramebuffer::from(drc, image_info, sampler);
	auto gradients_lfb = LinkedFramebuffer::from(drc, image_info, sampler);

	// Allocate the computation buffers
	size_t pixels = extent.width * extent.height;
	float *visibility = (float *) cuda_alloc(sizeof(float) * pixels);
	float *depth = (float *) cuda_alloc(sizeof(float) * pixels);
	int32_t *primitive = (int32_t *) cuda_alloc(sizeof(int32_t) * pixels);
	float2 *barycentrics = (float2 *) cuda_alloc(sizeof(float2) * pixels);

	float *boundary = (float *) cuda_alloc(sizeof(float) * pixels);
	float *sdf = (float *) cuda_alloc(sizeof(float) * pixels);
	float2 *gradients = (float2 *) cuda_alloc(sizeof(float2) * pixels);

	// Packet for raytracing
	Packet packet;
	CUdeviceptr device_packet = cuda_element_buffer(packet);

	return SilhouetteRenderContext {
		.visibility_lfb = visibility_lfb,
		.depth_lfb = depth_lfb,
		.boundary_lfb = boundary_lfb,
		.render_lfb = render_lfb,

		.sdf_lfb = sdf_lfb,
		.barycentrics_lfb = barycentrics_lfb,
		.gradients_lfb = gradients_lfb,

		.sbt = sbt,
		.gas = gas,
		.device_packet = device_packet,

		.render_targets {
			.visibility = visibility,
			.depth = depth,
			.primitives = primitive,
			.barycentrics = barycentrics,
			.render = cuda_alloc_buffer <float3> (pixels)
		},

		.image_space {
			.boundary = boundary,
			.sdf = sdf,
			.gradients = gradients,
		},

		.mesh {
			.triangles = (uint3 *) cuda_vector_buffer(mesh.triangles),
			.vgradients = cuda_alloc_buffer <float3> (mesh.positions.size()),
			.counts = cuda_alloc_buffer <uint> (mesh.positions.size())
		},

		.extent = extent
	};
}

void SilhouetteRenderContext::update(const OptixDeviceContext &optix_context, const Mesh &new_mesh)
{
	if (mesh.triangles)
		cudaFree(mesh.triangles);
	if (mesh.vgradients)
		cudaFree(mesh.vgradients);
	if (mesh.counts)
		cudaFree(mesh.counts);

	gas = gas_from_mesh(optix_context, new_mesh);
	mesh.triangles = (uint3 *) cuda_vector_buffer(new_mesh.triangles);
	mesh.vgradients = cuda_alloc_buffer <float3> (new_mesh.positions.size());
	mesh.counts = cuda_alloc_buffer <uint> (new_mesh.positions.size());
}

void SilhouetteRenderContext::render(const OptixPipeline &pipeline, const Camera &camera, const Transform &camera_transform)
{
	Packet packet;

	RayFrame rayframe = camera.rayframe(camera_transform);

	packet.gas = gas;

	packet.origin = glm_to_float3(rayframe.origin);
	packet.lower_left = glm_to_float3(rayframe.lower_left);
	packet.horizontal = glm_to_float3(rayframe.horizontal);
	packet.vertical = glm_to_float3(rayframe.vertical);

	packet.visibility = render_targets.visibility;
	packet.depth = render_targets.depth;
	packet.primitive = render_targets.primitives;
	packet.barycentrics = render_targets.barycentrics;
	packet.render = render_targets.render;

	packet.resolution = make_uint2(extent.width, extent.height);

	cuda_element_copy(device_packet, packet);

	optixLaunch
	(
		pipeline, nullptr,
		device_packet, sizeof(Packet), &sbt,
		extent.width, extent.height, 1
	);

	cudaDeviceSynchronize();

	// TODO: cuStreams
	silhouette_edges <<<64, 128>>> (render_targets.visibility, image_space.boundary, extent.width, extent.height);
	cudaDeviceSynchronize();

	signed_distance_field <<< 64, 128 >>> (render_targets.visibility, image_space.boundary, image_space.sdf, extent.width, extent.height);
	cudaDeviceSynchronize();

	sdf_spatial_gradient <<< 64, 128 >>> (image_space.sdf, image_space.gradients, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_mask <<< 64, 128 >>> (render_targets.visibility, visibility_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_mask <<< 64, 128 >>> (image_space.boundary, boundary_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_scalar_field <<< 64, 128 >>> (render_targets.depth, depth_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_sdf <<< 64, 128 >>> (image_space.sdf, sdf_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_image_space_vectors <<< 64, 128 >>> (image_space.gradients, gradients_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_vector_color <<< 64, 128 >>> (render_targets.barycentrics, barycentrics_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_vector_color <<< 64, 128 >>> (render_targets.render, render_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();
}