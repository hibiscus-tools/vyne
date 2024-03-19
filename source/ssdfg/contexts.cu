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
	auto sdf_lfb = LinkedFramebuffer::from(drc, image_info, sampler);
	auto gradients_lfb = LinkedFramebuffer::from(drc, image_info, sampler);

	// Allocate the computation buffers
	size_t pixels = extent.width * extent.height;
	float *visibility = (float *) cuda_alloc_buffer(sizeof(float) * pixels);
	float *depth = (float *) cuda_alloc_buffer(sizeof(float) * pixels);
	float *boundary = (float *) cuda_alloc_buffer(sizeof(float) * pixels);
	float *sdf = (float *) cuda_alloc_buffer(sizeof(float) * pixels);
	float2 *gradients = (float2 *) cuda_alloc_buffer(sizeof(float2) * pixels);

	// Packet for raytracing
	Packet packet;
	CUdeviceptr device_packet = cuda_element_buffer(packet);

	return SilhouetteRenderContext {
		.visibility_lfb = visibility_lfb,
		.depth_lfb = depth_lfb,
		.boundary_lfb = boundary_lfb,
		.sdf_lfb = sdf_lfb,
		.gradients_lfb = gradients_lfb,

		.sbt = sbt,
		.gas = gas,
		.device_packet = device_packet,

		.visibility = visibility,
		.depth = depth,
		.boundary = boundary,
		.sdf = sdf,
		.gradients = gradients,

		.extent = extent
	};
}

void SilhouetteRenderContext::render(const OptixPipeline &pipeline, const Camera &camera, const Transform &camera_transform)
{
	Packet packet;

	RayFrame rayframe = camera.rayframe(camera_transform);

	packet.origin = glm_to_float3(rayframe.origin);
	packet.lower_left = glm_to_float3(rayframe.lower_left);
	packet.horizontal = glm_to_float3(rayframe.horizontal);
	packet.vertical = glm_to_float3(rayframe.vertical);
	packet.resolution = make_uint2(extent.width, extent.height);
	packet.visibility = visibility;
	packet.depth = depth;
	packet.gas = gas;

	cuda_element_copy(device_packet, packet);

	optixLaunch
	(
		pipeline, 0,
		device_packet, sizeof(Packet), &sbt,
		extent.width, extent.height, 1
	);

	cudaDeviceSynchronize();

	// TODO: cuStreams
	silhouette_edges <<<64, 128>>> (visibility, boundary, extent.width, extent.height);
	cudaDeviceSynchronize();

	signed_distance_field <<< 64, 128 >>> (visibility, boundary, sdf, extent.width, extent.height);
	cudaDeviceSynchronize();

	sdf_spatial_gradient <<< 64, 128 >>> (sdf, gradients, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_mask <<< 64, 128 >>> (visibility, visibility_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_mask <<< 64, 128 >>> (boundary, boundary_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_scalar_field <<< 64, 128 >>> (depth, depth_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_sdf <<< 64, 128 >>> (sdf, sdf_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();

	render_image_space_vectors <<< 64, 128 >>> (gradients, gradients_lfb.surface, extent.width, extent.height);
	cudaDeviceSynchronize();
}