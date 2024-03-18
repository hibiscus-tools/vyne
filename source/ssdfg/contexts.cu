#include "cuda/optix/util.cuh"
#include "cuda/optix/acceleration.cuh"
#include "ssdfg/contexts.cuh"

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

	vk::Sampler sampler = littlevk::SamplerCompiler(drc.device, drc.dal);

	auto visibility_lfb = LinkedFramebuffer::from(drc, image_info, sampler);
	auto boundary_lfb = LinkedFramebuffer::from(drc, image_info, sampler);
	auto sdf_lfb = LinkedFramebuffer::from(drc, image_info, sampler);
	auto gradients_lfb = LinkedFramebuffer::from(drc, image_info, sampler);

	// Allocate the computation buffers
	size_t pixels = extent.width * extent.height;
	float *visibility = (float *) cuda_alloc_buffer(sizeof(float) * pixels);
	float *boundary = (float *) cuda_alloc_buffer(sizeof(float) * pixels);
	float *sdf = (float *) cuda_alloc_buffer(sizeof(float) * pixels);
	float2 *gradients = (float2 *) cuda_alloc_buffer(sizeof(float2) * pixels);

	return SilhouetteRenderContext {
		.visibility_lfb = visibility_lfb,
		.boundary_lfb = boundary_lfb,
		.sdf_lfb = sdf_lfb,
		.gradients_lfb = gradients_lfb,

		.sbt = sbt,
		.gas = gas,

		.visibility = visibility,
		.boundary = boundary,
		.sdf = sdf,
		.gradients = gradients,

		.extent = extent
	};
}