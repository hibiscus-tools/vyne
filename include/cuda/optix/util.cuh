#pragma once

#include <filesystem>

#include <optix/optix.h>
#include <optix/optix_stubs.h>

#include "util.hpp"

template <typename T>
struct Record {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT)
	char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

struct OptixProgramType {
	const char *entry;

	enum {
		eRayGeneration,
		eClosestHit,
		eMiss
	} type;

	static OptixProgramType ray_generation(const char *entry) {
		return OptixProgramType { entry, eRayGeneration };
	}

	static OptixProgramType closest_hit(const char *entry) {
		return OptixProgramType { entry, eClosestHit };
	}

	static OptixProgramType miss(const char *entry) {
		return OptixProgramType { entry, eMiss };
	}
};

static void context_logger(unsigned int level, const char *tag, const char *message, void *)
{
	ulog_info("optix", "%s %s\n", tag, message);
}

inline OptixDeviceContext make_context()
{
	cudaFree(0);
	optixInit();

	// Specify context options
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_logger;
	options.logCallbackLevel = 4;

	// Associate CUDA context
	CUcontext cuda_context = 0;

	OptixDeviceContext optix_context = 0;

	optixDeviceContextCreate(cuda_context, &options, &optix_context);
	optixDeviceContextSetCacheEnabled(optix_context, 0);

	return optix_context;
}

// TODO: pipeline/module compile/link optins as a template struct parameter?
static OptixModuleCompileOptions &module_compile_options()
{
	static OptixModuleCompileOptions module_compile_options {};
	module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
	module_compile_options.numPayloadTypes = 0;
	module_compile_options.payloadTypes = nullptr;
	return module_compile_options;
}

static OptixPipelineCompileOptions &pipeline_compile_options()
{
	static OptixPipelineCompileOptions pipeline_compile_options {};
	pipeline_compile_options.usesMotionBlur = false;
	pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipeline_compile_options.numPayloadValues = 2;
	pipeline_compile_options.numAttributeValues = 2;
	pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
	pipeline_compile_options.pipelineLaunchParamsVariableName = "packet";
	pipeline_compile_options.usesPrimitiveTypeFlags = 0;
	return pipeline_compile_options;
}

static OptixPipelineLinkOptions &pipeline_link_options()
{
	static OptixPipelineLinkOptions pipeline_link_options {};
	pipeline_link_options.maxTraceDepth = 1;
	return pipeline_link_options;
}

// TODO: source
inline OptixModule optix_module_from_source(OptixDeviceContext optix_context, const std::filesystem::path &path)
{
	OptixModule module = nullptr;

	const std::string ptx = read_file(path);
	optixModuleCreate
	(
		optix_context,
		&module_compile_options(),
		&pipeline_compile_options(),
		ptx.data(), ptx.size(),
		nullptr, 0,
		&module
	);

	return module;
}

template <size_t N>
std::array <OptixProgramGroup, N> optix_program_groups
(
	OptixDeviceContext optix_context,
	OptixModule module,
	const std::array <OptixProgramType, N> &types
)
{
	std::array <OptixProgramGroupDesc, N> descriptions {};
	for (size_t i = 0; i < N; i++) {
		OptixProgramGroupDesc &description = descriptions[i];

		OptixProgramType ptype = types[i];
		switch (ptype.type) {
		case OptixProgramType::eRayGeneration:
			description.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			description.raygen.module = module;
			description.raygen.entryFunctionName = ptype.entry;
			break;
		case OptixProgramType::eClosestHit:
			description.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			description.hitgroup.moduleCH = module;
			description.hitgroup.entryFunctionNameCH = ptype.entry;
			break;
		case OptixProgramType::eMiss:
			description.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			description.miss.module = module;
			description.miss.entryFunctionName = ptype.entry;
			break;
		default:
			break;
		}
	}

	std::array <OptixProgramGroup, N> program_groups;

	OptixProgramGroupOptions program_options {};
	optixProgramGroupCreate
	(
		optix_context,
		descriptions.data(), N,
		&program_options, nullptr, 0,
		program_groups.data()
	);

	return program_groups;
}
