#include <pybind11/pybind11.h>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl/filesystem.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

#include <cuda_runtime.h>

#include <optix/optix_function_table_definition.h>

#include <oak/mesh.hpp>
#include <oak/camera.hpp>
#include <oak/transform.hpp>

#include "io.hpp"
#include "cuda/optix/util.cuh"
#include "cuda/optix/acceleration.cuh"
#include "cuda/vector_math.cuh"
#include "shaders/optix/ssdfg.cuh"

namespace py = pybind11;

using Extent = std::pair <int64_t, int64_t>;

struct OptixCentral {
	OptixDeviceContext context;

	static OptixCentral from() {
		OptixDeviceContext context = make_context();

		return OptixCentral {
			.context = context
		};
	}

	static OptixCentral one() {
		static std::optional <OptixCentral> one;
		if (!one)
			one = from();

		return one.value();
	}
};

struct RaytracingHandle {
	OptixTraversableHandle gas;

	float3 *positions;
	uint3 *triangles;
	uint count;
};

struct OptixPipelinePackage {
	OptixPipeline pipeline;
	std::vector <OptixProgramGroup> groups;
};

// slsg = Silhouette Level-Set Gradients
OptixPipelinePackage load_slsg_package()
{
	OptixModule module = optix_module_from_source(OptixCentral::one().context, VYNE_ROOT "/bin/ssdfg.cu.o");

	// Load program groups
	auto program_groups = optix_program_groups
	(
		OptixCentral::one().context, module,
		std::array <OptixProgramType, 3> {
			OptixProgramType::ray_generation("__raygen__"),
			OptixProgramType::closest_hit("__closesthit__"),
			OptixProgramType::miss("__miss__"),
		}
	);

	OptixPipeline pipeline = nullptr;

	optixPipelineCreate
	(
		OptixCentral::one().context,
		&pipeline_compile_options(),
		&pipeline_link_options(),
		program_groups.data(), program_groups.size(),
		nullptr, 0,
		&pipeline
	);

	return OptixPipelinePackage {
		.pipeline = pipeline,
		.groups = {
			program_groups[0],
			program_groups[1],
			program_groups[2]
		}
	};
}

enum CUDAType {
	eInt32,
	eFloat,
	eFloat2,
	eFloat3
};

template <typename T>
constexpr CUDAType translate_type() { return eInt32; }

template <> constexpr CUDAType translate_type <int32_t> () { return eInt32; }
template <> constexpr CUDAType translate_type <float> () { return eFloat; }
template <> constexpr CUDAType translate_type <float2> () { return eFloat2; }
template <> constexpr CUDAType translate_type <float3> () { return eFloat3; }

constexpr size_t bytesize(CUDAType type)
{
	switch (type) {
	case eFloat:
	case eInt32:
		return sizeof(int32_t);
	case eFloat2:
		return 2 * sizeof(int32_t);
	case eFloat3:
		return 3 * sizeof(int32_t);
	default:
		break;
	}

	return sizeof(int32_t);
}

struct DeviceBuffer {
	std::vector <size_t> shape;
	CUdeviceptr ptr;
	CUDAType type;

	size_t numel() const {
		size_t N = 1;
		for (size_t s : shape)
			N *= s;
		return N;
	}

	size_t device_size() const {
		return numel() * bytesize(type);
	}

	std::vector <size_t> strides() const {
		std::vector <size_t> strides;

		size_t next = bytesize(type);
		for (int64_t i = shape.size() - 1; i >= 0; i--) {
			strides.insert(strides.begin(), next);
			next *= shape[i];
		}

		return strides;
	}

	template <typename t>
	t *as() {
		return (t *) ptr;
	}

	template <typename T>
	const T *as() const {
		return (T *) ptr;
	}

	template <typename T>
	static DeviceBuffer from(const std::vector <size_t> &shape) {
		DeviceBuffer db;
		db.shape = shape;
		db.ptr = cuda_alloc(sizeof(T) * db.numel());
		db.type = translate_type <T> ();
		return db;
	}

	template <typename T>
	static DeviceBuffer from(const Extent &extent) {
		std::vector <size_t> shape {
			(size_t) extent.first,
			(size_t) extent.second
		};

		return from <T> (shape);
	}
};

__global__
void kmemset(float *__restrict__ ptr, size_t elements)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t i = tid; i < elements; i += stride)
		ptr[i] = 1.0f;
}

std::tuple <
        DeviceBuffer
> render(const RaytracingHandle &handle, const Camera &camera, const Transform &camera_transform, const Extent &extent)
{
	auto package = load_slsg_package();

	// TODO: another method
	// Make the shader binding table records
	Record <float> raygen_record;
	optixSbtRecordPackHeader(package.groups[0], &raygen_record);
	CUdeviceptr raygen_record_sbt = cuda_element_buffer(raygen_record);

	Record <float> closesthit_record;
	optixSbtRecordPackHeader(package.groups[1], &closesthit_record);
	CUdeviceptr closesthit_record_sbt = cuda_element_buffer(closesthit_record);

	Record <float> miss_record;
	optixSbtRecordPackHeader(package.groups[2], &miss_record);
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

	// Allocate the necessary buffers
	auto visibility = DeviceBuffer::from <float> (extent);
	auto depth = DeviceBuffer::from <float> (extent);
	auto id = DeviceBuffer::from <int32_t> (extent);
	auto barycentrics = DeviceBuffer::from <float2> (extent);
	auto render = DeviceBuffer::from <float3> (extent);

	// Construct packet for raytracing
	Packet packet;

	RayFrame rayframe = camera.rayframe(camera_transform);

	packet.gas = handle.gas;

	packet.origin = glm_to_float3(rayframe.origin);
	packet.lower_left = glm_to_float3(rayframe.lower_left);
	packet.horizontal = glm_to_float3(rayframe.horizontal);
	packet.vertical = glm_to_float3(rayframe.vertical);

	packet.visibility = visibility.as <float> ();
	packet.depth = depth.as <float> ();
	packet.primitive = id.as <int32_t> ();
	packet.barycentrics = barycentrics.as <float2> ();
	packet.render = render.as <float3> ();

	packet.resolution = make_uint2(extent.first, extent.second);

	CUdeviceptr device_packet = cuda_element_buffer(packet);

	optixLaunch
	(
		package.pipeline, nullptr,
		device_packet, sizeof(Packet), &sbt,
		extent.first, extent.second, 1
	);

	cudaDeviceSynchronize();

	return { visibility };
}

PYBIND11_MODULE(vyne, m)
{
	// TODO: disable optix logging by default...
	py::class_ <Mesh> (m, "Mesh")
	        .def("rtx", [](const Mesh &m) {
			RaytracingHandle handle;
			handle.gas = gas_from_mesh(OptixCentral::one().context, m);
			handle.positions = (float3 *) cuda_vector_buffer(m.positions);
			handle.triangles = (uint3 *) cuda_vector_buffer(m.triangles);
			handle.count = m.triangles.size();
			return handle;
		})
		.def("load", &load_geometry)
	        .def("__repr__", [](const Mesh &m) {
			return "Mesh("
				+ std::to_string(m.positions.size()) + " vertices, "
				+ std::to_string(m.triangles.size()) + " triangles)";
		});

	py::class_ <RaytracingHandle> (m, "RaytracingHandle");

	py::class_ <DeviceBuffer> (m, "DeviceBuffer")
	        .def("numpy", [](const DeviceBuffer &db) {
			auto result = py::array_t <float, py::array::c_style> (db.numel());
			float *ptr = result.mutable_data();
			cudaMemcpy(ptr, db.as <float> (), db.numel() * sizeof(float), cudaMemcpyDeviceToHost);
			return result.reshape(db.shape);
		});

	py::class_ <glm::vec3> (m, "vec3")
	        .def(py::init <> ())
	        .def(py::init <float, float, float> ())
	        .def_readwrite("x", &glm::vec3::x)
	        .def_readwrite("y", &glm::vec3::y)
	        .def_readwrite("z", &glm::vec3::z)
		.def("__repr__", &glm::to_string <glm::vec3>);

	py::class_ <Transform> (m, "Transform")
	        .def(py::init <> ())
	        .def(py::init([](const glm::vec3 &position) {
			Transform transform;
			transform.position = position;
			return transform;
		}))
	        .def_readwrite("position", &Transform::position)
	        .def_readwrite("rotation", &Transform::rotation)
	        .def_readwrite("scale", &Transform::scale);

	py::class_ <Camera> (m, "Camera")
	        .def(py::init <> ())
	        .def(py::init([](float aspect) {
			Camera camera;
			camera.from(aspect);
			return camera;
		}))
	        .def_readwrite("aspect", &Camera::aspect)
	        .def_readwrite("fov", &Camera::fov)
	        .def_readwrite("near", &Camera::near)
	        .def_readwrite("far", &Camera::far);

	m.def("render", &render);
}