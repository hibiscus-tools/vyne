#include <vector>
#include <string>
#include <map>

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

#include "cuda/optix/acceleration.cuh"
#include "cuda/optix/util.cuh"
#include "cuda/vector_math.cuh"
#include "io.hpp"
#include "shaders/optix/ssdfg.cuh"
// TODO: refactor to slsg
#include "ssdfg/kernels.cuh"

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

	uint vertices;
	uint primitives;
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

	DeviceBuffer() : ptr(0) {}

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

	static void free(DeviceBuffer &db) {
		cudaFree((void *) db.ptr);
		db.ptr = 0;
		db.shape = {};
	}
};

std::string to_string(const DeviceBuffer &db)
{
	std::string result = "DeviceBuffer <";
	switch (db.type) {
	case eInt32:
		result += "Int32";
		break;
	case eFloat:
		result += "Float";
		break;
	case eFloat2:
		result += "Float2";
		break;
	case eFloat3:
		result += "Float3";
		break;
	}
	result += "> (";
	for (size_t i = 0; i < db.shape.size(); i++) {
		result += std::to_string(db.shape[i]);
		if (i + 1 < db.shape.size())
			result += ", ";
	}
	return result + ")";
}

// TODO: struct output
struct RenderResult {
	DeviceBuffer visibility;
	DeviceBuffer depth;
	DeviceBuffer index;
	DeviceBuffer barycentrics;
};

RenderResult render
(
	const RaytracingHandle &handle,
	const Camera &camera,
	const Transform &camera_transform,
	const Extent &extent
)
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
	auto index = DeviceBuffer::from <int32_t> (extent);
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
	packet.index = index.as <int32_t> ();
	packet.barycentrics = barycentrics.as <float2> ();

	packet.resolution = make_uint2(extent.first, extent.second);

	CUdeviceptr device_packet = cuda_element_buffer(packet);

	optixLaunch
	(
		package.pipeline, nullptr,
		device_packet, sizeof(Packet), &sbt,
		extent.first, extent.second, 1
	);

	cudaDeviceSynchronize();

	return { visibility, depth, index, barycentrics };
}

std::tuple <DeviceBuffer, DeviceBuffer, DeviceBuffer> evaluate_sdf(const DeviceBuffer &silhouette)
{
	auto boundary = DeviceBuffer::from <float> (silhouette.shape);
	auto sdf = DeviceBuffer::from <float> (silhouette.shape);
	auto index = DeviceBuffer::from <int32_t> (silhouette.shape);

	int32_t width = silhouette.shape[0];
	int32_t height = silhouette.shape[1];

	silhouette_edges <<< 64, 64 >>>
	(silhouette.as <float> (), boundary.as <float> (), width, height);

	signed_distance_field <<< 64, 64 >>>
	(silhouette.as <float> (), boundary.as <float> (), sdf.as <float> (), index.as <int32_t> (), width, height);

	cudaDeviceSynchronize();

	return { boundary, sdf, index };
}

DeviceBuffer evaluate_sdf_gradient(const DeviceBuffer &sdf)
{
	auto gradient = DeviceBuffer::from <float2> (sdf.shape);
	sdf_spatial_gradient <<< 64, 64 >>> (sdf.as <float> (), gradient.as <float2> (), sdf.shape[0], sdf.shape[1]);
	cudaDeviceSynchronize();
	return gradient;
}

__global__
void kernel_evaluate_boundary_gradient
(
	const float *__restrict__ target_sdf,
	const float *__restrict__ source_sdf,
	const int32_t *__restrict__ index,
	float2 *__restrict__ gradients,
	int32_t width,
	int32_t height
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	float2 extent = make_float2(width, height);
	for (int32_t i = tid; i < width * height; i += stride) {
		int32_t idx = index[i];
		if (i == idx) {
			gradients[i] = make_float2(0, 0);
			continue;
		}

		float2 u = make_float2(i % width, i / width)/extent;
		float2 alpha = make_float2(idx % width, idx / width)/extent;
		float2 d = u - alpha;

		gradients[i] = -2 * (target_sdf[i] - source_sdf[i]) * normalize(d);
//		gradients[i] = -2 * (target_sdf[i] - source_sdf[i]) * d;
	}
}

DeviceBuffer evaluate_boundary_gradient(const DeviceBuffer &target_sdf, const DeviceBuffer &source_sdf, const DeviceBuffer &index)
{
	auto gradients = DeviceBuffer::from <float2> (index.shape);
	kernel_evaluate_boundary_gradient <<< 64, 64 >>>
	(target_sdf.as <float> (), source_sdf.as <float> (), index.as <int32_t> (), gradients.as <float2> (), gradients.shape[0], gradients.shape[1]);
	cudaDeviceSynchronize();
	return gradients;
}

__global__
void kernel_gather_to_boundary
(
	const float2 *__restrict__ delta,
	const int32_t *__restrict__ index,
	float2 *__restrict__ dst,
	int32_t *__restrict__ counter,
	int32_t width,
	int32_t height
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = tid; i < width * height; i += stride) {
		float2 d = delta[i];
		int32_t idx = index[i];

		atomicAdd(&dst[idx].x, d.x);
		atomicAdd(&dst[idx].y, d.y);
		atomicAdd(&counter[idx], 1);
	}
}

// TODO: replace with arithmetic operators for devicebuffers (e.g. DeviceBuffer::_add)
__global__
void kernel_average_boundary_gradients
(
	const int32_t *__restrict__ counts,
	float2 *__restrict__ gradients,
	int32_t width,
	int32_t height
)
{

	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = tid; i < width * height; i += stride) {
		if (counts[i] > 0)
			gradients[i] = gradients[i] / float(counts[i]);
	}
}

template <typename T>
__global__
void set(T *__restrict__ ptr, size_t size, T value)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = tid; i < size; i += stride)
		ptr[i] = value;
}

DeviceBuffer integrate_to_boundary(const DeviceBuffer &gradients, const DeviceBuffer &indices)
{
	// TODO: size (and type) checking

	// TODO: memset
	auto integrated = DeviceBuffer::from <float2> (gradients.shape);
	auto counters = DeviceBuffer::from <int32_t> (gradients.shape);

	set <<< 64, 64 >>> (integrated.as <float2> (), integrated.numel(), make_float2(0, 0));
	set <<< 64, 64 >>> (counters.as <int32_t> (), integrated.numel(), 0);

	kernel_gather_to_boundary <<< 64, 64 >>>
	(
		gradients.as <float2> (), indices.as <int32_t> (),
		integrated.as <float2> (), counters.as <int32_t> (),
		gradients.shape[0], gradients.shape[1]
	);

	kernel_average_boundary_gradients <<< 64, 64 >>>
	(
		counters.as <int32_t> (),
		integrated.as <float2> (),
		gradients.shape[0], gradients.shape[1]
	);

	cudaDeviceSynchronize();

	return integrated;
}

__global__
void kernel_scatter_from_boundary
(
	const float2 *__restrict__ gradients,
	const int32_t *__restrict__ index,
	float2 *__restrict__ dst,
	int32_t width,
	int32_t height
)
{

	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = tid; i < width * height; i += stride) {
		dst[i] = gradients[index[i]];
	}
}

DeviceBuffer scatter_from_boundary(const DeviceBuffer &gradients, const DeviceBuffer &index)
{
	auto scattered = DeviceBuffer::from <float2> (gradients.shape);
	kernel_scatter_from_boundary <<< 64, 64 >>>
	(gradients.as <float2> (), index.as <int32_t> (), scattered.as <float2> (), gradients.shape[0], gradients.shape[1]);
	cudaDeviceSynchronize();
	return scattered;
}

__global__
void kernel_diffuse
(
	const float2 *__restrict__ src,
	const float *__restrict bdy,
	float2 *__restrict__ dst,
	int32_t width,
	int32_t height
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = tid; i < width * height; i += stride) {
		static constexpr int32_t dx[] = { 1, 1, -1, -1 };
		static constexpr int32_t dy[] = { 1, -1, 1, -1 };

		if (bdy[i] > 0) {
			dst[i] = src[i];
			continue;
		}

		int32_t x = i % width;
		int32_t y = i / width;

		float2 sum = make_float2(0, 0);
		for (uint32_t j = 0; j < 4; j++) {
			int32_t nx = x + dx[j];
			int32_t ny = y + dy[j];

			if (nx < 0 || nx >= width)
				continue;

			if (ny < 0 || ny >= height)
				continue;

			int32_t ni = nx + ny * width;

			sum = sum + src[ni];
		}

		dst[i] = sum/4.0f;
	}
}

DeviceBuffer diffuse_from_boundary(const DeviceBuffer &gradients, const DeviceBuffer &boundary, int32_t iterations)
{
	auto copy0 = DeviceBuffer::from <float2> (gradients.shape);
	auto copy1 = DeviceBuffer::from <float2> (gradients.shape);

	cudaMemcpy((void *) copy1.ptr, (void *) gradients.ptr, sizeof(float2) * gradients.numel(), cudaMemcpyDeviceToDevice);

	for (int32_t i = 0; i < iterations; i++) {
		kernel_diffuse <<< 64, 64 >>>
		(copy1.as <float2> (), boundary.as <float> (), copy0.as <float2> (), gradients.shape[0], gradients.shape[1]);

		kernel_diffuse <<< 64, 64 >>>
		(copy0.as <float2> (), boundary.as <float> (), copy1.as <float2> (), gradients.shape[0], gradients.shape[1]);
	}

	cudaDeviceSynchronize();

	DeviceBuffer::free(copy0);

	return copy1;
}

__global__
void kernel_backproject_vectors
(
	const float *__restrict__ visibility,
	const float *__restrict__ depth,
	const float2 *__restrict__ vectors,
	float3 *__restrict__ projected,
	float3 horizontal,
	float3 vertical,
	uint32_t width,
	uint32_t height
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = tid; i < width * height; i += stride) {
		if (visibility[i] > 0) {
			float2 vector = vectors[i] * depth[i];
			// TODO: negate the y direction?
			projected[i] = vector.x * horizontal + vector.y * vertical;
		} else {
			projected[i] = make_float3(0, 0, 0);
		}
	}
}

// TODO: encode vis with depth < 0
DeviceBuffer back_project_vectors(const DeviceBuffer &gradients, const DeviceBuffer &visibility, const DeviceBuffer &depth, const Camera &camera, const Transform &camera_transform)
{
	auto projected_gradients = DeviceBuffer::from <float3> (gradients.shape);

	RayFrame rayframe = camera.rayframe(camera_transform);
	kernel_backproject_vectors <<< 64, 64 >>>
	(
		visibility.as <float> (),
		depth.as <float> (),
		gradients.as <float2> (),
		projected_gradients.as <float3> (),
		glm_to_float3(rayframe.horizontal),
		glm_to_float3(rayframe.vertical),
		gradients.shape[0],
		gradients.shape[1]
	);

	cudaDeviceSynchronize();

	return projected_gradients;
}

__forceinline__ __device__
float3 atomicAdd(float3 *addr, float3 val)
{
	float x = atomicAdd(&addr->x, val.x);
	float y = atomicAdd(&addr->y, val.y);
	float z = atomicAdd(&addr->z, val.z);
	return make_float3(x, y, z);
}

__global__
void kernel_gather_mesh_gradients
(
	const float3 *__restrict__ projected,
	const float2 *__restrict__ barycentrics,
	const int32_t *__restrict__ ids,
	const uint3 *__restrict__ triangles,
	float3 *__restrict__ gradients,
	int32_t *__restrict__ counter,
	uint32_t width,
	uint32_t height
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = tid; i < width * height; i += stride) {
		float3 proj = projected[i];
		if (length(proj) < 1e-6)
			continue;

		uint index = ids[i];
		uint3 triangle = triangles[index];
		float2 bary = barycentrics[i];

		float3 v0 = (1 - bary.x - bary.y) * proj;
		float3 v1 = bary.x * proj;
		float3 v2 = bary.y * proj;

		atomicAdd(&gradients[triangle.x], v0);
		atomicAdd(&gradients[triangle.y], v1);
		atomicAdd(&gradients[triangle.z], v2);

		atomicAdd(&counter[triangle.x], 1);
		atomicAdd(&counter[triangle.y], 1);
		atomicAdd(&counter[triangle.z], 1);
	}
}

// TODO: template division
__global__
void kernel_average_mesh_gradients
(
	const int32_t *__restrict__ counts,
	float3 *__restrict__ gradients,
	uint32_t size
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = tid; i < size; i += stride)
		gradients[i] = gradients[i] / fmax(1.0f, float(counts[i]));
}

DeviceBuffer gather_mesh_gradients(const DeviceBuffer &projected, const DeviceBuffer &barycentrics, const DeviceBuffer &ids, const RaytracingHandle &handle)
{
	auto gradients = DeviceBuffer::from <float3> ({ handle.vertices });
	auto counter = DeviceBuffer::from <int32_t> ({ handle.vertices });

	kernel_gather_mesh_gradients <<< 64, 64 >>>
	(
		projected.as <float3> (),
		barycentrics.as <float2> (),
		ids.as <int32_t> (),
		handle.triangles,
		gradients.as <float3> (),
		counter.as <int32_t> (),
		projected.shape[0],
		projected.shape[1]
	);

	kernel_average_mesh_gradients <<< 64, 64 >>>
	(counter.as <int32_t> (), gradients.as <float3> (), handle.vertices);

	cudaDeviceSynchronize();

	return gradients;
}

struct MeshOptimizer {
	Mesh &dst;

	std::vector <glm::vec3> positions;

	float lr;

	explicit MeshOptimizer(Mesh &ref, float lr) : dst(ref), lr(lr) {
		positions.resize(dst.positions.size(), glm::vec3(0.0f));
	}

	void feed(const std::map <std::string, py::array_t <float, py::array::c_style>> &fields) {
		for (const auto &[field, delta] : fields) {
			if (field == "positions") {
				size_t ndim = delta.ndim();
				assert(ndim == 2);

				auto shape = delta.shape();
				if (shape[1] != 3)
					ulog_error("MeshOptimizer::feed", "Field \'positions\' expected buffer of shape (X, 3)\n");
				if (shape[0] != dst.positions.size())
					ulog_error("MeshOptimizer::feed", "Field \'positions\' expected to have same size as mesh positions\n");

				std::memcpy(positions.data(), delta.data(), sizeof(glm::vec3) * positions.size());
			} else {
				ulog_error("MeshOptimizer::feed", "Unexpected field: %s\n", field.c_str());
			}
		}
	}

	void step() {
		for (int32_t i = 0; i < positions.size(); i++)
			dst.positions[i] -= lr * positions[i];
	}
};

PYBIND11_MODULE(vyne, m)
{
	// TODO: disable optix logging by default...
	py::class_ <Mesh> (m, "Mesh")
	        .def("triangles", [](const Mesh &m) {
			py::array_t <int32_t, py::array::c_style> triangles(3 * m.triangles.size());
			std::memcpy(triangles.mutable_data(), m.triangles.data(), sizeof(glm::ivec3) * m.triangles.size());
			return triangles.reshape(std::vector <size_t> { m.triangles.size(), 3 });
		})
	        .def("rtx", [](const Mesh &m) {
			RaytracingHandle handle;
			handle.gas = gas_from_mesh(OptixCentral::one().context, m);
			handle.positions = (float3 *) cuda_vector_buffer(m.positions);
			handle.triangles = (uint3 *) cuda_vector_buffer(m.triangles);
			handle.vertices = m.positions.size();
			handle.primitives = m.triangles.size();
			return handle;
		})
		.def("deduplicate", [](const Mesh &m) {
			return m.deduplicate();
		})
		.def("load", &load_geometry)
	        .def("__repr__", [](const Mesh &m) {
			return "Mesh("
				+ std::to_string(m.positions.size()) + " vertices, "
				+ std::to_string(m.triangles.size()) + " triangles)";
		});

	py::class_ <RaytracingHandle> (m, "RaytracingHandle");

	py::class_ <DeviceBuffer> (m, "DeviceBuffer")
	        .def(py::init([](const py::array_t <float, py::array::c_style> &buffer) {
			size_t ndim = buffer.ndim();
			assert(ndim != 0);

			// Check promotability
			auto shape = buffer.shape();
			size_t last = shape[ndim - 1];

			DeviceBuffer db;
			if (last == 1) {
				std::vector <size_t> shape_vec;
				for (int32_t i = 0; i < ndim - 1; i++)
					shape_vec.push_back(shape[i]);
				db = DeviceBuffer::from <float> (shape_vec);
			} else if (last == 2) {
				std::vector <size_t> shape_vec;
				for (int32_t i = 0; i < ndim - 1; i++)
					shape_vec.push_back(shape[i]);
				db = DeviceBuffer::from <float2> (shape_vec);
			} else if (last == 3) {
				std::vector <size_t> shape_vec;
				for (int32_t i = 0; i < ndim - 1; i++)
					shape_vec.push_back(shape[i]);
				db = DeviceBuffer::from <float3> (shape_vec);
			} else {
				std::vector <size_t> shape_vec;
				for (int32_t i = 0; i < ndim; i++)
					shape_vec.push_back(shape[i]);
				db = DeviceBuffer::from <float> (shape_vec);
			}

			cudaMemcpy((void *) db.ptr, buffer.data(), db.device_size(), cudaMemcpyHostToDevice);
			return db;
		}))
	        .def("numpy", [](const DeviceBuffer &db) {
			// TODO: function
			auto int32_buffer = [&]() {
				auto result = py::array_t <int32_t, py::array::c_style> (db.numel());
				int32_t *ptr = result.mutable_data();
				cudaMemcpy(ptr, db.as <int32_t> (), db.numel() * sizeof(int32_t), cudaMemcpyDeviceToHost);
				return result.reshape(db.shape);
			};

			auto float_buffer = [&]() {
				auto result = py::array_t <float, py::array::c_style> (db.numel());
				float *ptr = result.mutable_data();
				cudaMemcpy(ptr, db.as <float> (), db.numel() * sizeof(float), cudaMemcpyDeviceToHost);
				return result.reshape(db.shape);
			};

			auto float2_buffer = [&]() {
				auto result = py::array_t <float, py::array::c_style> (2 * db.numel());
				float *ptr = result.mutable_data();
				cudaMemcpy(ptr, db.as <float> (), 2 * db.numel() * sizeof(float), cudaMemcpyDeviceToHost);
				auto new_shape = db.shape;
				new_shape.push_back(2);
				return result.reshape(new_shape);
			};

			auto float3_buffer = [&]() {
				auto result = py::array_t <float, py::array::c_style> (3 * db.numel());
				float *ptr = result.mutable_data();
				cudaMemcpy(ptr, db.as <float> (), 3 * db.numel() * sizeof(float), cudaMemcpyDeviceToHost);
				auto new_shape = db.shape;
				new_shape.push_back(3);
				return result.reshape(new_shape);
			};

			switch (db.type) {
			case eInt32:
				return int32_buffer();
			case eFloat:
				return float_buffer();
			case eFloat2:
				return float2_buffer();
			case eFloat3:
				return float3_buffer();
			default:
				break;
			}

			ulog_error("DeviceBuffer::numpy", "Unable to translate instance: unknown type!");
			throw std::runtime_error("DeviceBuffer::numpy");
		})
		.def("__exit__", DeviceBuffer::free)
		.def("__repr__", &to_string);

	py::class_ <RenderResult> (m, "RenderResult")
	        .def_readonly("visibility", &RenderResult::visibility)
	        .def_readonly("depth", &RenderResult::depth)
	        .def_readonly("index", &RenderResult::index)
	        .def_readonly("barycentrics", &RenderResult::barycentrics);

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

	py::class_ <MeshOptimizer> (m, "MeshOptimizer")
	        .def(py::init <Mesh &, float> ())
		.def("feed", &MeshOptimizer::feed)
		.def("step", &MeshOptimizer::step);

	// TODO: define inside an slsg module
	m.def("render", &render);
	m.def("evaluate_sdf", &evaluate_sdf);
	m.def("evaluate_sdf_gradient", &evaluate_sdf_gradient);
	m.def("evaluate_boundary_gradient", &evaluate_boundary_gradient);
	m.def("integrate_to_boundary", &integrate_to_boundary);
	m.def("scatter_from_boundary", &scatter_from_boundary);
	m.def("diffuse_from_boundary", &diffuse_from_boundary);
	m.def("back_project_vectors", &back_project_vectors);
	m.def("gather_mesh_gradients", &gather_mesh_gradients);
}