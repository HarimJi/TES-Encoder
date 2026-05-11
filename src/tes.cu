#include "tes.cuh"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include <cuda/functional>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

#include "barycentric_utils.cuh"
#include "cuda_error.cuh"
#include "geometry_utils.cuh"
#include "itypes.cuh"
#include "vector4d.cuh"

namespace tes {

namespace {

constexpr int K_BLOCK_SIZE = 256;

__device__ inline double atomic_min_double(double* addr, double const value) {
  unsigned long long* ull = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    double const old_val = __longlong_as_double(assumed);
    if (old_val <= value) {
      break;
    }
    old = atomicCAS(ull, assumed, __double_as_longlong(value));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__global__ void distance_to_mesh_kernel(vector4d const p, vector4d const* points_d,
                                        iface_t const* faces_d, int const number_of_faces,
                                        double* distances_d) {
  int const face_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (face_index >= number_of_faces) {
    return;
  }
  iface_t const f = faces_d[face_index];
  vector4d const c = project_point_to_triangle(p, points_d[f.x], points_d[f.y], points_d[f.z]);
  distances_d[face_index] = norm(p - c);
}

__global__ void prune_faces_with_lipschitz_kernel(vector4d const* points_d, iface_t const* faces_d,
                                                  vector4d const center, double const distance_at_center,
                                                  double const radius, int const number_of_faces,
                                                  int* atomic_count_d, int* remaining_face_indices_d) {
  int const face_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (face_index >= number_of_faces) {
    return;
  }
  iface_t const f = faces_d[face_index];
  vector4d const c = project_point_to_triangle(center, points_d[f.x], points_d[f.y], points_d[f.z]);
  double const d = norm(center - c);
  if (d - distance_at_center <= 2.0 * radius) {
    int const slot = atomicAdd(atomic_count_d, 1);
    remaining_face_indices_d[slot] = face_index;
  }
}

__global__ void importance_kernel(vector4d const t_w, vector4d const t_x, vector4d const t_y,
                                  vector4d const t_z, vector4d const* points_d, iface_t const* faces_d,
                                  int const* remaining_faces_d, int const quality,
                                  int const number_of_remaining_faces, double const epsilon,
                                  double* importances_d) {
  int const grid_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (grid_index >= barycentric_grid_size(quality)) {
    return;
  }

  int3 const ijk = barycentric_index(grid_index, quality);
  double const inv = 1.0 / static_cast<double>(quality - 1);
  double const alpha = ijk.x * inv;
  double const beta = ijk.y * inv;
  double const gamma = ijk.z * inv;
  vector4d const x = alpha * t_x + beta * t_y + gamma * t_z + (1.0 - alpha - beta - gamma) * t_w;

  double winning_distance_sq = 0;
  int winning_local_index = 0;
  for (int i = 0; i < number_of_remaining_faces; ++i) {
    iface_t const f = faces_d[remaining_faces_d[i]];
    vector4d const c = project_point_to_triangle(x, points_d[f.x], points_d[f.y], points_d[f.z]);
    double const d_sq = norm_squared(x - c);
    if (i == 0 || d_sq < winning_distance_sq) {
      winning_distance_sq = d_sq;
      winning_local_index = i;
    }
  }

  if (epsilon == 0.0) {
    atomic_min_double(importances_d + winning_local_index, winning_distance_sq);
    return;
  }

  for (int i = 0; i < number_of_remaining_faces; ++i) {
    iface_t const f = faces_d[remaining_faces_d[i]];
    vector4d const c = project_point_to_triangle(x, points_d[f.x], points_d[f.y], points_d[f.z]);
    double const d_sq = norm_squared(x - c);
    if (d_sq <= winning_distance_sq + epsilon) {
      atomic_min_double(importances_d + i, d_sq);
    }
  }
}

double calculate_distance_to_mesh(vector4d const p, thrust::device_vector<vector4d> const& points_d,
                                  thrust::device_vector<iface_t> const& faces_d,
                                  thrust::device_vector<double>& distance_buffer_d) {
  if (faces_d.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  if (distance_buffer_d.size() < faces_d.size()) {
    distance_buffer_d.resize(faces_d.size());
  }
  int const grid_size = (faces_d.size() + K_BLOCK_SIZE - 1) / K_BLOCK_SIZE;
  distance_to_mesh_kernel<<<grid_size, K_BLOCK_SIZE>>>(p, thrust::raw_pointer_cast(points_d.data()),
                                                       thrust::raw_pointer_cast(faces_d.data()),
                                                       static_cast<int>(faces_d.size()),
                                                       thrust::raw_pointer_cast(distance_buffer_d.data()));
  CUDA_CHECK_KERNEL();
  return thrust::reduce(distance_buffer_d.begin(), distance_buffer_d.begin() + faces_d.size(),
                        std::numeric_limits<double>::infinity(), cuda::minimum<double>{});
}

boundary_features_t compute_boundary_features(itetra_t const t, int const number_of_surface_nodes,
                                              MSH& reference_msh) {
  boundary_features_t features;
  features.vertices = {-1, -1, -1, -1};
  features.edges = {make_edge(-1, -1), make_edge(-1, -1), make_edge(-1, -1),
                    make_edge(-1, -1), make_edge(-1, -1), make_edge(-1, -1)};
  features.faces = {-1, -1, -1, -1};

  std::array<int, 4> const v = {t.w, t.x, t.y, t.z};
  std::array<iedge_t, 6> const edges = {make_edge(v[0], v[1]), make_edge(v[0], v[2]),
                                        make_edge(v[0], v[3]), make_edge(v[1], v[2]),
                                        make_edge(v[1], v[3]), make_edge(v[2], v[3])};
  std::array<iface_t, 4> const faces = {make_face(v[0], v[1], v[2]), make_face(v[0], v[1], v[3]),
                                        make_face(v[0], v[2], v[3]), make_face(v[1], v[2], v[3])};

  int counter = 0;
  for (int i = 0; i < 4; ++i) {
    if (v[i] < number_of_surface_nodes) {
      features.vertices[counter++] = v[i];
    }
  }

  counter = 0;
  for (int i = 0; i < 6; ++i) {
    if (reference_msh.Edge_To_Face_Map.find(edges[i]) != reference_msh.Edge_To_Face_Map.end()) {
      features.edges[counter++] = edges[i];
    }
  }

  counter = 0;
  for (int i = 0; i < 4; ++i) {
    int const face_index = reference_msh.Find_Surface_Face_Index(faces[i]);
    if (face_index >= 0) {
      features.faces[counter++] = face_index;
    }
  }

  return features;
}

}  // namespace

bool TES::Load_Inner_MSH(std::string const& file_name, MSH::msh_file_type_e type) {
  return Inner_MSH.Load(file_name, type) && Inner_MSH.Construct_Maps();
}

bool TES::Load_Outer_MSH(std::string const& file_name, MSH::msh_file_type_e type) {
  return Outer_MSH.Load(file_name, type) && Outer_MSH.Construct_Maps();
}

bool TES::Make_Geometry() {
  if (!Inner_MSH.Inspect()) {
    std::cerr << "[ERROR][TES::Make_Geometry] Inner MSH is invalid\n";
    return false;
  }

  bool const has_outer = Outer_MSH.Inspect();
  if (!has_outer) {
    std::cout << "[DEBUG][TES::Make_Geometry] Outer MSH not provided -> using inner only\n";
  }

  // Map each outer-MSH node index to a TES node index. Common nodes (lying on the inner surface)
  // collapse to the inner-surface index; new outer nodes are appended after inner internal nodes.
  std::vector<int> node_index_map(Outer_MSH.Nodes.size(), -1);
  int number_of_outer_nodes = 0;
  for (int i = 0; i < static_cast<int>(Outer_MSH.Nodes.size()); ++i) {
    bool is_common = false;
    for (int j = 0; j < Inner_MSH.Number_Of_Surface_Nodes; ++j) {
      if (is_near(Outer_MSH.Nodes[i], Inner_MSH.Nodes[j])) {
        is_common = true;
        node_index_map[i] = j;
        break;
      }
    }
    if (!is_common) {
      node_index_map[i] = static_cast<int>(Inner_MSH.Nodes.size()) + number_of_outer_nodes;
      ++number_of_outer_nodes;
    }
  }

  Surface_Nodes.assign(Inner_MSH.Nodes.begin(),
                       Inner_MSH.Nodes.begin() + Inner_MSH.Number_Of_Surface_Nodes);
  Internal_Nodes.assign(Inner_MSH.Nodes.begin() + Inner_MSH.Number_Of_Surface_Nodes,
                        Inner_MSH.Nodes.end());
  Outer_Nodes.resize(number_of_outer_nodes);
  for (int i = 0; i < static_cast<int>(Outer_MSH.Nodes.size()); ++i) {
    int const mapped = node_index_map[i];
    if (mapped >= static_cast<int>(Inner_MSH.Nodes.size())) {
      Outer_Nodes[mapped - static_cast<int>(Inner_MSH.Nodes.size())] = Outer_MSH.Nodes[i];
    }
  }

  Faces = Inner_MSH.Faces;

  int const number_of_inner_surface_nodes = Inner_MSH.Number_Of_Surface_Nodes;

  for (itetra_t const t : Inner_MSH.Tetrahedras) {
    boundary_features_t const features = compute_boundary_features(t, number_of_inner_surface_nodes,
                                                                   Inner_MSH);
    bool const is_boundary = features.vertices[0] != -1;
    if (is_boundary) {
      Surface_Inner_Tetras.push_back(t);
      Surface_Inner_Boundary_Features.push_back(features);
    } else {
      Inner_Tetras.push_back(t);
    }
  }

  for (itetra_t const original : Outer_MSH.Tetrahedras) {
    itetra_t const remapped = make_tetra(node_index_map[original.w], node_index_map[original.x],
                                         node_index_map[original.y], node_index_map[original.z]);
    if (!is_valid(remapped)) {
      continue;
    }
    boundary_features_t const features = compute_boundary_features(remapped, number_of_inner_surface_nodes,
                                                                   Inner_MSH);
    bool const is_boundary = features.vertices[0] != -1;
    if (is_boundary) {
      Surface_Outer_Tetras.push_back(remapped);
      Surface_Outer_Boundary_Features.push_back(features);
    } else {
      Outer_Tetras.push_back(remapped);
    }
  }

  std::cout << "[DEBUG][TES::Make_Geometry] Success\n";
  return true;
}

bool TES::Encode_With_Optimization(int const grid_quality, double const alpha, double const epsilon,
                                   bool const is_deformable) {
  std::cout << "[DEBUG][TES::Encode_With_Optimization] Start\n";

  std::vector<itetra_t> tetras_host;
  tetras_host.reserve(Surface_Inner_Tetras.size() + Inner_Tetras.size() + Surface_Outer_Tetras.size()
                      + Outer_Tetras.size());
  tetras_host.insert(tetras_host.end(), Surface_Inner_Tetras.begin(), Surface_Inner_Tetras.end());
  tetras_host.insert(tetras_host.end(), Inner_Tetras.begin(), Inner_Tetras.end());
  tetras_host.insert(tetras_host.end(), Surface_Outer_Tetras.begin(), Surface_Outer_Tetras.end());
  tetras_host.insert(tetras_host.end(), Outer_Tetras.begin(), Outer_Tetras.end());

  std::vector<vector4d> points_host;
  points_host.reserve(Surface_Nodes.size() + Internal_Nodes.size() + Outer_Nodes.size());
  points_host.insert(points_host.end(), Surface_Nodes.begin(), Surface_Nodes.end());
  points_host.insert(points_host.end(), Internal_Nodes.begin(), Internal_Nodes.end());
  points_host.insert(points_host.end(), Outer_Nodes.begin(), Outer_Nodes.end());

  thrust::device_vector<iface_t> faces_d(Faces.begin(), Faces.end());
  thrust::device_vector<vector4d> points_d(points_host.begin(), points_host.end());

  Encoded_Faces.assign(tetras_host.size(), {});

  int* atomic_count_d = nullptr;
  cudaMalloc(&atomic_count_d, sizeof(int));
  CUDA_CHECK_LAST();

  thrust::device_vector<int> remaining_faces_d(Faces.size());
  std::vector<int> remaining_faces_host(Faces.size());
  thrust::device_vector<double> distance_buffer_d;
  thrust::device_vector<double> importance_d(Faces.size());

  int const number_of_surface_inner = static_cast<int>(Surface_Inner_Tetras.size());
  int const number_of_inner = static_cast<int>(Inner_Tetras.size());
  int const number_of_surface_outer = static_cast<int>(Surface_Outer_Tetras.size());

  auto const tik = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < static_cast<int>(tetras_host.size()); ++i) {
    itetra_t const t = tetras_host[i];
    vector4d const center = (points_host[t.w] + points_host[t.x] + points_host[t.y] + points_host[t.z])
                            / 4.0;
    double const radius = std::max({norm(center - points_host[t.w]), norm(center - points_host[t.x]),
                                    norm(center - points_host[t.y]), norm(center - points_host[t.z])});

    double const distance_at_center = calculate_distance_to_mesh(center, points_d, faces_d,
                                                                 distance_buffer_d);

    cudaMemset(atomic_count_d, 0, sizeof(int));
    CUDA_CHECK_LAST();

    int const number_of_faces = static_cast<int>(Faces.size());
    int prune_grid_size = (number_of_faces + K_BLOCK_SIZE - 1) / K_BLOCK_SIZE;
    if (prune_grid_size > 0) {
      prune_faces_with_lipschitz_kernel<<<prune_grid_size, K_BLOCK_SIZE>>>(
          thrust::raw_pointer_cast(points_d.data()), thrust::raw_pointer_cast(faces_d.data()), center,
          distance_at_center, radius, number_of_faces, atomic_count_d,
          thrust::raw_pointer_cast(remaining_faces_d.data()));
      CUDA_CHECK_KERNEL();
    }

    int number_of_remaining_faces = 0;
    cudaMemcpy(&number_of_remaining_faces, atomic_count_d, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST();
    thrust::copy(remaining_faces_d.begin(), remaining_faces_d.begin() + number_of_remaining_faces,
                 remaining_faces_host.begin());

    vector4d const t_w = alpha * points_host[t.w] + (1.0 - alpha) * center;
    vector4d const t_x = alpha * points_host[t.x] + (1.0 - alpha) * center;
    vector4d const t_y = alpha * points_host[t.y] + (1.0 - alpha) * center;
    vector4d const t_z = alpha * points_host[t.z] + (1.0 - alpha) * center;

    thrust::fill(importance_d.begin(), importance_d.begin() + number_of_remaining_faces,
                 std::numeric_limits<double>::infinity());

    int const importance_grid_size = (barycentric_grid_size(grid_quality) + K_BLOCK_SIZE - 1)
                                     / K_BLOCK_SIZE;
    importance_kernel<<<importance_grid_size, K_BLOCK_SIZE>>>(
        t_w, t_x, t_y, t_z, thrust::raw_pointer_cast(points_d.data()),
        thrust::raw_pointer_cast(faces_d.data()), thrust::raw_pointer_cast(remaining_faces_d.data()),
        grid_quality, number_of_remaining_faces, epsilon, thrust::raw_pointer_cast(importance_d.data()));
    CUDA_CHECK_KERNEL();

    std::vector<double> importance_host(number_of_remaining_faces);
    thrust::copy(importance_d.begin(), importance_d.begin() + number_of_remaining_faces,
                 importance_host.begin());

    std::vector<std::pair<int, double>> face_importance(number_of_remaining_faces);
    for (int j = 0; j < number_of_remaining_faces; ++j) {
      face_importance[j] = {remaining_faces_host[j], importance_host[j]};
    }
    std::sort(face_importance.begin(), face_importance.end(),
              [](auto const& a, auto const& b) { return a.second < b.second; });

    for (auto const& kv : face_importance) {
      if (std::isfinite(kv.second)) {
        Encoded_Faces[i].push_back(kv.first);
      }
    }

    if (is_deformable) {
      bool const is_boundary_inner = (i < number_of_surface_inner);
      bool const is_boundary_outer = (i >= number_of_surface_inner + number_of_inner)
                                     && (i < number_of_surface_inner + number_of_inner
                                         + number_of_surface_outer);
      std::array<int, 4> boundary_vertices = {-1, -1, -1, -1};
      if (is_boundary_inner) {
        boundary_vertices = Surface_Inner_Boundary_Features[i].vertices;
      } else if (is_boundary_outer) {
        boundary_vertices = Surface_Outer_Boundary_Features[i - number_of_surface_inner
                                                            - number_of_inner]
                                .vertices;
      }

      std::set<int> boundary_faces;
      for (int j = 0; j < 4; ++j) {
        if (boundary_vertices[j] != -1) {
          auto const& neighbors = Inner_MSH.Vertex_To_Face_Map[boundary_vertices[j]];
          boundary_faces.insert(neighbors.begin(), neighbors.end());
        }
      }

      std::vector<int> const current = Encoded_Faces[i];
      int already_encoded = 0;
      for (int const encoded : current) {
        if (boundary_faces.erase(encoded) > 0) {
          ++already_encoded;
        }
      }

      std::vector<int> reordered;
      reordered.reserve(current.size() + boundary_faces.size());
      reordered.insert(reordered.end(), current.begin(), current.begin() + already_encoded);
      reordered.insert(reordered.end(), boundary_faces.begin(), boundary_faces.end());
      reordered.insert(reordered.end(), current.begin() + already_encoded, current.end());

      Encoded_Faces[i] = std::move(reordered);
    }

    std::cout << "\r\033[K[DEBUG][TES::Encode_With_Optimization] Progress: "
              << (i + 1) * 100.0f / static_cast<float>(tetras_host.size()) << "%" << std::flush;
  }
  std::cout << "\n";

  cudaFree(atomic_count_d);
  CUDA_CHECK_LAST();

  auto const tok = std::chrono::high_resolution_clock::now();
  std::cout << "[DEBUG][TES::Encode_With_Optimization] Done in "
            << std::chrono::duration_cast<std::chrono::seconds>(tok - tik).count() << "s\n";
  return true;
}

bool TES::Export(std::string const& file_name) {
  std::ofstream file(file_name, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    std::cerr << "[ERROR][TES::Export] Cannot open file: " << file_name << " ("
              << std::strerror(errno) << ")\n";
    return false;
  }

  file << "# " << file_name << "\n";

  file << "# Number of surface nodes\n" << Surface_Nodes.size() << "\n";
  file << "# Number of internal nodes\n" << Internal_Nodes.size() << "\n";
  file << "# Number of outer nodes\n" << Outer_Nodes.size() << "\n";
  file << "# Number of faces\n" << Faces.size() << "\n";
  file << "# Number of surface inner tetras\n" << Surface_Inner_Tetras.size() << "\n";
  file << "# Number of inner tetras\n" << Inner_Tetras.size() << "\n";
  file << "# Number of surface outer tetras\n" << Surface_Outer_Tetras.size() << "\n";
  file << "# Number of outer tetras\n" << Outer_Tetras.size() << "\n";

  std::size_t total_encoded = 0;
  std::size_t max_encoded = 0;
  for (auto const& v : Encoded_Faces) {
    total_encoded += v.size();
    if (v.size() > max_encoded) {
      max_encoded = v.size();
    }
  }
  file << "# Number of encoded faces\n" << total_encoded << "\n";
  file << "# Maximum number of encoded faces\n" << max_encoded << "\n";

  file << "# Surface nodes\n";
  for (vector4d const& v : Surface_Nodes) {
    file << std::setprecision(17) << v.x << " " << v.y << " " << v.z << "\n";
  }
  file << "# Internal nodes\n";
  for (vector4d const& v : Internal_Nodes) {
    file << std::setprecision(17) << v.x << " " << v.y << " " << v.z << "\n";
  }
  file << "# Outer nodes\n";
  for (vector4d const& v : Outer_Nodes) {
    file << std::setprecision(17) << v.x << " " << v.y << " " << v.z << "\n";
  }

  file << "# Faces\n";
  for (iface_t const& f : Faces) {
    file << f.x << " " << f.y << " " << f.z << "\n";
  }

  auto write_tetras = [&file](char const* label, std::vector<itetra_t> const& tetras) {
    file << label << "\n";
    for (itetra_t const& t : tetras) {
      file << t.w << " " << t.x << " " << t.y << " " << t.z << "\n";
    }
  };
  write_tetras("# Surface inner tetras", Surface_Inner_Tetras);
  write_tetras("# Inner tetras", Inner_Tetras);
  write_tetras("# Surface outer tetras", Surface_Outer_Tetras);
  write_tetras("# Outer tetras", Outer_Tetras);

  file << "# Encoded faces\n";
  for (std::vector<int> const& encoded : Encoded_Faces) {
    for (int const idx : encoded) {
      file << idx << " ";
    }
    file << "\n";
  }

  std::cout << "[DEBUG][TES::Export] Success: " << file_name << "\n";
  return true;
}

void TES::Print_Statistics() {
  constexpr double k_bytes_per_mb = 1024.0 * 1024.0;
  constexpr int k_coords_per_node = 3;
  constexpr int k_indices_per_face = 3;
  constexpr int k_indices_per_tetra = 4;

  double const memory_inner_geometry_mb =
      ((Surface_Nodes.size() + Internal_Nodes.size()) * sizeof(float) * k_coords_per_node
       + Faces.size() * sizeof(int) * k_indices_per_face
       + (Surface_Inner_Tetras.size() + Inner_Tetras.size()) * sizeof(int) * k_indices_per_tetra)
      / k_bytes_per_mb;

  double const memory_outer_geometry_mb =
      (Outer_Nodes.size() * sizeof(float) * k_coords_per_node
       + (Surface_Outer_Tetras.size() + Outer_Tetras.size()) * sizeof(int) * k_indices_per_tetra)
      / k_bytes_per_mb;

  double const extra_memory_mb =
      (Internal_Nodes.size() * sizeof(float) * k_coords_per_node
       + Outer_Nodes.size() * sizeof(float) * k_coords_per_node
       + (Surface_Inner_Tetras.size() + Inner_Tetras.size() + Surface_Outer_Tetras.size()
          + Outer_Tetras.size())
             * sizeof(int) * k_indices_per_tetra)
      / k_bytes_per_mb;

  std::size_t encoded_surface_inner = 0;
  std::size_t encoded_inner = 0;
  std::size_t encoded_surface_outer = 0;
  std::size_t encoded_outer = 0;

  std::size_t const offset_inner = Surface_Inner_Tetras.size();
  std::size_t const offset_surface_outer = offset_inner + Inner_Tetras.size();
  std::size_t const offset_outer = offset_surface_outer + Surface_Outer_Tetras.size();
  std::size_t const total_tetras = offset_outer + Outer_Tetras.size();

  for (std::size_t i = 0; i < total_tetras && i < Encoded_Faces.size(); ++i) {
    std::size_t const count = Encoded_Faces[i].size();
    if (i < offset_inner) {
      encoded_surface_inner += count;
    } else if (i < offset_surface_outer) {
      encoded_inner += count;
    } else if (i < offset_outer) {
      encoded_surface_outer += count;
    } else {
      encoded_outer += count;
    }
  }

  double const mb_surface_inner = encoded_surface_inner * sizeof(int) / k_bytes_per_mb;
  double const mb_inner = encoded_inner * sizeof(int) / k_bytes_per_mb;
  double const mb_surface_outer = encoded_surface_outer * sizeof(int) / k_bytes_per_mb;
  double const mb_outer = encoded_outer * sizeof(int) / k_bytes_per_mb;

  std::cout << "======== TES::Print_Statistics ========\n";
  std::cout << "Inner geometry: " << memory_inner_geometry_mb << " MB\n";
  std::cout << "Outer geometry: " << memory_outer_geometry_mb << " MB\n";
  std::cout << "Encoded faces, surface inner tetras: " << mb_surface_inner << " MB\n";
  std::cout << "Encoded faces, inner tetras: " << mb_inner << " MB\n";
  std::cout << "Encoded faces, surface outer tetras: " << mb_surface_outer << " MB\n";
  std::cout << "Encoded faces, outer tetras: " << mb_outer << " MB\n";
  std::cout << "TED total: " << memory_inner_geometry_mb + mb_surface_inner + mb_inner << " MB\n";
  std::cout << "TES total: "
            << memory_inner_geometry_mb + memory_outer_geometry_mb + mb_surface_inner + mb_inner
                   + mb_surface_outer + mb_outer
            << " MB\n";
  std::cout << "TES pure cost: " << extra_memory_mb + mb_surface_inner + mb_inner + mb_surface_outer
                                        + mb_outer
            << " MB\n";
}

}  // namespace tes
