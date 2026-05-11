#pragma once

#include <array>
#include <string>
#include <tuple>
#include <vector>

#include "itypes.cuh"
#include "msh.hpp"
#include "vector4d.cuh"

namespace tes {

struct boundary_features_t {
  std::array<int, 4> vertices;
  std::array<iedge_t, 6> edges;
  std::array<int, 4> faces;
};

class TES {
 public:
  MSH Inner_MSH;
  MSH Outer_MSH;

  std::vector<vector4d> Surface_Nodes;
  std::vector<vector4d> Internal_Nodes;
  std::vector<vector4d> Outer_Nodes;
  // Node ordering: surface -> internal -> outer

  std::vector<iface_t> Faces;

  std::vector<itetra_t> Surface_Inner_Tetras;
  std::vector<itetra_t> Inner_Tetras;
  std::vector<itetra_t> Surface_Outer_Tetras;
  std::vector<itetra_t> Outer_Tetras;
  // Tetra ordering: surface inner -> inner -> surface outer -> outer

  std::vector<boundary_features_t> Surface_Inner_Boundary_Features;
  std::vector<boundary_features_t> Surface_Outer_Boundary_Features;

  std::vector<std::vector<int>> Encoded_Faces;

  TES() = default;

  bool Load_Inner_MSH(std::string const& file_name, MSH::msh_file_type_e type = MSH::from_gmsh);
  bool Load_Outer_MSH(std::string const& file_name, MSH::msh_file_type_e type = MSH::from_gmsh);
  bool Make_Geometry();
  bool Encode_With_Optimization(int const grid_quality, double const alpha = 0.999,
                                double const epsilon = 0.0, bool const is_deformable = false);
  bool Export(std::string const& file_name);
  void Print_Statistics();
};

}  // namespace tes
