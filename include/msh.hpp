#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "itypes.cuh"
#include "vector4d.cuh"

namespace tes {

class MSH {
 public:
  enum msh_file_type_e { from_gmsh };
  enum edge_type_e { undefined, convex, concave, planar };
  enum tetra_type_e { boundary_face = 0, boundary_edge = 1, boundary_vertex = 2, internal = 3 };

  std::vector<vector4d> Nodes;
  int Number_Of_Surface_Nodes = 0;
  int Number_Of_Internal_Nodes = 0;

  std::vector<iface_t> Faces;
  std::vector<itetra_t> Tetrahedras;

  std::vector<std::set<int>> Vertex_To_Face_Map;
  std::map<iedge_t, std::pair<int, int>, iedge_less_t> Edge_To_Face_Map;
  std::map<iedge_t, edge_type_e, iedge_less_t> Edge_To_Type_Map;
  std::map<std::set<int>, int> Face_Indices_To_Face_Map;

  MSH() = default;

  bool Load(std::string const& file_name, msh_file_type_e type = from_gmsh);
  bool Construct_Maps();
  bool Inspect();
  bool Export_OBJ(std::string const& file_name);
  int Find_Surface_Face_Index(iface_t const f);
};

}  // namespace tes
