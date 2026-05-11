#include "msh.hpp"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "geometry_utils.cuh"
#include "itypes.cuh"

namespace tes {

namespace {

bool parse_block_header(std::istringstream& iss, int& dimension, int& tag, int& parameter, int& count) {
  return static_cast<bool>(iss >> dimension >> tag >> parameter >> count);
}

}  // namespace

bool MSH::Load(std::string const& file_name, msh_file_type_e type) {
  std::ifstream ifs(file_name);
  if (!ifs.is_open()) {
    std::cerr << "[ERROR][MSH::Load] Failed to open file: " << file_name << "\n";
    return false;
  }

  if (type != from_gmsh) {
    std::cerr << "[ERROR][MSH::Load] Unsupported file type\n";
    return false;
  }

  int node_start_index = 1;
  std::string line;

  while (std::getline(ifs, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    if (line == "$MeshFormat") {
      std::getline(ifs, line);
      std::istringstream iss(line);
      double version = 0;
      int file_type = 0;
      int data_size = 0;
      iss >> version >> file_type >> data_size;
      if (file_type != 0) {
        std::cerr << "[ERROR][MSH::Load] Binary .msh not supported (file_type=" << file_type << ")\n";
        return false;
      }
      std::getline(ifs, line);  // $EndMeshFormat
      continue;
    }

    if (line == "$Nodes") {
      std::getline(ifs, line);
      std::istringstream iss(line);
      int number_of_blocks = 0;
      int total_nodes = 0;
      int min_index = 0;
      int max_index = 0;
      iss >> number_of_blocks >> total_nodes >> min_index >> max_index;

      if (number_of_blocks != 1 && number_of_blocks != 2) {
        std::cerr << "[ERROR][MSH::Load] Invalid number_of_blocks for nodes: " << number_of_blocks << "\n";
        return false;
      }
      if (total_nodes <= 0 || total_nodes != (max_index - min_index + 1)) {
        std::cerr << "[ERROR][MSH::Load] Invalid node range: [" << min_index << ", " << max_index << "]\n";
        return false;
      }

      node_start_index = min_index;
      Nodes.resize(total_nodes);

      // Surface nodes block
      std::getline(ifs, line);
      iss = std::istringstream(line);
      int dimension = 0;
      int tag = 0;
      int parameter = 0;
      if (!parse_block_header(iss, dimension, tag, parameter, Number_Of_Surface_Nodes)) {
        std::cerr << "[ERROR][MSH::Load] Failed to parse surface block header\n";
        return false;
      }
      if (dimension != 2) {
        std::cerr << "[ERROR][MSH::Load] Expected surface block dimension 2, got " << dimension << "\n";
        return false;
      }
      if (Number_Of_Surface_Nodes < 0 || Number_Of_Surface_Nodes > total_nodes) {
        std::cerr << "[ERROR][MSH::Load] Invalid number of surface nodes: " << Number_Of_Surface_Nodes << "\n";
        return false;
      }

      for (int i = 0; i < Number_Of_Surface_Nodes; ++i) {
        std::getline(ifs, line);
        int index = std::stoi(line);
        if (index != min_index + i) {
          std::cerr << "[ERROR][MSH::Load] Surface node index mismatch: expected " << (min_index + i)
                    << ", got " << index << "\n";
          return false;
        }
      }
      for (int i = 0; i < Number_Of_Surface_Nodes; ++i) {
        std::getline(ifs, line);
        std::istringstream cs(line);
        double x = 0;
        double y = 0;
        double z = 0;
        if (!(cs >> x >> y >> z)) {
          std::cerr << "[ERROR][MSH::Load] Failed to parse surface coordinates\n";
          return false;
        }
        Nodes[i] = vector4d(x, y, z);
      }

      if (number_of_blocks == 1) {
        std::getline(ifs, line);
        if (line != "$EndNodes") {
          std::cerr << "[ERROR][MSH::Load] Missing $EndNodes\n";
          return false;
        }
        continue;
      }

      // Internal nodes block
      std::getline(ifs, line);
      iss = std::istringstream(line);
      if (!parse_block_header(iss, dimension, tag, parameter, Number_Of_Internal_Nodes)) {
        std::cerr << "[ERROR][MSH::Load] Failed to parse internal block header\n";
        return false;
      }
      if (dimension != 3) {
        std::cerr << "[ERROR][MSH::Load] Expected internal block dimension 3, got " << dimension << "\n";
        return false;
      }
      if (Number_Of_Internal_Nodes + Number_Of_Surface_Nodes != total_nodes) {
        std::cerr << "[ERROR][MSH::Load] Invalid number of internal nodes: " << Number_Of_Internal_Nodes
                  << "\n";
        return false;
      }

      for (int i = 0; i < Number_Of_Internal_Nodes; ++i) {
        std::getline(ifs, line);
        int index = std::stoi(line);
        int const expected = min_index + Number_Of_Surface_Nodes + i;
        if (index != expected) {
          std::cerr << "[ERROR][MSH::Load] Internal node index mismatch: expected " << expected << ", got "
                    << index << "\n";
          return false;
        }
      }
      for (int i = 0; i < Number_Of_Internal_Nodes; ++i) {
        std::getline(ifs, line);
        std::istringstream cs(line);
        double x = 0;
        double y = 0;
        double z = 0;
        if (!(cs >> x >> y >> z)) {
          std::cerr << "[ERROR][MSH::Load] Failed to parse internal coordinates\n";
          return false;
        }
        Nodes[Number_Of_Surface_Nodes + i] = vector4d(x, y, z);
      }

      std::getline(ifs, line);
      if (line != "$EndNodes") {
        std::cerr << "[ERROR][MSH::Load] Missing $EndNodes\n";
        return false;
      }
      continue;
    }

    if (line == "$Elements") {
      std::getline(ifs, line);
      std::istringstream iss(line);
      int number_of_blocks = 0;
      int total_elements = 0;
      int min_index = 0;
      int max_index = 0;
      iss >> number_of_blocks >> total_elements >> min_index >> max_index;

      if (number_of_blocks != 2) {
        std::cerr << "[ERROR][MSH::Load] Invalid number_of_blocks for elements: " << number_of_blocks
                  << "\n";
        return false;
      }
      if (total_elements <= 0 || total_elements != (max_index - min_index + 1)) {
        std::cerr << "[ERROR][MSH::Load] Invalid element range\n";
        return false;
      }

      // Faces block
      std::getline(ifs, line);
      iss = std::istringstream(line);
      int dimension = 0;
      int tag = 0;
      int parameter = 0;
      int number_of_faces = 0;
      if (!parse_block_header(iss, dimension, tag, parameter, number_of_faces)) {
        std::cerr << "[ERROR][MSH::Load] Failed to parse faces block header\n";
        return false;
      }
      if (dimension != 2) {
        std::cerr << "[ERROR][MSH::Load] Expected face dimension 2, got " << dimension << "\n";
        return false;
      }
      if (number_of_faces < 0 || number_of_faces > total_elements) {
        std::cerr << "[ERROR][MSH::Load] Invalid number of faces: " << number_of_faces << "\n";
        return false;
      }

      Faces.resize(number_of_faces);
      for (int i = 0; i < number_of_faces; ++i) {
        std::getline(ifs, line);
        std::istringstream fs(line);
        int element_index = 0;
        int a = 0;
        int b = 0;
        int c = 0;
        if (!(fs >> element_index >> a >> b >> c)) {
          std::cerr << "[ERROR][MSH::Load] Failed to parse face\n";
          return false;
        }
        if (element_index != min_index + i) {
          std::cerr << "[ERROR][MSH::Load] Face index mismatch\n";
          return false;
        }
        Faces[i] = make_face(a - node_start_index, b - node_start_index, c - node_start_index);
      }

      // Tetrahedra block
      std::getline(ifs, line);
      iss = std::istringstream(line);
      int number_of_tetras = 0;
      if (!parse_block_header(iss, dimension, tag, parameter, number_of_tetras)) {
        std::cerr << "[ERROR][MSH::Load] Failed to parse tetras block header\n";
        return false;
      }
      if (dimension != 3) {
        std::cerr << "[ERROR][MSH::Load] Expected tetra dimension 3, got " << dimension << "\n";
        return false;
      }
      if (number_of_tetras + number_of_faces != total_elements) {
        std::cerr << "[ERROR][MSH::Load] Invalid number of tetras: " << number_of_tetras << "\n";
        return false;
      }

      Tetrahedras.resize(number_of_tetras);
      for (int i = 0; i < number_of_tetras; ++i) {
        std::getline(ifs, line);
        std::istringstream ts(line);
        int element_index = 0;
        int a = 0;
        int b = 0;
        int c = 0;
        int d = 0;
        if (!(ts >> element_index >> a >> b >> c >> d)) {
          std::cerr << "[ERROR][MSH::Load] Failed to parse tetra\n";
          return false;
        }
        if (element_index != min_index + number_of_faces + i) {
          std::cerr << "[ERROR][MSH::Load] Tetra index mismatch\n";
          return false;
        }
        Tetrahedras[i] = make_tetra(a - node_start_index, b - node_start_index, c - node_start_index,
                                    d - node_start_index);
      }

      std::getline(ifs, line);
      if (line != "$EndElements") {
        std::cerr << "[ERROR][MSH::Load] Missing $EndElements\n";
        return false;
      }
    }
  }

  std::cout << "[DEBUG][MSH::Load] Success: " << Nodes.size() << " nodes, " << Faces.size() << " faces, "
            << Tetrahedras.size() << " tetras\n";
  return true;
}

bool MSH::Construct_Maps() {
  Vertex_To_Face_Map.assign(Number_Of_Surface_Nodes, {});

  for (int face_index = 0; face_index < static_cast<int>(Faces.size()); ++face_index) {
    iface_t const f = Faces[face_index];
    if (f.x < 0 || f.y < 0 || f.z < 0 || f.x >= Number_Of_Surface_Nodes
        || f.y >= Number_Of_Surface_Nodes || f.z >= Number_Of_Surface_Nodes) {
      std::cerr << "[ERROR][MSH::Construct_Maps] Invalid node index in face #" << face_index << "\n";
      return false;
    }
    Vertex_To_Face_Map[f.x].insert(face_index);
    Vertex_To_Face_Map[f.y].insert(face_index);
    Vertex_To_Face_Map[f.z].insert(face_index);
  }

  for (int node_index = 0; node_index < Number_Of_Surface_Nodes; ++node_index) {
    if (Vertex_To_Face_Map[node_index].size() < 3) {
      std::cerr << "[ERROR][MSH::Construct_Maps] Non-manifold surface node " << node_index << " ("
                << Vertex_To_Face_Map[node_index].size() << " connected faces)\n";
      return false;
    }
  }

  for (int face_index = 0; face_index < static_cast<int>(Faces.size()); ++face_index) {
    iface_t const f = Faces[face_index];
    iedge_t const edges[3] = {make_edge(f.x, f.y), make_edge(f.y, f.z), make_edge(f.z, f.x)};

    for (iedge_t const e : edges) {
      if (!is_valid(e)) {
        std::cerr << "[ERROR][MSH::Construct_Maps] Invalid edge in face #" << face_index << "\n";
        return false;
      }
      auto it = Edge_To_Face_Map.find(e);
      if (it == Edge_To_Face_Map.end()) {
        Edge_To_Face_Map.insert({e, {face_index, -1}});
      } else {
        it->second.second = face_index;
      }
    }
  }

  for (auto const& kv : Edge_To_Face_Map) {
    if (kv.second.second == -1) {
      std::cerr << "[ERROR][MSH::Construct_Maps] Boundary edge (" << kv.first.x << ", " << kv.first.y
                << ") on face " << kv.second.first << "\n";
      return false;
    }
  }

  // Reorder face pair so that f1 is on the side where (e1->e2->p1) is CCW relative to f1's outward normal
  for (auto& kv : Edge_To_Face_Map) {
    iedge_t const e = kv.first;
    iface_t const f1 = Faces[kv.second.first];
    vector4d const p1 = Nodes[subtract_edge(f1, e)];
    vector4d const e1 = Nodes[e.x];
    vector4d const e2 = Nodes[e.y];
    vector4d const a = Nodes[f1.x];
    vector4d const b = Nodes[f1.y];
    vector4d const c = Nodes[f1.z];
    if (dot(cross(b - a, c - a), cross(e2 - e1, p1 - e1)) < 0) {
      std::swap(kv.second.first, kv.second.second);
    }
  }

  for (int face_index = 0; face_index < static_cast<int>(Faces.size()); ++face_index) {
    iface_t const f = Faces[face_index];
    Face_Indices_To_Face_Map.emplace(std::set<int>{f.x, f.y, f.z}, face_index);
  }

  for (auto const& kv : Edge_To_Face_Map) {
    iedge_t const e = kv.first;
    iface_t const f1 = Faces[kv.second.first];
    iface_t const f2 = Faces[kv.second.second];
    vector4d const p1 = Nodes[subtract_edge(f1, e)];
    vector4d const p2 = Nodes[subtract_edge(f2, e)];
    vector4d const e1 = Nodes[e.x];
    vector4d const e2 = Nodes[e.y];
    vector4d const n1 = normalize_or_zero(cross(e2 - e1, p1 - e1));
    vector4d const n2 = normalize_or_zero(cross(p2 - e1, e2 - e1));
    vector4d const pi1 = normalize_or_zero(cross(n1, e2 - e1));
    vector4d const pi2 = normalize_or_zero(cross(n2, e1 - e2));

    double const pred1 = dot(pi1, n2);
    double const pred2 = dot(pi2, n1);

    constexpr double planar_threshold = 1e-9;
    edge_type_e type = undefined;
    if (pred1 < 0 && pred2 < 0) {
      type = convex;
    } else if (pred1 > 0 && pred2 > 0) {
      type = concave;
    } else if (std::abs(pred1) < planar_threshold || std::abs(pred2) < planar_threshold) {
      type = planar;
    }
    Edge_To_Type_Map.insert({e, type});
  }

  std::cout << "[DEBUG][MSH::Construct_Maps] Success\n";
  return true;
}

bool MSH::Inspect() {
  if (Faces.size() < 4) {
    std::cout << "[DEBUG][MSH::Inspect] Not a valid mesh (faces=" << Faces.size() << ")\n";
    return false;
  }

  double minimum_area = 1e10;
  for (iface_t const f : Faces) {
    vector4d const v1 = Nodes[f.x];
    vector4d const v2 = Nodes[f.y];
    vector4d const v3 = Nodes[f.z];
    double const area = norm(cross(v2 - v1, v3 - v1));
    if (area < minimum_area) {
      minimum_area = area;
    }
  }

  std::cout << "[DEBUG][MSH::Inspect] Minimum triangle area: " << minimum_area << "\n";
  return true;
}

int MSH::Find_Surface_Face_Index(iface_t const f) {
  auto const it = Face_Indices_To_Face_Map.find({f.x, f.y, f.z});
  if (it == Face_Indices_To_Face_Map.end()) {
    return -1;
  }
  return it->second;
}

bool MSH::Export_OBJ(std::string const& file_name) {
  std::string const full_file_name = file_name + ".obj";
  std::ofstream file(full_file_name, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    std::cerr << "[ERROR][MSH::Export_OBJ] Cannot open file: " << full_file_name << " ("
              << std::strerror(errno) << ")\n";
    return false;
  }

  for (int i = 0; i < Number_Of_Surface_Nodes; ++i) {
    file << "v " << Nodes[i].x << " " << Nodes[i].y << " " << Nodes[i].z << "\n";
  }
  for (iface_t const& f : Faces) {
    file << "f " << (f.x + 1) << " " << (f.y + 1) << " " << (f.z + 1) << "\n";
  }

  std::cout << "[DEBUG][MSH::Export_OBJ] Success: " << full_file_name << "\n";
  return true;
}

}  // namespace tes
