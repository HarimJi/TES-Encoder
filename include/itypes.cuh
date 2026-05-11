#pragma once

#include <cuda_runtime.h>

namespace tes {

struct iedge_t {
  int x;
  int y;
};

struct iface_t {
  int x;
  int y;
  int z;
};

struct itetra_t {
  int x;
  int y;
  int z;
  int w;
};

__host__ __device__ __forceinline__ iedge_t make_edge(int const a, int const b) {
  if (a < 0 || b < 0 || a == b) {
    return iedge_t{-1, -1};
  }
  return (a > b) ? iedge_t{b, a} : iedge_t{a, b};
}

__host__ __device__ __forceinline__ iface_t make_face(int const a, int const b, int const c) {
  if (a < 0 || b < 0 || c < 0 || a == b || b == c || a == c) {
    return iface_t{-1, -1, -1};
  }
  return iface_t{a, b, c};
}

__host__ __device__ __forceinline__ itetra_t make_tetra(int const a, int const b, int const c, int const d) {
  if (a < 0 || b < 0 || c < 0 || d < 0 || a == b || a == c || a == d || b == c || b == d || c == d) {
    return itetra_t{-1, -1, -1, -1};
  }
  return itetra_t{a, b, c, d};
}

__host__ __device__ __forceinline__ bool is_valid(iedge_t const e) {
  return !(e.x < 0 || e.y < 0 || e.x == e.y);
}

__host__ __device__ __forceinline__ bool is_valid(iface_t const f) {
  return !(f.x < 0 || f.y < 0 || f.z < 0 || f.x == f.y || f.y == f.z || f.x == f.z);
}

__host__ __device__ __forceinline__ bool is_valid(itetra_t const t) {
  return !(t.x < 0 || t.y < 0 || t.z < 0 || t.w < 0);
}

__host__ __device__ __forceinline__ bool has_edge(iface_t const f, iedge_t const e) {
  return (e.x == f.x || e.x == f.y || e.x == f.z) && (e.y == f.x || e.y == f.y || e.y == f.z);
}

__host__ __device__ __forceinline__ int subtract_edge(iface_t const f, iedge_t const e) {
  if (!has_edge(f, e)) {
    return -1;
  }
  if (f.x != e.x && f.x != e.y) {
    return f.x;
  }
  if (f.y != e.x && f.y != e.y) {
    return f.y;
  }
  return f.z;
}

struct iedge_less_t {
  bool operator()(iedge_t const& a, iedge_t const& b) const {
    if (a.x != b.x) {
      return a.x < b.x;
    }
    return a.y < b.y;
  }
};

}  // namespace tes
