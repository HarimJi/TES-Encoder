#pragma once

#include "vector4d.cuh"

namespace tes {

__host__ __device__ __forceinline__ double dot(vector4d const a, vector4d const b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__ bool is_near(vector4d const a, vector4d const b,
                                                 double const threshold = 1e-6) {
  return norm_squared(a - b) < threshold * threshold;
}

__host__ __device__ __forceinline__ vector4d project_point_to_triangle(vector4d const p, vector4d const a,
                                                                      vector4d const b, vector4d const c) {
  vector4d const ab = b - a;
  vector4d const ac = c - a;
  vector4d const ap = p - a;

  double const d1 = dot(ab, ap);
  double const d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) {
    return a;
  }

  vector4d const bp = p - b;
  double const d3 = dot(ab, bp);
  double const d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) {
    return b;
  }

  double const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    double const v = d1 / (d1 - d3);
    return a + v * ab;
  }

  vector4d const cp = p - c;
  double const d5 = dot(ab, cp);
  double const d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) {
    return c;
  }

  double const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    double const w = d2 / (d2 - d6);
    return a + w * ac;
  }

  double const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 >= d3 && d5 >= d6) {
    double const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return b + w * (c - b);
  }

  double const denom = 1.0 / (va + vb + vc);
  double const v = vb * denom;
  double const w = vc * denom;
  return a + v * ab + w * ac;
}

}  // namespace tes
