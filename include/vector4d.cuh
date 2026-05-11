// vector4d ignores w element, which is used for padding to ensure 16-byte
// alignment for efficient memory access on the GPU. The w element can be used
// for other purposes if needed, but it is not utilized in the current
// implementation.

#pragma once

namespace tes {
__align__(16) struct vector4d {
  double x;
  double y;
  double z;
  double w;
  __forceinline__ constexpr __host__ __device__ vector4d(): x(0), y(0), z(0), w(0) {}
  __forceinline__ constexpr __host__ __device__ vector4d(double x, double y, double z): x(x), y(y), z(z), w(0) {};
};

__forceinline__ __host__ __device__ vector4d operator+(const vector4d a, const vector4d b) {
  return vector4d(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __host__ __device__ vector4d operator-(const vector4d a, const vector4d b) {
  return vector4d(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __host__ __device__ vector4d operator-(const vector4d v) {
  return vector4d(-v.x, -v.y, -v.z);
}

__forceinline__ __host__ __device__ vector4d operator*(const vector4d a, const double b) {
  return vector4d(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __host__ __device__ vector4d operator*(const double a, const vector4d b) {
  return vector4d(a * b.x, a * b.y, a * b.z);
}

__forceinline__ __host__ __device__ double operator*(const vector4d a, const vector4d b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __host__ __device__ vector4d operator/(const vector4d a, const double b) {
  const double inv_b = 1.0 / b;
  return vector4d(a.x * inv_b, a.y * inv_b, a.z * inv_b);
}  // unsafe division, caller must ensure b != 0

__forceinline__ __host__ __device__ vector4d cross(const vector4d a, const vector4d b) {
  return vector4d(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__forceinline__ __host__ __device__ double norm_squared(const vector4d v) {
  return v.x * v.x + v.y * v.y + v.z * v.z;
}

__forceinline__ __host__ __device__ double norm(const vector4d v) {
  return sqrt(norm_squared(v));
}

__forceinline__ __host__ __device__ vector4d normalize_or_zero(const vector4d v) {
  const double len = norm(v);
  if (len <= 1e-15) {
    return vector4d(0.0, 0.0, 0.0);
  }
  return v / len;
}

}  // namespace tes