#pragma once

#include <cuda_runtime.h>

namespace tes {

__host__ __device__ __forceinline__ int barycentric_grid_size(int const q) {
  return q * (q + 1) * (q + 2) / 6;
}

__host__ __device__ __forceinline__ int3 barycentric_index(int const i, int const q) {
  if (i < 0 || i >= barycentric_grid_size(q)) {
    return make_int3(0, 0, 0);
  }

  int idx = i;

  int z = 0;
  for (; z < q; ++z) {
    int const layer_size = (q - z) * (q - z + 1) / 2;
    if (idx < layer_size) {
      break;
    }
    idx -= layer_size;
  }

  int y = 0;
  for (; y < q - z; ++y) {
    int const row_size = q - z - y;
    if (idx < row_size) {
      break;
    }
    idx -= row_size;
  }

  int const x = idx;

  return make_int3(x, y, z);
}

}  // namespace tes
