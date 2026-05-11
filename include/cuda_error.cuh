#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK_LAST()                                                                          \
  do {                                                                                             \
    cudaError_t const _err = cudaGetLastError();                                                   \
    if (_err != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

#define CUDA_CHECK_KERNEL()                                                                            \
  do {                                                                                                 \
    cudaError_t _launch_err = cudaGetLastError();                                                      \
    if (_launch_err != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "[CUDA LAUNCH ERROR] %s:%d: %s\n", __FILE__, __LINE__,                      \
                   cudaGetErrorString(_launch_err));                                                   \
      std::exit(EXIT_FAILURE);                                                                         \
    }                                                                                                  \
    cudaError_t _sync_err = cudaDeviceSynchronize();                                                   \
    if (_sync_err != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "[CUDA RUNTIME ERROR] %s:%d: %s\n", __FILE__, __LINE__,                     \
                   cudaGetErrorString(_sync_err));                                                     \
      std::exit(EXIT_FAILURE);                                                                         \
    }                                                                                                  \
  } while (0)
