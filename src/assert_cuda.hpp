#pragma once

#include <cuda_runtime.h>
#include <iostream>

/**
 * Launch a cuda runtime function, and abort if it raised an error.
 */
#define cuda(...)  cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__, true);

/**
 * Check if code is cudaSuccess, otherwise print an error on stderr, and optionnaly abort.
 */
inline cudaError_t cuda_assert(cudaError_t code, std::string file, int line, bool abort)
{
  if (code != cudaSuccess) {
    std::cerr << "cuda_assert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;

    if(abort) {
      cudaDeviceReset();
      exit(code);
    }
  }
  return code;
}

