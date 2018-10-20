#pragma once

#include <cuda.h>

class Sphere;

/**
 * Launch the main kernel
 */
void kernel_launcher(cudaArray_const_t array, const int width, const int height, Sphere* spheres);
