#pragma once

#include <cuda.h>
#include "object.h"

class Sphere;

/**
 * Launch the main kernel
 */
void kernel_launcher(cudaArray_const_t array, const int width, const int height,
                     Object *objects, unsigned n_objects);
