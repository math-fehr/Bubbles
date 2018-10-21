#pragma once

#include <cuda.h>
#include "object.h"
#include "camera.h"

class Sphere;

/**
 * Launch the main kernel
 */
void kernel_launcher(cudaArray_const_t array, Object *objects,
                     unsigned n_objects, Camera camera);
