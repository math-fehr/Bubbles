#pragma once

#include "camera.h"
#include "light.h"
#include "object.h"
#include <cuda.h>

struct Scene {
  Object *objects; // GPU pointer
  unsigned n_objects;
  PointLight light;
  AmbiantLight ambiant_light;
};

/**
 * Launch the main kernel
 */
void kernel_launcher(cudaArray_const_t array, Scene scene, Camera camera);
