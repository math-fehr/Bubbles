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
  HD Object &operator[](unsigned i) { return objects[i]; }
  HD const Object &operator[](unsigned i) const { return objects[i]; }
};

/**
 * Launch the main kernel
 */
void kernel_launcher(cudaArray_const_t array, Scene scene, Camera camera);
