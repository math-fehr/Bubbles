#include "assert_cuda.hpp"
#include "kernel.cuh"

#include "geom.h"

#define THREADS_PER_BLOCK 128
#define FOV (51.52f * M_PI / 180.0f)

// The surface where CUDA will write
surface<void, cudaSurfaceType2D> surf;

// The value of a pixel
struct RGBA {
  unsigned r : 8;
  unsigned g : 8;
  unsigned b : 8;
  unsigned a : 8;
};


__device__ Rayf get_ray(real x_pixel, real y_pixel, real width, real height) {
  real scale = tanf(FOV * 0.5f);
  real image_aspect_ratio = width / height;

  real x = (2.0f * ((x_pixel + 0.5f) / width) - 1) * scale;
  real y = (1 - 2.0f * ((y_pixel + 0.5f) / height)) * scale / image_aspect_ratio;

  Vec3f camera_pos{4.208271f, 8.374532f, 17.932925f};
  Vec3f camera_to_world_x{0.945519, -0.179534, 0.271593};
  Vec3f camera_to_world_y{0, 0.834209, 0.551447};
  Vec3f camera_to_world_z{-0.325569, -0.521403, 0.78876};
  Mat3f camera_to_world{camera_to_world_x,camera_to_world_y,camera_to_world_z};

  Vec3f direction_camera{x,y,-1};
  Vec3f direction = camera_to_world * direction_camera;
  Vec3f position = camera_pos;
  return Rayf(position,direction);
}


/**
 * Entry CUDA kernel. This is the code for one pixel
 */
__global__ void kernel(int width, int height, int counter) {
  // pixel coordinates
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int x_pixel = idx % width;
  int y_pixel = idx / width;

  Rayf ray = get_ray(x_pixel,y_pixel, width, height);

  Vec3f pos{0.f,0.f,0.f};
  float radius = 0.75f;
  Color color{1.f,1.f,1.f};
  Sphere sphere(pos,radius,color);

  RGBA rgbx;
  if(sphere.inter(ray) < 0) {
    rgbx.r =0,rgbx.g=0,rgbx.b=0;
  } else {
    rgbx.r=255,rgbx.g=255,rgbx.b=255;
  }

  surf2Dwrite(rgbx, surf, x_pixel * sizeof(rgbx), y_pixel, cudaBoundaryModeZero);
}

void kernel_launcher(cudaArray_const_t array, const int width,
                     const int height) {
  // Count the number of frames displayed
  static unsigned counter = 0;
  counter += 1;

  cuda(BindSurfaceToArray(surf, array));

  const int blocks =
      (width * height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (blocks > 0) {
    kernel<<<blocks, THREADS_PER_BLOCK>>>(width, height, counter);
  }
}
