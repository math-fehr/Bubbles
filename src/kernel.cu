#include "assert_cuda.hpp"
#include "kernel.cuh"

#include "geom.h"
#include "object.h"
#include "camera.h"
#include "light.h"

#define THREADS_PER_BLOCK 256

// The surface where CUDA will write
surface<void, cudaSurfaceType2D> surf;

// The value of a pixel
struct RGBA {
  unsigned r : 8;
  unsigned g : 8;
  unsigned b : 8;
  unsigned a : 8;
};

__device__ real intersect(Object object, Rayf ray) {
  if(object.type == ObjectType::sphere) {
    return object.sphere.inter(ray);
  } else if(object.type == ObjectType::plane) {
    return object.plane.inter(ray);
  } else {
    return -1.f;
  }
}

__device__ Vec3f normal(Object object, Rayf ray, real intersection_distance) {
  if(object.type == ObjectType::sphere) {
    return object.sphere.normal(ray(intersection_distance));
  } else if(object.type == ObjectType::plane) {
    return object.plane.normal(ray);
  } else {
    return Vec3f{};
  }
}


struct Intersection {
  int front_object;
  real intersection_distance;
  Vec3f intersection_point;
  Vec3f normal_point;
};

__device__ Intersection intersect_all(Object *objects,
                                              unsigned n_objects, Rayf ray) {
  int front_object = -1;
  real intersection_point = 1.f/0.f;

  for(int i = 0; i < n_objects; ++i) {
    float intersection_i = intersect(objects[i], ray);
    if(intersection_i > 0.f && intersection_i < intersection_point) {
      intersection_point = intersection_i;
      front_object = i;
    }
  }

  return Intersection{front_object, intersection_point, ray(intersection_point), normal(objects[front_object], ray, intersection_point)};
}

__device__ Color compute_diffuse_color(Object* objects, unsigned n_objects, PointLight light, Intersection intersection) {
  Rayf light_ray = light.ray_to_point(intersection.intersection_point);
  Intersection light_intersection = intersect_all(objects, n_objects, light_ray);
  bool light_touch = intersection.front_object == light_intersection.front_object;
  light_touch &= (intersection.intersection_point - light_intersection.intersection_point).norm() < 1e-3;

  if(!light_touch) {
    return Color{0.0f,0.0f,0.0f};
  }
  return objects[intersection.front_object].color * (intersection.normal_point | light_ray.dir) * light.color;
}

/**
 * Entry CUDA kernel. This is the code for one pixel
 */
__global__ void kernel(int counter, Object* objects, unsigned n_objects, Camera camera) {
  PointLight light{Vec3f{-30.0f,0.0f,0.0f}, Color{1,1,1}};

  // pixel coordinates
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int x_pixel = idx % camera.screen_width;
  int y_pixel = idx / camera.screen_width;

  Rayf ray = camera.get_ray(x_pixel, y_pixel);

  RGBA rgbx;
  rgbx.r = 0, rgbx.g=0,rgbx.b=0;

  Intersection intersection = intersect_all(objects, n_objects, ray);
  int front_object = intersection.front_object;
  Vec3f intersection_point = intersection.intersection_point;
  Rayf light_ray = light.ray_to_point(intersection_point);

  Vec3f normal_vec = normal(objects[front_object], ray, intersection.intersection_distance);

  Color ambiant_light{1.0f,1.0f,1.0f};
  Color ambiant_color = objects[front_object].color * ambiant_light;

  Color diffuse_color = compute_diffuse_color(objects, n_objects, light, intersection);

  Color final_color = ambiant_color * 0.1f + diffuse_color * 0.9f;
  rgbx.r = final_color.r * 255;
  rgbx.g = final_color.g * 255;
  rgbx.b = final_color.b * 255;

  if(idx < camera.screen_height * camera.screen_width) {
    surf2Dwrite(rgbx, surf, x_pixel * sizeof(rgbx), y_pixel, cudaBoundaryModeZero);
  }
}

void kernel_launcher(cudaArray_const_t array, Object* objects, unsigned n_objects, Camera camera) {
  // Count the number of frames displayed
  static unsigned counter = 0;
  counter += 1;

  cuda(BindSurfaceToArray(surf, array));

  const int blocks =
      (camera.screen_width * camera.screen_height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (blocks > 0) {
    kernel<<<blocks, THREADS_PER_BLOCK>>>(counter, objects, n_objects, camera);
  }
}
