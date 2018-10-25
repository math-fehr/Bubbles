#include "assert_cuda.hpp"
#include "kernel.cuh"

#include "camera.h"
#include "geom.h"
#include "light.h"
#include "object.h"

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
  if (object.type == ObjectType::sphere) {
    return object.sphere.inter(ray);
  } else if (object.type == ObjectType::plane) {
    return object.plane.inter(ray);
  } else if (object.type == ObjectType::box) {
    return object.box.inter(ray);
  } else {
    return -1.f;
  }
}

__device__ Vec3f normal(Object object, Rayf ray, real intersection_distance) {
  if (object.type == ObjectType::sphere) {
    return object.sphere.normal(ray(intersection_distance));
  } else if (object.type == ObjectType::plane) {
    return object.plane.normal(ray);
  } else if (object.type == ObjectType::box) {
    return object.box.normal(ray, ray(intersection_distance));
  } else {
    return Vec3f{};
  }
}

struct Intersection {
  int object_id;
  Object object;
  real distance;
  Vec3f point;
  Vec3f normal;
};

__device__ Intersection intersect_all(Object *objects, unsigned n_objects,
                                      Rayf ray) {
  int front_object = -1;
  real intersection_point = 1.f / 0.f;

  for (int i = 0; i < n_objects; ++i) {
    float intersection_i = intersect(objects[i], ray);
    if (intersection_i > 0.f && intersection_i < intersection_point) {
      intersection_point = intersection_i;
      front_object = i;
    }
  }

  return Intersection{front_object, objects[front_object], intersection_point,
                      ray(intersection_point),
                      normal(objects[front_object], ray, intersection_point)};
}

__device__ Color compute_phong_color(Object *objects, unsigned n_objects,
                                     PointLight light,
                                     AmbiantLight ambiant_light,
                                     Intersection intersection) {

  Color ambiant_color = intersection.object.texture.phong.color *
                        ambiant_light.color *
                        intersection.object.texture.phong.ambiant_factor;

  Rayf light_ray = light.ray_to_point(intersection.point);
  Intersection light_intersection =
      intersect_all(objects, n_objects, light_ray);
  bool light_touch = intersection.object_id == light_intersection.object_id;
  light_touch &= (intersection.point - light_intersection.point).norm() < 1e-3;

  if (!light_touch) {
    return ambiant_color;
  }
  real diffusion_factor = intersection.normal | light_ray.dir;
  diffusion_factor = max(0.0f, min(1.0f, diffusion_factor));
  diffusion_factor *= intersection.object.texture.phong.diffusion_factor;
  Color diffuse_color =
      intersection.object.texture.phong.color * diffusion_factor * light.color;
  return diffuse_color + ambiant_color;
}

__device__ Color compute_texture(Object *objects, unsigned n_objects,
                                 PointLight light, AmbiantLight ambiant_light,
                                 Intersection intersection) {
  switch (intersection.object.texture.type) {
  case TextureType::phong:
    return compute_phong_color(objects, n_objects, light, ambiant_light,
                               intersection);
  }
  return Color{1, 1, 1};
}

/**
 * Entry CUDA kernel. This is the code for one pixel
 */
__global__ void kernel(int counter, Object *objects, unsigned n_objects,
                       Camera camera) {
  PointLight light{Vec3f{-30.0f, 0.0f, 0.0f}, Color{1, 1, 1}};
  AmbiantLight ambiant_light{Color{1.0f, 1.0f, 1.0f}};

  // pixel coordinates
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int x_pixel = idx % camera.screen_width;
  int y_pixel = idx / camera.screen_width;

  Rayf ray = camera.get_ray(x_pixel, y_pixel);

  RGBA rgbx;
  rgbx.r = 0, rgbx.g = 0, rgbx.b = 0;

  Intersection intersection = intersect_all(objects, n_objects, ray);
  Rayf light_ray = light.ray_to_point(intersection.point);

  Vec3f normal_vec = normal(intersection.object, ray, intersection.distance);

  Color color = compute_phong_color(objects, n_objects, light, ambiant_light,
                                    intersection);
  rgbx.r = color.r * 255;
  rgbx.g = color.g * 255;
  rgbx.b = color.b * 255;

  if (idx < camera.screen_height * camera.screen_width) {
    surf2Dwrite(rgbx, surf, x_pixel * sizeof(rgbx), y_pixel,
                cudaBoundaryModeZero);
  }
}

void kernel_launcher(cudaArray_const_t array, Object *objects,
                     unsigned n_objects, Camera camera) {
  // Count the number of frames displayed
  static unsigned counter = 0;
  counter += 1;

  cuda(BindSurfaceToArray(surf, array));

  const int blocks =
      (camera.screen_width * camera.screen_height + THREADS_PER_BLOCK - 1) /
      THREADS_PER_BLOCK;

  if (blocks > 0) {
    kernel<<<blocks, THREADS_PER_BLOCK>>>(counter, objects, n_objects, camera);
  }
}
