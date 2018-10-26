#include <utility>

#include "assert_cuda.hpp"
#include "kernel.cuh"

#include "camera.h"
#include "geom.h"
#include "light.h"
#include "object.h"

using namespace std;

// The surface where CUDA will write
surface<void, cudaSurfaceType2D> surf;

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
    float intersection_i = objects[i].intersect(ray);
    if (intersection_i > 0.f && intersection_i < intersection_point) {
      intersection_point = intersection_i;
      front_object = i;
    }
  }

  return Intersection{front_object, objects[front_object], intersection_point,
                      ray(intersection_point),
                      objects[front_object].normal(ray, intersection_point)};
}

__device__ Color compute_texture__(Object *objects, unsigned n_objects,
                                   PointLight light, AmbiantLight ambiant_light,
                                   Intersection intersection, Rayf ray) {
  Vec2f uv = intersection.object.uv(intersection.point);
  Color point_color = intersection.object.texture.get_color(uv);
  Color ambiant_color = point_color * ambiant_light.color *
                        intersection.object.texture.ambiant_factor;

  Rayf light_ray = light.ray_to_point(intersection.point);
  Intersection light_intersection =
      intersect_all(objects, n_objects, light_ray);
  bool light_touch = intersection.object_id == light_intersection.object_id;
  light_touch &= (intersection.point - light_intersection.point).norm() < 1e-3;

  Color diffuse_color{0.0f, 0.0f, 0.0f};
  if (light_touch) {
    // Diffusion color
    real diffusion_factor = -intersection.normal | light_ray.dir;
    diffusion_factor = max(0.0f, min(1.0f, diffusion_factor));
    diffusion_factor *= intersection.object.texture.diffusion_factor;
    diffuse_color = intersection.object.texture.uniform_color.color *
                    diffusion_factor * light.color;
  }

  return diffuse_color + ambiant_color;
}

__device__ bool compute_refraction(Vec3f incident, Vec3f inter, Vec3f normal,
                                   real index_in, real index_out,
                                   Rayf *out_ray) {
  real n = index_in / index_out;
  real cos_i = -(incident | normal);
  real sin_t_squared = n * n * (1.0f - cos_i * cos_i);
  if (sin_t_squared > 1.0f) {
    return false;
  }
  real cos_t = sqrtf(1.0f - sin_t_squared);
  Vec3f out_ray_dir = n * incident + (n * cos_i - cos_t) * normal;
  out_ray_dir.normalize();
  Vec3f out_ray_orig = inter + 1e-3f * out_ray_dir;
  *out_ray = Rayf(out_ray_orig, out_ray_dir);
  return true;
}

__device__ Color compute_texture(Object *objects, unsigned n_objects,
                                 PointLight light, AmbiantLight ambiant_light,
                                 Intersection intersection, Rayf ray) {
  Vec2f uv = intersection.object.uv(intersection.point);
  Color point_color = intersection.object.texture.get_color(uv);
  Color ambiant_color = point_color * ambiant_light.color *
                        intersection.object.texture.ambiant_factor;

  Rayf light_ray = light.ray_to_point(intersection.point);
  Intersection light_intersection =
      intersect_all(objects, n_objects, light_ray);
  bool light_touch = intersection.object_id == light_intersection.object_id;
  light_touch &= (intersection.point - light_intersection.point).norm() < 1e-3;

  Color diffuse_color{0.0f, 0.0f, 0.0f};
  if (light_touch) {
    // Diffusion color
    real diffusion_factor = -intersection.normal | light_ray.dir;
    diffusion_factor = max(0.0f, min(1.0f, diffusion_factor));
    diffusion_factor *= intersection.object.texture.diffusion_factor;
    diffuse_color = intersection.object.texture.uniform_color.color *
                    diffusion_factor * light.color;
  }

  // Refraction color
  if (intersection.object.texture.refract_factor > 0.0f) {
    Rayf refract_ray_out({}, {});
    if (intersection.object.is_in(ray.orig)) {
      bool has_refract = compute_refraction(
          ray.dir, intersection.point, intersection.normal,
          intersection.object.texture.refract_index, 1.0f, &refract_ray_out);
      if (!has_refract) {
        return diffuse_color + ambiant_color;
      }
    } else {
      Rayf refract_ray_in({}, {});
      bool has_refract = compute_refraction(
          ray.dir, intersection.point, intersection.normal, 1.0f,
          intersection.object.texture.refract_index, &refract_ray_in);
      if (!has_refract) {
        return diffuse_color + ambiant_color;
      }
      real out_point = intersection.object.intersect(refract_ray_in);
      has_refract = compute_refraction(
          refract_ray_in.dir, refract_ray_in(out_point),
          intersection.object.normal(refract_ray_in, out_point),
          intersection.object.texture.refract_index, 1.0f, &refract_ray_out);
      if (!has_refract) {
        return diffuse_color + ambiant_color;
      }
    }
    Intersection refract_intersection =
        intersect_all(objects, n_objects, refract_ray_out);

    Color color_refract =
        intersection.object.texture.refract_factor *
        compute_texture__(objects, n_objects, light, ambiant_light,
                          refract_intersection, refract_ray_out);
    return diffuse_color + ambiant_color + color_refract;
  }
  return diffuse_color + ambiant_color;
}

/**
 * Entry CUDA kernel. This is the code for one pixel
 */
__global__ void kernel(int counter, Object *objects, unsigned n_objects,
                       Camera camera) {
  PointLight light{Vec3f{-30.0f, 0.0f, 0.0f}, Color{1, 1, 1}};
  AmbiantLight ambiant_light{1.0f, 1.0f, 1.0f};

  // pixel coordinates
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int x_pixel = idx % camera.screen_width;
  int y_pixel = idx / camera.screen_width;

  Rayf ray = camera.get_ray(x_pixel, y_pixel);

  RGBA rgbx;
  rgbx.r = 0, rgbx.g = 0, rgbx.b = 0;

  Intersection intersection = intersect_all(objects, n_objects, ray);
  Rayf light_ray = light.ray_to_point(intersection.point);

  Vec3f normal_vec = intersection.object.normal(ray, intersection.distance);

  Color color = compute_texture(objects, n_objects, light, ambiant_light,
                                intersection, ray);
  rgbx = color.to8bit(camera.gamma);

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
