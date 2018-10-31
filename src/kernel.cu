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
  Vec2f uv;
};

__device__ Intersection intersect_all(Object *objects, unsigned n_objects,
                                      const Rayf &ray) {

  int front_object = -1;
  real intersection_point = 1.f / 0.f;


  for (int i = 0; i < n_objects; ++i) {
    float intersection_i = objects[i].intersect(ray);
    if (intersection_i > 0.f && intersection_i < intersection_point) {
      intersection_point = intersection_i;
      front_object = i;
    }
  }
  // TODO handle out of box things : add skybox

  IntersectionData inter_data =
      objects[front_object].intersection_data(ray, intersection_point);

  return Intersection{front_object,   objects[front_object], intersection_point,
                      inter_data.pos, inter_data.normal,     inter_data.uv};
}

__device__ Intersection intersect_all(const Scene &scene, const Rayf &ray) {
  return intersect_all(scene.objects, scene.n_objects, ray);
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

__device__ Color compute_phong_color(const Scene &scene,
                                     const Intersection &intersection,
                                     Rayf ray) {
  Color point_color = intersection.object.texture.get_color(intersection.uv);
  Color ambiant_color = point_color * scene.ambiant_light.color *
                        intersection.object.texture.ambiant_factor;

  Rayf light_ray = scene.light.ray_to_point(intersection.point);
  Intersection light_intersection = intersect_all(scene, light_ray);
  bool light_touch = intersection.object_id == light_intersection.object_id;
  light_touch &= (intersection.point - light_intersection.point).norm() < 1e-3;

  Color diffuse_color{0.0f, 0.0f, 0.0f};
  if (light_touch) {
    // Diffusion color
    real diffusion_factor = -intersection.normal | light_ray.dir;
    diffusion_factor = max(0.0f, min(1.0f, diffusion_factor));
    diffusion_factor *= intersection.object.texture.diffusion_factor;
    diffuse_color = point_color *
                    diffusion_factor * scene.light.color;
  }

  return diffuse_color + ambiant_color;
}

struct BouncingRays {
  bool has_refraction;
  real refraction_factor;
  Rayf refraction_ray;
};

__device__ Color compute_texture(const Scene &scene, Intersection intersection,
                                 Rayf ray) {
  // Refraction color
  if (intersection.object.texture.refract_factor > 0.01f) {
    Rayf refract_ray_out({}, {});
    if (intersection.object.is_in(ray.orig)) {
      bool has_refract = compute_refraction(
          ray.dir, intersection.point, intersection.normal,
          intersection.object.texture.refract_index, 1.0f, &refract_ray_out);
      if (!has_refract) {
        return Color{0.0f, 0.0f, 0.0f};
      }
    } else {
      Rayf refract_ray_in({}, {});
      bool has_refract = compute_refraction(
          ray.dir, intersection.point, intersection.normal, 1.0f,
          intersection.object.texture.refract_index, &refract_ray_in);
      if (!has_refract) {
        return Color{0.0f, 0.0f, 0.0f};
      }
      real out_point = intersection.object.intersect(refract_ray_in);
      has_refract = compute_refraction(
          refract_ray_in.dir, refract_ray_in(out_point),
          intersection.object.normal(refract_ray_in, out_point),
          intersection.object.texture.refract_index, 1.0f, &refract_ray_out);
      if (!has_refract) {
        return Color{0.0f, 0.0f, 0.0f};
      }
    }
    Intersection refract_intersection = intersect_all(scene, refract_ray_out);

    Color color_refract =
        intersection.object.texture.refract_factor *
        compute_phong_color(scene, refract_intersection, refract_ray_out);
    return color_refract;
  }
  return Color{0.0f, 0.0f, 0.0f};
}

__device__ Color cast_ray(const Scene &scene, Rayf ray) {
  Color final_color{0.0f, 0.0f, 0.0f};
  real factor = 1.0f;
  int recursion = 0;

  while (true) {
    // Compute the ambiant and diffuse color of the object intersected
    Intersection intersection = intersect_all(scene, ray);
    Color phong_color = compute_phong_color(scene, intersection, ray);

    bool stop_recursion = (recursion == NUM_REFL);

    // If we can't cast more rays, we augment the diffuse and ambiant factor
    if (stop_recursion) {
      phong_color /= intersection.object.texture.ambiant_factor +
                     intersection.object.texture.diffusion_factor;
    }

    // We add the ambiant and diffuse color to the total color
    final_color += phong_color * factor;

    bool cast_more_rays =
        !stop_recursion && intersection.object.texture.refract_factor != 0.0f;

    // If there is no ray left to compute, we have the final color
    if (!cast_more_rays) {
      return final_color;
    }

    Rayf ray_refract;
    bool is_in = intersection.object.is_in(ray.orig);
    real index_in = is_in ? intersection.object.texture.refract_index : 1.0f;
    real index_out = !is_in ? intersection.object.texture.refract_index : 1.0f;
    bool has_refraction =
        compute_refraction(ray.dir, intersection.point, intersection.normal,
                           index_in, index_out, &ray_refract);

    if (has_refraction) {
      ray = ray_refract;
      factor *= intersection.object.texture.refract_factor;
      recursion++;
      continue;
    } else {
      return final_color;
    }
  }
}

/**
 * Entry CUDA kernel. This is the code for one pixel
 */
__global__ void kernel(Scene scene, Camera camera) {
  // pixel coordinates
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

  if (idx < camera.screen_height * camera.screen_width) {
    int x_pixel = idx % camera.screen_width;
    int y_pixel = idx / camera.screen_width;

    Rayf ray = camera.get_ray(x_pixel, y_pixel);

    Color color = cast_ray(scene, ray);

    RGBA rgbx = color.to8bit(camera.gamma);

    surf2Dwrite(rgbx, surf, x_pixel * sizeof(rgbx), y_pixel,
                cudaBoundaryModeZero);
  }
}

void kernel_launcher(cudaArray_const_t array, Scene scene, Camera camera) {

  cuda(BindSurfaceToArray(surf, array));

  const int blocks =
      (camera.screen_width * camera.screen_height + THREADS_PER_BLOCK - 1) /
      THREADS_PER_BLOCK;

  if (blocks > 0) {
    kernel<<<blocks, THREADS_PER_BLOCK>>>(scene, camera);
  }
}
