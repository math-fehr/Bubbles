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

struct IntersectionBase {
  int id;
  real distance;
};

struct Intersection : public IntersectionBase, public IntersectionData {
  HD Intersection(IntersectionBase ib, IntersectionData id)
      : IntersectionBase(ib), IntersectionData(id) {}
};

__device__ IntersectionBase intersect_scene(const Scene &scene, Rayf ray) {

  IntersectionBase res{-1, 1.f / 0.f};

  for (int i = 0; i < scene.n_objects; ++i) {
    real distance = scene[i].inter(ray);
    if (distance > 0. and distance < res.distance) {
      res = IntersectionBase{i, distance};
    }
  }
  return res;
}

__device__ Intersection intersect_scene_full(const Scene &scene, Rayf ray) {

  IntersectionBase res = intersect_scene(scene, ray);

  return Intersection(res, scene[res.id].inter_data(ray, res.distance));
}

/*__device__ Intersection intersect_all(Object *objects, unsigned n_objects,
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
  }*/

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
  Vec3f out_ray_orig = inter + 1e-2f * out_ray_dir;
  *out_ray = Rayf(out_ray_orig, out_ray_dir);
  return true;
}

__device__ Rayf compute_reflexion(Vec3f incident, Vec3f inter, Vec3f normal) {
  Vec3f refl_dir = incident + 2 * (-incident | normal) * normal;
  inter += 1e-2f * refl_dir;
  return Rayf(inter, refl_dir);
}

__device__ Color compute_phong_color(const Scene &scene,
                                     const Intersection &intersection,
                                     Rayf ray) {
  const Object &obj = scene[intersection.id];
  Color point_color =
      obj.texture.get_color(intersection.pos, intersection.uv);
  Color ambiant_color =
      point_color * scene.ambiant_light.color * obj.texture.factors.ambiant;

  Rayf light_ray = scene.light.ray_to_point(intersection.pos);
  IntersectionBase light_intersection = intersect_scene(scene, light_ray);
  bool light_touch = intersection.id == light_intersection.id and
                     abs((scene.light.center - intersection.pos).norm() -
                         light_intersection.distance) < 1e-3;

  Color diffuse_color{0.0f, 0.0f, 0.0f};
  if (light_touch) {
    // Diffusion color
    real diffusion_factor = -intersection.normal | light_ray.dir;
    diffusion_factor = max(0.0f, diffusion_factor);
    diffusion_factor *= obj.texture.factors.diffuse;
    diffuse_color = point_color * diffusion_factor * scene.light.color;
  }

  return diffuse_color + ambiant_color;
}

__device__ Color cast_ray(const Scene &scene, Rayf ray) {
  Color final_color{0.0f, 0.0f, 0.0f};
  Rayf rays[NUM_REFL * 3 + 1];
  // color filters to be applied to the output of each ray.
  Color filters[NUM_REFL * 3 + 1];
  // color filters to be applied to the output of each ray.
  int depths[NUM_REFL * 3 + 1];
  int last = 0;

  // initialisation
  rays[0] = ray;
  filters[0] = white;
  depths[0] = 0;

  constexpr real power_cap =
      0.01f; // min amount of power to be worth of tracing

  while (last >= 0) {
    // poping current ray from stack
    Rayf ray = rays[last];
    Color filter = filters[last];
    int depth = depths[last];
    last--;

    // find next intersected object
    Intersection intersection = intersect_scene_full(scene, ray);
    if (intersection.id == -1) continue;

    // Compute the ambiant and diffuse color of the object intersected
    const Object &object = scene[intersection.id];
    Color point_color = object.texture.get_color(intersection.pos,intersection.uv);
    Color phong_color = compute_phong_color(scene, intersection, ray);

    if (depth >= NUM_REFL) {
      final_color += phong_color * filter;
      continue;
    }

    real diffuse_power = object.texture.factors.opacity;

    //  ____       __                _   _
    // |  _ \ ___ / _|_ __ __ _  ___| |_(_) ___  _ __
    // | |_) / _ \ |_| '__/ _` |/ __| __| |/ _ \| '_ \
    // |  _ <  __/  _| | | (_| | (__| |_| | (_) | | | |
    // |_| \_\___|_| |_|  \__,_|\___|\__|_|\___/|_| |_|

    if (filter.max() * object.texture.factors.refract < power_cap) {
      diffuse_power += object.texture.factors.refract;
    } else {
      // refraction
      Rayf ray_refract;
      bool is_in = object.is_in(ray.orig);
      real index_in = is_in ? object.texture.factors.index : 1.0f;
      real index_out = !is_in ? object.texture.factors.index : 1.0f;
      bool has_refraction =
          compute_refraction(ray.dir, intersection.pos, intersection.normal,
                             index_in, index_out, &ray_refract);
      if (!has_refraction) {
        diffuse_power += object.texture.factors.refract;
      } else {
        // push refracted ray
        last++;
        rays[last] = ray_refract;
        filters[last] = filter * object.texture.factors.refract;
        depths[last] = depth + 1;
      }
    }

    //  ____       __ _           _
    // |  _ \ ___ / _| | _____  _(_) ___  _ __
    // | |_) / _ \ |_| |/ _ \ \/ / |/ _ \| '_ \
    // |  _ <  __/  _| |  __/>  <| | (_) | | | |
    // |_| \_\___|_| |_|\___/_/\_\_|\___/|_| |_|

    if (filter.max() * object.texture.factors.reflect < power_cap) {
      diffuse_power += object.texture.factors.reflect;
    } else {
      last++;
      rays[last] =
          compute_reflexion(ray.dir, intersection.pos, intersection.normal);
      filters[last] = filter * object.texture.factors.reflect;
      depths[last] = depth + 1;
    }

    final_color += phong_color * filter * diffuse_power;
  }
  return final_color;
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
