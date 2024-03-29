#include <utility>

#include "assert_cuda.hpp"
#include "kernel.cuh"

#include "camera.h"
#include "geom.h"
#include "light.h"
#include "object.h"

using namespace std;

// File containing the functions used for ray tracing

// Base struct for an intersection
struct IntersectionBase {
  int id;
  real distance;
};

// Full information for an intersection.
struct Intersection : public IntersectionBase, public IntersectionData {
  HD Intersection(IntersectionBase ib, IntersectionData id)
      : IntersectionBase(ib), IntersectionData(id) {}
};

// Intersect a ray to all the objects in the scene and return the index and
// distance of the nearest object.
__device__ IntersectionBase intersect_scene(const Scene &scene, Rayf ray) {
  IntersectionBase res{-1, 1.f / 0.f};

  // Intersect all objects and take the nearest one
  for (int i = 0; i < scene.n_objects; ++i) {
    real distance = scene[i].inter(ray);
    if (distance > 0. and distance < res.distance) {
      res = IntersectionBase{i, distance};
    }
  }
  return res;
}

// Intersect a ray to all the objects in the scene and return all the
// information of this object in this point.
__device__ Intersection intersect_scene_full(const Scene &scene, Rayf ray) {

  IntersectionBase res = intersect_scene(scene, ray);

  return Intersection(res, scene[res.id].inter_data(ray, res.distance));
}

// Compute the refracting ray of an intersection
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

// Compute the reflecting ray of an intersection
__device__ Rayf compute_reflexion(Vec3f incident, Vec3f inter, Vec3f normal) {
  Vec3f refl_dir = incident + 2 * (-incident | normal) * normal;
  inter += 1e-2f * refl_dir;
  return Rayf(inter, refl_dir);
}

// Compute the phong color of an object
__device__ Color compute_phong_color(const Scene &scene,
                                     const Intersection &intersection,
                                     Rayf ray) {
  const Object &obj = scene[intersection.id];
  Color point_color = obj.texture.get_color(intersection.pos, intersection.uv);
  Color ambiant_color =
      point_color * scene.ambiant_light.color * obj.texture.factors.ambiant;

  Rayf light_ray = scene.light.ray_to_point(intersection.pos);
  Color light_color = scene.light.color;
  real max_distance = (scene.light.center - intersection.pos).norm() - 1e-3;

  // compute shadow : which amount of light reach the point
  for (int i = 0; i < scene.n_objects; ++i) {
    real distance = scene[i].inter(light_ray);
    if (distance > 0. and distance < max_distance) {
      Vec2f uv = scene[i].uv(light_ray, distance);
      light_color *=
          scene[i].texture.factors.refract * scene[i].texture.factors.refract *
          (0.15 * scene[i].texture.get_color(light_ray(distance), uv) +
           0.85 * white);
      if (light_color.max() < 1e-3) break;
    }
  }

  real diffusion_factor = -intersection.normal | light_ray.dir;
  diffusion_factor = max(0.0f, diffusion_factor);
  diffusion_factor *= obj.texture.factors.diffuse;
  Color diffuse_color = point_color * diffusion_factor * light_color;

  Vec3f light_refl_dir =
      light_ray.dir +
      2 * (-light_ray.dir | intersection.normal) * intersection.normal;
  real specular_factor = -light_refl_dir | ray.dir;
  specular_factor = max(0.0f, specular_factor);
  specular_factor = pow(specular_factor, obj.texture.factors.shininess);
  specular_factor *= obj.texture.factors.specular;
  Color specular_color = specular_factor * light_color;

  return specular_color + diffuse_color + ambiant_color;
}

// Cast a ray and get the color associated.
// The color contains the reflective and refractive part
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

    Vec3f light_proj = ray.projpoint(scene.light.center);
    real light_dist = (light_proj - scene.light.center).norm();
    if (light_dist < 0.5f) {
      real distance_light = (light_proj - ray.orig).norm2();
      real distance_object = (intersection.pos - ray.orig).norm2();
      if (distance_light < distance_object && distance_light > 0) {
        real factor = 2.0f * (0.5f - light_dist) * (0.5f - light_dist);
        final_color += filter * scene.light.color * factor;
      }
    }

    // Phong part
    const Object &object = scene[intersection.id];
    Color point_color =
        object.texture.get_color(intersection.pos, intersection.uv);
    Color phong_color = compute_phong_color(scene, intersection, ray);

    if (depth >= NUM_REFL) {
      final_color += phong_color * filter;
      continue;
    }

    real diffuse_power = object.texture.factors.opacity;

    // Refractive part

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

    // Reflective part

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

// Entry CUDA kernel. This is the code for one pixel

__global__ void kernel(Scene scene, Camera camera, cudaSurfaceObject_t surf) {
  // pixel coordinates
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

  if (idx < camera.screen_height * camera.screen_width) {
    int x_pixel = idx % camera.screen_width;
    int y_pixel = idx / camera.screen_width;

    Rayf ray = camera.get_ray(x_pixel, y_pixel);

    Color color = cast_ray(scene, ray);

    RGBA rgbx = color.to8bit(camera.gamma);
    uchar4 rgbxchar = {rgbx.r, rgbx.g, rgbx.b, rgbx.a};

    surf2Dwrite(rgbxchar, surf, (int)(x_pixel * sizeof(rgbx)), y_pixel,
                cudaBoundaryModeZero);
  }
}

// Launch the kernel by associating each pixel with a thread
void kernel_launcher(cudaArray_const_t array, Scene scene, Camera camera) {
  // The surface where CUDA will write
  // This is the pixels that will be displayed on screen.
  cudaSurfaceObject_t surf = 0;

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = (cudaArray_t)array;
  cuda(CreateSurfaceObject(&surf, &resDesc));
  // cuda(BindSurfaceToArray(surf, array));

  const int blocks =
      (camera.screen_width * camera.screen_height + THREADS_PER_BLOCK - 1) /
      THREADS_PER_BLOCK;

  if (blocks > 0) {
    kernel<<<blocks, THREADS_PER_BLOCK>>>(scene, camera, surf);
  }
}
