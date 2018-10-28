#pragma once

#include "geom.h"
#include "texture.h"

struct IntersectionData {
  Vec3f pos;
  Vec3f normal;
  Vec2f uv;
};

struct Sphere {
  Vec3f center;
  real radius2;
  Sphere() = default;
  HD Sphere(Vec3f center, real radius)
      : center(center), radius2(radius * radius) {}
  HD real inter(Rayf ray) const {
    real pi = ray.projindex(center);
    Vec3f pn = ray(pi);
    real n2 = (center - pn).norm2();
    if (n2 > radius2) {
      return -1;
    } else {
      return pi - sqrt(radius2 - n2);
    }
  }
  HD Vec3f normal(Vec3f pos) const { return (center - pos).normalized(); }
  HD Vec2f uv(Vec3f pos) const {
    Vec3f d = (center - pos).normalized();
    return Vec2f{0.5f + atan2f(d.z, d.x) / 2.f * 3.14159f,
                 0.5f - asinf(d.y) / 3.14159f};
  }
  HD IntersectionData intersection_data(const Rayf &ray,
                                        const Vec3f &pos) const {
    return IntersectionData{pos, normal(pos), uv(pos)};
  }

  HD bool is_in(Vec3f pos) const { return (pos - center).norm2() < radius2; }
};

// The plane follows the equation normal_vec | point = constant
struct Plane {
  Vec3f normal_vec;
  real constant;
  Plane() = default;
  HD Plane(Vec3f normal_vec, real constant)
      : normal_vec(normal_vec.normalized()), constant(constant) {}
  HD real inter(Rayf ray) const {
    float dot = ray.dir | normal_vec;
    if (abs(dot) < 1e-6) {
      return -1;
    }
    return (constant - (ray.orig | normal_vec)) / dot;
  }
  HD Vec3f normal(Rayf ray) const {
    if ((ray.orig | normal_vec) < 0) {
      return -normal_vec;
    } else {
      return normal_vec;
    }
  }
};

struct Box {
  Vec3f bounds[2];
  Box() = default;
  Box(Vec3f a, Vec3f b) {
    Vec3f mini, maxi;
    mini.x = min(a.x, b.x);
    maxi.x = max(a.x, b.x);
    mini.y = min(a.y, b.y);
    maxi.y = max(a.y, b.y);
    mini.z = min(a.z, b.z);
    maxi.z = max(a.z, b.z);
    bounds[0] = mini;
    bounds[1] = maxi;
  }

  HD real inter(Rayf ray) const {
    real tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = (bounds[1 - ray.sign[0]].x - ray.orig.x) * ray.inv_dir.x;
    tmax = (bounds[ray.sign[0]].x - ray.orig.x) * ray.inv_dir.x;
    tymin = (bounds[1 - ray.sign[1]].y - ray.orig.y) * ray.inv_dir.y;
    tymax = (bounds[ray.sign[1]].y - ray.orig.y) * ray.inv_dir.y;
    tzmin = (bounds[1 - ray.sign[2]].z - ray.orig.z) * ray.inv_dir.z;
    tzmax = (bounds[ray.sign[2]].z - ray.orig.z) * ray.inv_dir.z;

    if ((tmin > tymax) || (tymin > tmax)) return -1.0f;

    tmin = max(tmin, tymin);
    tmax = min(tmax, tymax);

    if ((tmin > tzmax) || (tzmin > tmax)) return -1.0f;

    tmin = max(tmin, tzmin);
    tmax = min(tmax, tzmax);

    if (tmin < 0) {
      if (tmax < 0) {
        return -1.0f;
      }
      return tmax;
    }

    return tmin;
  }

  HD Vec3f normal(Rayf ray, Vec3f pos) const {
    int is_interior = (is_in(ray.orig)) ? 1 : 0;
    real x_0 = !ray.sign[0] ? 1.0f / 0.0f : abs(pos.x - bounds[is_interior].x);
    real x_1 =
        ray.sign[0] ? 1.0f / 0.0f : abs(pos.x - bounds[1 - is_interior].x);
    real y_0 = !ray.sign[1] ? 1.0f / 0.0f : abs(pos.y - bounds[is_interior].y);
    real y_1 =
        ray.sign[1] ? 1.0f / 0.0f : abs(pos.y - bounds[1 - is_interior].y);
    real z_0 = !ray.sign[2] ? 1.0f / 0.0f : abs(pos.z - bounds[is_interior].z);
    real z_1 =
        ray.sign[2] ? 1.0f / 0.0f : abs(pos.z - bounds[1 - is_interior].z);
    real mini = min(x_0, min(x_1, min(y_0, min(y_1, min(z_0, z_1)))));
    if (x_0 == mini) {
      return Vec3f{-1.0f, 0.0f, 0.0f};
    } else if (x_1 == mini) {
      return Vec3f{1.0f, 0.0f, 0.0f};
    } else if (y_0 == mini) {
      return Vec3f{0.0f, -1.0f, 0.0f};
    } else if (y_1 == mini) {
      return Vec3f{0.0f, 1.0f, 0.0f};
    } else if (z_0 == mini) {
      return Vec3f{0.0f, 0.0f, -1.0f};
    } else {
      return Vec3f{0.0f, 0.0f, 1.0f};
    }
  }

  HD Vec2f uv(Vec3f pos) const {
    real x_uv = clamp((bounds[1].x - pos.x) / (bounds[1].x - bounds[0].x));
    real y_uv = clamp((bounds[1].y - pos.y) / (bounds[1].y - bounds[0].y));
    real z_uv = clamp((bounds[1].z - pos.z) / (bounds[1].z - bounds[0].z));
    real x_min = abs(pos.x - bounds[0].x);
    real x_max = abs(pos.x - bounds[1].x);
    real y_min = abs(pos.y - bounds[0].y);
    real y_max = abs(pos.y - bounds[1].y);
    real z_min = abs(pos.z - bounds[0].z);
    real z_max = abs(pos.z - bounds[1].z);
    real mini =
        min(x_min, min(x_max, min(y_min, min(y_max, min(z_min, z_max)))));
    if (mini == x_min || mini == x_max) {
      return Vec2f{y_uv, z_uv};
    } else if (mini == y_min || mini == y_max) {
      return Vec2f{x_uv, z_uv};
    } else {
      return Vec2f{x_uv, y_uv};
    }
  }

  HD IntersectionData intersection_data(const Rayf &ray, Vec3f pos) const {
    int is_interior = (is_in(ray.orig)) ? 1 : 0;
    real x_0 = !ray.sign[0] ? 1.0f / 0.0f : abs(pos.x - bounds[is_interior].x);
    real x_1 =
        ray.sign[0] ? 1.0f / 0.0f : abs(pos.x - bounds[1 - is_interior].x);
    real y_0 = !ray.sign[1] ? 1.0f / 0.0f : abs(pos.y - bounds[is_interior].y);
    real y_1 =
        ray.sign[1] ? 1.0f / 0.0f : abs(pos.y - bounds[1 - is_interior].y);
    real z_0 = !ray.sign[2] ? 1.0f / 0.0f : abs(pos.z - bounds[is_interior].z);
    real z_1 =
        ray.sign[2] ? 1.0f / 0.0f : abs(pos.z - bounds[1 - is_interior].z);
    real mini = min(x_0, min(x_1, min(y_0, min(y_1, min(z_0, z_1)))));
    real x_uv = clamp((bounds[1].x - pos.x) / (bounds[1].x - bounds[0].x));
    real y_uv = clamp((bounds[1].y - pos.y) / (bounds[1].y - bounds[0].y));
    real z_uv = clamp((bounds[1].z - pos.z) / (bounds[1].z - bounds[0].z));
    if (x_0 == mini) {
      return IntersectionData{pos, Vec3f{-1.0f, 0.0f, 0.0f}, Vec2f{y_uv, z_uv}};
    } else if (x_1 == mini) {
      return IntersectionData{pos, Vec3f{1.0f, 0.0f, 0.0f}, Vec2f{y_uv, z_uv}};
    } else if (y_0 == mini) {
      return IntersectionData{pos, Vec3f{0.0f, -1.0f, 0.0f}, Vec2f{x_uv, z_uv}};
    } else if (y_1 == mini) {
      return IntersectionData{pos, Vec3f{0.0f, 1.0f, 0.0f}, Vec2f{x_uv, z_uv}};
    } else if (z_0 == mini) {
      return IntersectionData{pos, Vec3f{0.0f, 0.0f, -1.0f}, Vec2f{x_uv, y_uv}};
    } else {
      return IntersectionData{pos, Vec3f{0.0f, 0.0f, 1.0f}, Vec2f{x_uv, y_uv}};
    }
  }

  HD bool is_in(Vec3f pos) const { return bounds[0] < pos && pos < bounds[1]; }
};

enum class ObjectType { sphere, box, plane };

struct Object {
  Texture texture;
  ObjectType type;
  union {
    Sphere sphere;
    Plane plane;
    Box box;
  };

  HD real intersect(Rayf ray) const {
    switch (type) {
    case ObjectType::sphere:
      return sphere.inter(ray);
    case ObjectType::plane:
      return plane.inter(ray);
    case ObjectType::box:
      return box.inter(ray);
    default:
      return -1.0f;
    }
  }

  HD Vec3f normal(Rayf ray, real intersection_distance) const {
    switch (type) {
    case ObjectType::sphere:
      return sphere.normal(ray(intersection_distance));
    case ObjectType::plane:
      return plane.normal(ray);
    case ObjectType::box:
      return box.normal(ray, ray(intersection_distance));
    default:
      return {0.0f, 0.0f, 0.0f};
    }
  }

  HD Vec2f uv(Vec3f intersection_point) const {
    switch (type) {
    case ObjectType::box:
      return box.uv(intersection_point);
    default:
      return Vec2f{0.0f, 0.0f};
    }
  }

  HD IntersectionData intersection_data(const Rayf &ray, real distance) const {
    switch (type) {
    case ObjectType::sphere:
      return sphere.intersection_data(ray, ray(distance));
    case ObjectType::box:
      return box.intersection_data(ray, ray(distance));
    default:
      return {};
    }
  }

  HD bool is_in(Vec3f point) const {
    switch (type) {
    case ObjectType::box:
      return box.is_in(point);
    case ObjectType::sphere:
      return sphere.is_in(point);
    default:
      return false;
    }
  }
};
