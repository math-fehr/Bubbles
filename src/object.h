#pragma once

#include "geom.h"

struct Sphere {
  Vec3f center;
  real radius2;
  Sphere() = default;
  HD Sphere(Vec3f center, real radius)
      : center(center), radius2(radius * radius) {}
  HD real inter(Rayf ray) {
    real pi = ray.projindex(center);
    Vec3f pn = ray(pi);
    real n2 = (center - pn).norm2();
    if (n2 > radius2) {
      return -1;
    } else {
      return pi - sqrt(radius2 - n2);
    }
  }
  HD Vec3f normal(Vec3f pos) { return (center - pos).normalized(); }
};

// The plane follows the equation normal_vec | point = constant
struct Plane {
  Vec3f normal_vec;
  real constant;
  Plane() = default;
  HD Plane(Vec3f normal_vec, real constant)
      : normal_vec(normal_vec.normalized()), constant(constant) {}
  HD real inter(Rayf ray) {
    float dot = ray.dir | normal_vec;
    if (abs(dot) < 1e-6) {
      return -1;
    }
    return (constant - (ray.orig | normal_vec)) / dot;
  }
  HD Vec3f normal(Rayf ray) {
    if ((ray.orig | normal_vec) < 0) {
      return -normal_vec;
    } else {
      return normal_vec;
    }
  }
};

enum class ObjectType { sphere, plane };

struct Object {
  Color color;
  ObjectType type;
  union {
    Sphere sphere;
    Plane plane;
  };
};
