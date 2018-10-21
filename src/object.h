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
};

struct Plane {
  Vec3f normal;
  real constant;
  Plane() = default;
  HD Plane(Vec3f normal, real constant) : normal(normal), constant(constant) {}
  HD real inter(Rayf ray) {
    float dot = ray.dir | normal;
    if (abs(dot) < 1e-6) {
      return -1;
    }
    return (constant - (ray.orig | normal)) / dot;
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
