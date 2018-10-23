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

struct Box {
  Vec3f bounds[2];
  Box() = default;
  Box(Vec3f a, Vec3f b) {
    Vec3f mini,maxi;
    mini.x = min(a.x,b.x);
    maxi.x = max(a.x,b.x);
    mini.y = min(a.y,b.y);
    maxi.y = max(a.y,b.y);
    mini.z = min(a.z,b.z);
    maxi.z = max(a.z,b.z);
    bounds[0] = mini;
    bounds[1] = maxi;
  }
  HD real inter(Rayf ray) {
    real tmin,tmax,tymin,tymax,tzmin,tzmax;

    tmin = (bounds[1-ray.sign[0]].x - ray.orig.x) * ray.inv_dir.x;
    tmax = (bounds[ray.sign[0]].x - ray.orig.x) * ray.inv_dir.x;
    tymin = (bounds[1-ray.sign[1]].y - ray.orig.y) * ray.inv_dir.y;
    tymax = (bounds[ray.sign[1]].y - ray.orig.y) * ray.inv_dir.y;
    tzmin = (bounds[1-ray.sign[2]].z - ray.orig.z) * ray.inv_dir.z;
    tzmax = (bounds[ray.sign[2]].z - ray.orig.z) * ray.inv_dir.z;

    if((tmin > tymax) || (tymin > tmax))
      return -1.0f;

    tmin = max(tmin,tymin);
    tmax = min(tmax,tymax);

    if((tmin > tzmax) || (tzmin > tmax))
      return -1.0f;

    tmin = max(tmin,tzmin);
    tmax = min(tmax,tzmax);

    if(tmin < 0) {
      if(tmax < 0) {
        return -1.0f;
      }
      return tmax;
    }

    return tmin;
  }
  HD Vec3f normal(Rayf ray, Vec3f pos) {
    if(ray.sign[0] && abs(pos.x - bounds[0].x) < 1e-3)
      return Vec3f{1.0f,0.0f,0.0f};
    if(!ray.sign[0] && abs(pos.x - bounds[1].x) < 1e-3)
      return Vec3f{-1.0f,0.0f,0.0f};
    if(ray.sign[1] && abs(pos.y - bounds[0].y) < 1e-3)
      return Vec3f{0.0f,1.0f,0.0f};
    if(!ray.sign[1] && abs(pos.y - bounds[1].y) < 1e-3)
      return Vec3f{0.0f,-1.0f,0.0f};
    if(ray.sign[2] && abs(pos.z - bounds[0].z) < 1e-3)
      return Vec3f{0.0f,0.0f,1.0f};
    if(!ray.sign[2] && abs(pos.z - bounds[1].z) < 1e-3)
      return Vec3f{0.0f,0.0f,-1.0f};
    return Vec3f{1.0f,1.0f,1.0f};
  }
};

enum class ObjectType { sphere, box, plane };

struct Object {
  Color color;
  ObjectType type;
  union {
    Sphere sphere;
    Plane plane;
    Box box;
  };
};
