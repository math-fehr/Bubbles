#pragma once

#include "geom.h"

struct PointLight {
  Vec3f center;
  Color color;
  PointLight() = default;
  // Get the ray pointing to point, and coming from the light
  HD Rayf ray_to_point(Vec3f point) {
    return Rayf(center, point - center);
  }
};
