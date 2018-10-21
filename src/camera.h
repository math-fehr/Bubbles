#pragma once

#include "geom.h"

struct Camera {
  Vec3f pos;
  Mat3f camera_to_world;
  real fov;           // The field of view in radians
  real scale;
  unsigned screen_width;  // The screen width in pixel
  unsigned screen_height; // The screen height in pixel

  Camera() = delete;
  Camera(Vec3f pos, Mat3f camera_to_world, real fov, unsigned screen_width,
         unsigned screen_height)
      : pos(pos), camera_to_world(camera_to_world), fov(fov),
        scale(tanf(fov * 0.5f)), screen_width(screen_width),
        screen_height(screen_height) {}

  HD Rayf get_ray(real x_pixel, real y_pixel) {
    real image_aspect_ratio = (real)screen_width / (real)screen_height;

    real x = (2.0f * ((x_pixel + 0.5f) / (real)screen_width) - 1) * scale;
    real y = (1 - 2.0f * ((y_pixel + 0.5f) / (real)screen_height)) * scale /
             image_aspect_ratio;

    Vec3f direction_camera{x,y,-1};
    Vec3f direction = camera_to_world * direction_camera;
    Vec3f position = pos;
    return Rayf(position,direction);
  }

  HD void set_fov(real fov) {
    this->fov = fov;
    scale = tanf(fov * 0.5f);
  }
};
