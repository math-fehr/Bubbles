#pragma once

#include "geom.h"

class Camera {
  Vec3f pos;      // position
  Vec3f basedir;  // direction camera points to
  Vec3f up;       // world up
  Vec3f xunit;    // x unit vector in camera plane
  Vec3f yunit;    // y unit vector in camera plane
  real lim_angle; // minimal angle between axis and up.
  real fov;       // The field of view in radians
  real scale;
public:
  unsigned screen_width;  // The screen width in pixel
  unsigned screen_height; // The screen height in pixel

  Camera() = delete;
  Camera(Vec3f pos, Vec3f basedir, Vec3f up, real fov, unsigned screen_width,
         unsigned screen_height)
    : pos(pos), basedir(basedir.normalize()), up(up.normalize()), fov(fov),
        scale(tanf(fov * 0.5f)), screen_width(screen_width),
        screen_height(screen_height) {
    update_units();
  }

  void update_units() {
    xunit = (basedir ^ up).normalize();
    yunit = (xunit ^ basedir).normalize();
  }

  HD Rayf get_ray(real x_pixel, real y_pixel) {
    real image_aspect_ratio = (real)screen_width / (real)screen_height;

    real x = (2.0f * ((x_pixel + 0.5f) / (real)screen_width) - 1) * scale;
    real y = (1 - 2.0f * ((y_pixel + 0.5f) / (real)screen_height)) * scale /
             image_aspect_ratio;

    // Here x and y are the coordinate in the camera plane (distance one from
    // pos) with actual 3D world length.

    // Vec3f direction_camera{x, y, -1};
    // Vec3f direction = camera_to_world * direction_camera;
    Vec3f dir = basedir + x * xunit + y * yunit;
    return Rayf(pos, dir);
  }

  void set_fov(real fov) {
    this->fov = fov;
    scale = tanf(fov * 0.5f);
  }

  void move_front(real step) { pos += step * basedir; }

  void move_lat(real step) { pos += step * xunit; }

  void move_up(real step) { pos += step * up; }

};
