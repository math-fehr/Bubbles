#pragma once

#include "geom.h"

class Camera {
  Vec3f pos;     // position
  Vec3f basedir; // direction camera points to
  Vec3f up;      // world up
  Vec3f xunit;   // x unit vector in camera plane
  Vec3f yunit;   // y unit vector in camera plane
  real fov;      // The field of view in radians
  real scale;

public:
  unsigned screen_width;  // The screen width in pixel
  unsigned screen_height; // The screen height in pixel
  real gamma;             // gamma color space to render on screen

  Camera() = delete;
  Camera(Vec3f pos, Vec3f basedir, Vec3f up, real fov, unsigned screen_width,
         unsigned screen_height)
      : pos(pos), basedir(basedir.normalized()), up(up.normalized()), fov(fov),
        scale(tanf(fov * 0.5f)), screen_width(screen_width),
        screen_height(screen_height), gamma(1.5f) {
    update_units();
  }

  void update_units() {
    xunit = (basedir ^ up).normalized();
    yunit = (xunit ^ basedir).normalized();
  }

  HD Rayf get_ray(real x_pixel, real y_pixel) const {
    real image_aspect_ratio = (real)screen_width / (real)screen_height;

    // TODO expand this expression to simple affine function (x - xoff) *
    // xunitscaled

    real x = (2.0f * ((x_pixel + 0.5f) / (real)screen_width) - 1) * scale;
    real y = (1 - 2.0f * ((y_pixel + 0.5f) / (real)screen_height)) * scale /
             image_aspect_ratio;

    // Here x and y are the coordinate in the camera plane (distance one from
    // pos) with actual 3D world length.

    Vec3f dir = basedir + x * xunit + y * yunit;
    return Rayf(pos, dir);
  }

  void set_fov(real fov) {
    this->fov = fov;
    scale = tanf(fov * 0.5f);
  }

  void set_pos(Vec3f pos) { this->pos = pos; }

  void set_dir(Vec3f dir) {
    this->basedir = dir.normalized();
    update_units();
  }

  void move_front(real step) { pos += step * basedir; }

  void move_lat(real step) { pos += step * xunit; }

  void move_up(real step) { pos += step * up; }

  void rotate_lat(real angle) {
    // we must have |angle| < pi/2 radians or it will be wrapped by tan.
    basedir += (basedir ^ up) * tan(angle);
    basedir.normalize();
    update_units();
  }

  void rotate_up(real angle) {
    Vec3f new_basedir = basedir + tan(angle) * yunit;
    new_basedir.normalize();
    real check = xunit | (new_basedir ^ up);
    if (check < 0.001) {
      Vec3f upd = (basedir ^ up) ^ up;
      // here basedir + upd is colinear to up
      upd -= upd.normalized() * 0.001;
      basedir += upd;
    } else {
      basedir = new_basedir;
    }
    update_units();
  }

  Vec3f get_pos() const { return pos; }
};
