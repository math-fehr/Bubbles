#pragma once

#include "geom.h"

struct UniformColor {
  Color color;

  HD Color get_color(Vec2f uv) const { return color; }
};

struct CheckBoard {
  Color color1, color2;
  float n_subdivision;

  HD Color get_color(Vec2f uv) const {
    int a = uv.x * n_subdivision * 2.0f;
    int b = uv.y * n_subdivision * 2.0f;
    if ((a + b) % 2) {
      return color1;
    } else {
      return color2;
    }
  }
};

enum class TextureType { uniform_color, checkboard };

struct Texture {
  TextureType type;
  real ambiant_factor;
  real diffusion_factor;
  real refract_factor = 0.0;
  real refract_index = 0.0;
  union {
    UniformColor uniform_color;
    CheckBoard checkboard;
  };

  HD Color get_color(Vec2f uv) const {
    switch (type) {
    case TextureType::uniform_color:
      return uniform_color.get_color(uv);
    case TextureType::checkboard: {
      return checkboard.get_color(uv);
    }
    default:
      return Color{0.0f, 0.0f, 0.0f};
    }
  }
};
