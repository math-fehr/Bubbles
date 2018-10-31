#pragma once

#include "geom.h"
#include "noise.h"
#include "object.h"

inline HD Color hsv2rgb(Vec3f c) {
  Vec3f p =
      (Vec3f{c.x, c.x, c.x} + Vec3f{1.0, 2.0 / 3.0, 1.0 / 3.0}).frac() * 6.0f;
  p -= Vec3f{3.0, 3.0, 3.0};
  p = p.abs();
  p = (p - Vec3f{1.0, 1.0, 1.0}).clamp();
  p = c.z * Vec3f{3.0, 3.0, 3.0}.mix(p, c.y);
  return {p.x, p.y, p.z};
}

struct UniformColor {
  Color color;

  HD Color get_color(Vec2f uv) const { return color; }
};

struct CheckBoard {
  Color color1, color2;
  float n_subdivision;

  HD Color get_color(Vec2f uv) const {
    int a =
        clamp(uv.x * n_subdivision * 2.0f, 0.1f, n_subdivision * 2.0f - 0.1f);
    int b =
        clamp(uv.y * n_subdivision * 2.0f, 0.1f, n_subdivision * 2.0f - 0.1f);
    if ((a + b) % 2) {
      return color1;
    } else {
      return color2;
    }
  }
};

// Texture for a soap bubble
struct BubbleTexture {
  HD Color get_color(Vec3f pos) const {
    real r = fractal_perlin(pos, 5, 2.0, 0.5);
    return hsv2rgb(Vec3f{r, 1.0, 1.0});
  }
};

enum class TextureType { uniform_color, checkboard, bubble };

struct Texture {
  TextureType type;
  real ambiant_factor;
  real diffusion_factor;
  real refract_factor = 0.0;
  real refract_index = 1.0;
  union {
    UniformColor uniform_color;
    CheckBoard checkboard;
    BubbleTexture bubble;
  };

  HD Color get_color(Vec3f pos, Vec2f uv) const {
    switch (type) {
    case TextureType::uniform_color: {
      return uniform_color.get_color(uv);
    }
    case TextureType::checkboard: {
      return checkboard.get_color(uv);
    }
    case TextureType::bubble: {
      return bubble.get_color(pos);
    }
    default:
      return Color{0.0f, 0.0f, 0.0f};
    }
  }
};
