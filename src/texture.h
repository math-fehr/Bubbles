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
  real scale;
  HD Color get_color(Vec3f pos) const {
    real r = fractal_perlin(pos * scale, 5, 2.0, 0.3);
    return hsv2rgb(Vec3f{r, 1.0, 1.0});
  }
};

struct WoodTexture {
  HD Color get_color(Vec3f pos) const {
    pos.z *= 10;
    real r = fractal_perlin(pos, 20, 2.0, 0.8);
    return Color{0.58, 0.37, 0.13} * r;
  }
};

struct Factors {
  real opacity;
  real ambiant;
  real diffuse;
  real specular;
  real shininess;
  real reflect;
  real refract;
  real index;

public:
  static Factors opaque(real diffuse, real specular = 0.0, real shininess = 1.0) {
    Factors factor;
    factor.opacity = 1;
    factor.ambiant = 1 - diffuse;
    factor.diffuse = diffuse;
    factor.specular = specular;
    factor.shininess = shininess;
    factor.reflect = 0;
    factor.refract = 0;
    factor.index = 1;
    return factor;
  }
  static Factors full(real diffuse, real specular, real shininess, real reflexion, real refraction,
                      real index) {
    Factors factor;
    factor.opacity = 1 - reflexion - refraction;
    factor.ambiant = 1 - diffuse;
    factor.diffuse = diffuse;
    factor.specular = specular;
    factor.shininess = shininess;
    factor.reflect = reflexion;
    factor.refract = refraction;
    factor.index = index;
    return factor;
  }
};

enum class TextureType { uniform_color, checkboard, bubble, wood };

struct Texture {
  TextureType type;
  Factors factors;

  union {
    UniformColor uniform_color;
    CheckBoard checkboard;
    BubbleTexture bubble;
    WoodTexture wood;
  };
  Texture() = default;

  Texture(UniformColor uc)
      : type(TextureType::uniform_color), uniform_color(uc) {}
  Texture(CheckBoard cb) : type(TextureType::checkboard), checkboard(cb) {}
  Texture(BubbleTexture b) : type(TextureType::bubble), bubble(b) {}
  Texture(WoodTexture w) : type(TextureType::wood), wood(w) {}
  Texture &set(Factors nfactors) {
    factors = nfactors;
    return *this;
  }

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
    case TextureType::wood: {
      return wood.get_color(pos);
    }
    default:
      return Color{0.0f, 0.0f, 0.0f};
    }
  }
};
