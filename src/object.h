#pragma once

#include "geom.h"
#include "texture.h"
#include <stdio.h>

struct IntersectionData {
  Vec3f pos;
  Vec3f normal;
  Vec2f uv;
};

struct Sphere {
  Vec3f center;
  real radius2;
  Sphere() = default;
  HD real sdf(Vec3f pos) const {
    return (pos - center).norm() - sqrtf(radius2);
  }
  HD Sphere(Vec3f center, real radius)
      : center(center), radius2(radius * radius) {}
  HD real inter(const Rayf &ray) const {
    real pi = ray.projindex(center);
    Vec3f pn = ray(pi);
    real n2 = (center - pn).norm2();
    if (n2 > radius2) {
      return -1;
    } else {
      real d = sqrtf(radius2 - n2);
      if (pi - d < 0)
        return pi + d;
      else
        return pi - d;
    }
  }
  HD Vec3f normal(Rayf ray, Vec3f pos) const {
    return (is_in(ray.orig) ? -1 : 1) * (pos - center).normalized();
  }
  HD Vec2f uv(Vec3f pos) const {
    Vec3f d = (pos - center).normalized();
    return Vec2f{0.5f + atan2f(d.z, d.x) / (2.f * 3.14159f),
                 0.5f - asinf(d.y) / 3.14159f};
  }
  HD IntersectionData inter_data(const Rayf &ray, const Vec3f &pos) const {
    return IntersectionData{pos, normal(ray, pos), uv(pos)};
  }

  HD bool is_in(Vec3f pos) const { return (pos - center).norm2() < radius2; }
  HD void move(Vec3f displ) { center += displ; }
};

struct Bubble {
  Vec3f center;
  real radius;
  real radius2;
  real noise_ampl;
  Bubble() = default;
  // HD real sdf(Vec3f pos) const {
  //   return (pos - center).norm() - sqrtf(radius2);
  // }
  HD real sdf(Vec3f pos) const {
    Vec3f out = pos - center;
    real noise = perlin(out.normalized() + center) + 1.f;
    return out.norm() - radius + noise_ampl * noise;
  }
  HD Bubble(Vec3f center, real radius, real noise_ampl)
      : center(center), radius(radius), radius2(radius * radius),
        noise_ampl(noise_ampl) {}

  HD real inter(const Rayf &ray) const {
    real pi = ray.projindex(center);
    Vec3f pn = ray(pi);
    real n2 = (center - pn).norm2();
    if (n2 > radius2) {
      return -1;
    } else {
      real d = sqrtf(radius2 - n2);
      if (pi - d < 0)
        return pi + d;
      else
        return pi - d;
    }
  }
  HD Vec3f normal(Rayf ray, Vec3f pos) const {
    return (is_in(ray.orig) ? -1 : 1) *
           (pos - center +
            0.05 * Vec3f{perlin(10 * pos), perlin(10 * pos + X),
                         perlin(10 * pos + Y)})
               .normalized();
  }

  HD Vec2f uv(Vec3f pos) const {
    Vec3f d = (pos - center).normalized();
    return Vec2f{0.5f + atan2f(d.z, d.x) / (2.f * 3.14159f),
                 0.5f - asinf(d.y) / 3.14159f};
  }
  HD IntersectionData inter_data(const Rayf &ray, const Vec3f &pos) const {
    return IntersectionData{pos, normal(ray, pos), uv(pos)};
  }

  HD bool is_in(Vec3f pos) const { return (pos - center).norm2() < radius2; }
  HD void move(Vec3f displ) { center += displ; }
};

// The plane follows the equation normal_vec | point = constant
struct Plane {
  Vec3f normal_vec;
  real constant;
  Plane() = default;
  HD Plane(Vec3f normal_vec, real constant)
      : normal_vec(normal_vec.normalized()), constant(constant) {}
  HD real inter(Rayf ray) const {
    float dot = ray.dir | normal_vec;
    if (abs(dot) < 1e-6) {
      return -1;
    }
    return (constant - (ray.orig | normal_vec)) / dot;
  }

  HD Vec3f normal(Rayf ray) const {
    if ((ray.orig | normal_vec) < 0) {
      return -normal_vec;
    } else {
      return normal_vec;
    }
  }

  HD bool is_in(Vec3f pos) const { return (pos | normal_vec) < 0; }
  HD IntersectionData inter_data(const Rayf &ray, const Vec3f &pos) const {
    return IntersectionData{pos, normal(ray), Vec2f{0, 0}};
  }
};

struct Box {
  Vec3f bounds[2];
  Box() = default;
  HD Box(Vec3f a, Vec3f b) {
    Vec3f mini, maxi;
    mini.x = min(a.x, b.x);
    maxi.x = max(a.x, b.x);
    mini.y = min(a.y, b.y);
    maxi.y = max(a.y, b.y);
    mini.z = min(a.z, b.z);
    maxi.z = max(a.z, b.z);
    bounds[0] = mini;
    bounds[1] = maxi;
  }

  HD real sdf(Vec3f pos) const {
    Vec3f center = (bounds[0] + bounds[1]) / 2;
    Vec3f d = (pos - center).abs() - (bounds[1] - bounds[0]) / 2;
    return min(max(d.x, max(d.y, d.z)), 0.f) + d.clamp(0, 1.f / 0.f).norm();
  }

  HD real inter(Rayf ray) const {
    real tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = (bounds[1 - ray.sign[0]].x - ray.orig.x) * ray.inv_dir.x;
    tmax = (bounds[ray.sign[0]].x - ray.orig.x) * ray.inv_dir.x;
    tymin = (bounds[1 - ray.sign[1]].y - ray.orig.y) * ray.inv_dir.y;
    tymax = (bounds[ray.sign[1]].y - ray.orig.y) * ray.inv_dir.y;
    tzmin = (bounds[1 - ray.sign[2]].z - ray.orig.z) * ray.inv_dir.z;
    tzmax = (bounds[ray.sign[2]].z - ray.orig.z) * ray.inv_dir.z;

    if ((tmin > tymax) || (tymin > tmax)) return -1.0f;

    tmin = max(tmin, tymin);
    tmax = min(tmax, tymax);

    if ((tmin > tzmax) || (tzmin > tmax)) return -1.0f;

    tmin = max(tmin, tzmin);
    tmax = min(tmax, tzmax);

    if (tmin < 0) {
      if (tmax < 0) {
        return -1.0f;
      }
      return tmax;
    }

    return tmin;
  }

  HD Vec3f normal(Rayf ray, Vec3f pos) const {
    int is_interior = (is_in(ray.orig)) ? 1 : 0;
    real x_0 = !ray.sign[0] ? 1.0f / 0.0f : abs(pos.x - bounds[is_interior].x);
    real x_1 =
        ray.sign[0] ? 1.0f / 0.0f : abs(pos.x - bounds[1 - is_interior].x);
    real y_0 = !ray.sign[1] ? 1.0f / 0.0f : abs(pos.y - bounds[is_interior].y);
    real y_1 =
        ray.sign[1] ? 1.0f / 0.0f : abs(pos.y - bounds[1 - is_interior].y);
    real z_0 = !ray.sign[2] ? 1.0f / 0.0f : abs(pos.z - bounds[is_interior].z);
    real z_1 =
        ray.sign[2] ? 1.0f / 0.0f : abs(pos.z - bounds[1 - is_interior].z);
    real mini = min(x_0, min(x_1, min(y_0, min(y_1, min(z_0, z_1)))));
    if (x_0 == mini) {
      return Vec3f{-1.0f, 0.0f, 0.0f};
    } else if (x_1 == mini) {
      return Vec3f{1.0f, 0.0f, 0.0f};
    } else if (y_0 == mini) {
      return Vec3f{0.0f, -1.0f, 0.0f};
    } else if (y_1 == mini) {
      return Vec3f{0.0f, 1.0f, 0.0f};
    } else if (z_0 == mini) {
      return Vec3f{0.0f, 0.0f, -1.0f};
    } else {
      return Vec3f{0.0f, 0.0f, 1.0f};
    }
  }

  HD Vec2f uv(Vec3f pos) const {
    real x_uv = clamp((bounds[1].x - pos.x) / (bounds[1].x - bounds[0].x));
    real y_uv = clamp((bounds[1].y - pos.y) / (bounds[1].y - bounds[0].y));
    real z_uv = clamp((bounds[1].z - pos.z) / (bounds[1].z - bounds[0].z));
    real x_min = abs(pos.x - bounds[0].x);
    real x_max = abs(pos.x - bounds[1].x);
    real y_min = abs(pos.y - bounds[0].y);
    real y_max = abs(pos.y - bounds[1].y);
    real z_min = abs(pos.z - bounds[0].z);
    real z_max = abs(pos.z - bounds[1].z);
    real mini =
        min(x_min, min(x_max, min(y_min, min(y_max, min(z_min, z_max)))));
    if (mini == x_min || mini == x_max) {
      return Vec2f{y_uv, z_uv};
    } else if (mini == y_min || mini == y_max) {
      return Vec2f{x_uv, z_uv};
    } else {
      return Vec2f{x_uv, y_uv};
    }
  }

  HD IntersectionData inter_data(const Rayf &ray, Vec3f pos) const {
    int is_interior = (is_in(ray.orig)) ? 1 : 0;
    real x_0 = !ray.sign[0] ? 1.0f / 0.0f : abs(pos.x - bounds[is_interior].x);
    real x_1 =
        ray.sign[0] ? 1.0f / 0.0f : abs(pos.x - bounds[1 - is_interior].x);
    real y_0 = !ray.sign[1] ? 1.0f / 0.0f : abs(pos.y - bounds[is_interior].y);
    real y_1 =
        ray.sign[1] ? 1.0f / 0.0f : abs(pos.y - bounds[1 - is_interior].y);
    real z_0 = !ray.sign[2] ? 1.0f / 0.0f : abs(pos.z - bounds[is_interior].z);
    real z_1 =
        ray.sign[2] ? 1.0f / 0.0f : abs(pos.z - bounds[1 - is_interior].z);
    real mini = min(x_0, min(x_1, min(y_0, min(y_1, min(z_0, z_1)))));
    real x_uv = clamp((bounds[1].x - pos.x) / (bounds[1].x - bounds[0].x));
    real y_uv = clamp((bounds[1].y - pos.y) / (bounds[1].y - bounds[0].y));
    real z_uv = clamp((bounds[1].z - pos.z) / (bounds[1].z - bounds[0].z));
    if (x_0 == mini) {
      return IntersectionData{pos, Vec3f{-1.0f, 0.0f, 0.0f}, Vec2f{y_uv, z_uv}};
    } else if (x_1 == mini) {
      return IntersectionData{pos, Vec3f{1.0f, 0.0f, 0.0f}, Vec2f{y_uv, z_uv}};
    } else if (y_0 == mini) {
      return IntersectionData{pos, Vec3f{0.0f, -1.0f, 0.0f}, Vec2f{x_uv, z_uv}};
    } else if (y_1 == mini) {
      return IntersectionData{pos, Vec3f{0.0f, 1.0f, 0.0f}, Vec2f{x_uv, z_uv}};
    } else if (z_0 == mini) {
      return IntersectionData{pos, Vec3f{0.0f, 0.0f, -1.0f}, Vec2f{x_uv, y_uv}};
    } else {
      return IntersectionData{pos, Vec3f{0.0f, 0.0f, 1.0f}, Vec2f{x_uv, y_uv}};
    }
  }

  HD bool is_in(Vec3f pos) const { return bounds[0] < pos && pos < bounds[1]; }
};

// orthogonal box
class Boxv2 {
  Vec3f center;
  real radius2;
  Vec3f xedge, yedge, zedge;
  real xedge2, yedge2, zedge2;
  // xedge, yedge, zedge must form an orthogonal basis and represent the mid
  // length of the Box

  void fix() {
    Vec3f xunit = xedge.normalized();
    yedge -= (yedge | xunit) * xunit;
    // now yedge orthogonal to xedge.
    Vec3f yunit = yedge.normalized();
    Vec3f zunit = xunit ^ yunit;
    zedge = (zedge | zunit) * zunit;
    xedge2 = xedge.norm2();
    yedge2 = yedge.norm2();
    zedge2 = zedge.norm2();
    radius2 = xedge2 + yedge2 + zedge2;
  }

public:
  Boxv2() = default;
  Boxv2(Vec3f center, Vec3f xedge, Vec3f yedge, Vec3f zedge)
      : center(center), xedge(xedge), yedge(yedge), zedge(zedge) {
    fix();
  }

  HD real inter(Rayf ray) const {
    Vec3f outbound = ray.orig - center;
    real raypos = -(ray.dir | outbound);
    if (outbound.norm2() - raypos * raypos > radius2) return -1;
    real xdirinv = xedge2 / (xedge | ray.dir);
    real xdirsign = sign(xdirinv);
    real ydirinv = yedge2 / (yedge | ray.dir);
    real ydirsign = sign(ydirinv);
    real zdirinv = zedge2 / (zedge | ray.dir);
    real zdirsign = sign(zdirinv);
    real xorigscal = (xedge | outbound) / xedge2;
    real yorigscal = (yedge | outbound) / yedge2;
    real zorigscal = (zedge | outbound) / zedge2;

    real txmin = (-xdirsign - xorigscal) * xdirinv;
    real txmax = txmin + 2 * xdirinv * xdirsign;
    real tymin = (-ydirsign - yorigscal) * ydirinv;
    real tymax = tymin + 2 * ydirinv * ydirsign;
    real tzmin = (-zdirsign - zorigscal) * zdirinv;
    real tzmax = tzmin + 2 * zdirinv * zdirsign;

    real tmin = max(txmin, max(tymin, tzmin));
    real tmax = min(txmax, min(tymax, tzmax));

    // Comment this line to make it work again
    if (tmin > tmax) return -1;

    if (tmin < 0) {
      if (tmax < 0) {
        return -1.0f;
      }
      return tmax;
    }

    return tmin;
  }

  HD bool is_in(Vec3f point) const {
    Vec3f out = (point - center);
    return abs(out | xedge) < xedge2 and abs(out | yedge) < yedge2 and
           abs(out | zedge) < zedge2;
  }

  HD IntersectionData inter_data(const Rayf &ray, Vec3f inter_pos) const {
    real interior = is_in(ray.orig) ? 1.0 : -1.0;
    Vec3f out = inter_pos - center;
    real xdist = 1. / 0.;
    real ydist = 1. / 0.;
    real zdist = 1. / 0.;
    constexpr real coeff = .5;

    real xout = out | xedge;
    real xdirsign = sign(ray.dir | xedge);
    if (xout > xedge2 * coeff and xdirsign * interior == 1) {
      xdist = abs(xout - xedge2);
    } else if (xout < -xedge2 * coeff and xdirsign * interior == -1) {
      xdist = abs(xout + xedge2);
    }

    real yout = out | yedge;
    real ydirsign = sign(ray.dir | yedge);
    if (yout > yedge2 * coeff and ydirsign * interior == 1) {
      ydist = abs(yout - yedge2);
    } else if (yout < -yedge2 * coeff and ydirsign * interior == -1) {
      ydist = abs(yout + yedge2);
    }

    real zout = out | zedge;
    real zdirsign = sign(ray.dir | zedge);
    if (zout > zedge2 * coeff and zdirsign * interior == 1) {
      zdist = abs(zout - zedge2);
    } else if (zout < -zedge2 * coeff and zdirsign * interior == -1) {
      zdist = abs(zout + zedge2);
    }

    if (xdist < ydist and xdist < zdist) {
      return IntersectionData{
          inter_pos, -xedge.normalized() * xdirsign,
          Vec2f{0.5f + yout / yedge2 / 2, 0.5f + zout / zedge2 / 2}};
    } else if (ydist < zdist) {
      return IntersectionData{
          inter_pos, -yedge.normalized() * ydirsign,
          Vec2f{0.5f + xout / xedge2 / 2, 0.5f + zout / zedge2 / 2}};
    } else {
      return IntersectionData{
          inter_pos, -zedge.normalized() * zdirsign,
          Vec2f{0.5f + xout / xedge2 / 2, 0.5f + yout / yedge2 / 2}};
    }
  }
  HD Vec3f normal(Rayf ray, Vec3f inter_pos) const {
    return inter_data(ray, inter_pos).normal;
  }
};

struct Pipe {
  Vec3f pos;
  HD real sdf_handle(Vec3f p) const {
    if (p.x >= 0.0f && p.x < 1.0f) {
      real a = 1.0f - p.x;
      p.y += (1.5f - a) * (a * a);
    } else if (p.x > -1.f && p.x < 0.0f) {
      real a = p.x + 1.0f;
      p.y += (1.5f - a) * (a * a);
    }

    real radius = 0.1f;
    real start = -0.1f;
    real end = 1.0f;
    // bounds distance (x)
    real db = abs(p.x - (end + start) / 2.0f) - (end - start) / 2.0f;
    // radius distance (y,z)
    real dr = Vec2f{(1.f + 2.5f * atan(abs(p.x))) * p.y, p.z}.norm() - radius;
    dr *= 0.37f;
    return min(max(db, dr), 0.0f) +
           Vec2f{max(db, 0.f), max(dr, 0.f)}.norm() * 0.75f;
  }

  HD real sdf_tube_ball(Vec3f p) const {
    Vec3f center{-0.1f, -0.425f, 0.0f};
    real radius = 0.2;
    return (p - center).norm() - radius;
  }

  HD real sdf_tube_cylinder(Vec3f p) const {
    Vec3f center{-0.1f, -0.425f, 0.0f};
    p = p - center;
    real start = 0.f;
    real end = 0.3f;
    real radius = 0.2f;
    // bounds distance (y)
    real db = abs(p.y - (end + start) / 2.0f) - (end - start) / 2.0f;
    // radius distance (x,z)
    real dr = Vec2f{p.x, p.z}.norm() - radius;
    return min(max(db, dr), 0.0f) + Vec2f{max(db, 0.f), max(dr, 0.f)}.norm();
  }

  HD real sdf_tube_hole(Vec3f p) const {
    Vec3f center{-0.1f, -0.425f, 0.0f};
    p = p - center;
    real start = -0.1f;
    real end = 0.4f;
    real radius = 0.12f;
    // bounds distance (y)
    real db = abs(p.y - (end + start) / 2.0f) - (end - start) / 2.0f;
    // radius distance (x,z)
    real dr = Vec2f{p.x, p.z}.norm() - radius;
    return min(max(db, dr), 0.0f) + Vec2f{max(db, 0.f), max(dr, 0.f)}.norm();
  }

  HD real sdf(Vec3f p) const {
    p -= pos;
    real d = min(sdf_tube_ball(p), sdf_tube_cylinder(p));
    d = max(d, -sdf_tube_hole(p));
    real d2 = sdf_handle(p);
    real k = 0.06f;
    float h = clamp(0.5 + 0.5 * (d2 - d) / k, 0.0, 1.0);
    return mix(d2, d, h) - k * h * (1.0 - h);
  }

  HD real inter(const Rayf &ray) const {
    Box box(pos + Vec3f{-0.3, 0.1, -0.21}, pos + Vec3f{1.2, -0.7f, 0.21});
    real depth = box.inter(ray);
    if (depth < 0) {
      return depth;
    }
    for (unsigned i = 0; i < 512; ++i) {
      real dist = sdf(ray(depth));
      depth += dist;
      if (dist < 1e-4f) {
        break;
      }
    }
    if (sdf(ray(depth)) > 1e-3f) {
      return -1.0f;
    }
    return depth;
  }
  HD Vec3f normal(const Vec3f &point) const {
    float epsilon = 1e-5f;
    float x =
        sdf(point + Vec3f{epsilon, 0, 0}) - sdf(point - Vec3f{epsilon, 0, 0});
    float y =
        sdf(point + Vec3f{0, epsilon, 0}) - sdf(point - Vec3f{0, epsilon, 0});
    float z =
        sdf(point + Vec3f{0, 0, epsilon}) - sdf(point - Vec3f{0, 0, epsilon});
    return Vec3f{x, y, z}.normalized();
  }

  HD IntersectionData inter_data(const Rayf &ray, Vec3f pos) const {
    return IntersectionData{pos, normal(pos), Vec2f{0.0, 0.0}};
  }

  HD bool is_in(const Vec3f &pos) const { return sdf(pos) > 0.0f; }
};

enum class ObjectType { sphere, bubble, box, plane, box2, pipe };

struct Object {
  Texture texture;
  Vec3f speed;
  ObjectType type;
  union {
    Sphere sphere;
    Bubble bubble;
    Plane plane;
    Box box;
    Boxv2 box2;
    Pipe pipe;
  };

  Object() = default;

  Object(Sphere s) : type(ObjectType::sphere), sphere(s) {}
  Object(Bubble b) : type(ObjectType::bubble), bubble(b) {}
  Object(Plane p) : type(ObjectType::plane), plane(p) {}
  Object(Box b) : type(ObjectType::box), box(b) {}
  Object(Boxv2 b) : type(ObjectType::box2), box2(b) {}
  Object(Pipe p) : type(ObjectType::pipe), pipe(p) {}

  Object &set(const Texture &tex) {
    texture = tex;
    return *this;
  }

  HD real sdf(Vec3f pos) const {
    switch (type) {
    case ObjectType::sphere:
      return sphere.sdf(pos);
    case ObjectType::bubble:
      return bubble.sdf(pos);
    case ObjectType::box:
      return box.sdf(pos);
    default:
      return 1.f / 0.f;
    }
  }

  HD real inter(Rayf ray) const {
    switch (type) {
    case ObjectType::sphere:
      return sphere.inter(ray);
    case ObjectType::bubble:
      return bubble.inter(ray);
    case ObjectType::plane:
      return plane.inter(ray);
    case ObjectType::box:
      return box.inter(ray);
    case ObjectType::box2:
      return box2.inter(ray);
    case ObjectType::pipe:
      return pipe.inter(ray);
    default:
      return -1.0;
    }
  }

  HD Vec3f normal(Rayf ray, real distance) const {
    switch (type) {
    case ObjectType::sphere:
      return sphere.normal(ray, ray(distance));
    case ObjectType::bubble:
      return bubble.normal(ray, ray(distance));
    case ObjectType::plane:
      return plane.normal(ray);
    case ObjectType::box:
      return box.normal(ray, ray(distance));
    case ObjectType::box2:
      return box2.normal(ray, ray(distance));
    default:
      return {0.0f, 0.0f, 0.0f};
    }
  }

  HD Vec2f uv(Rayf ray, real distance) const {
    switch (type) {
    case ObjectType::sphere:
      return sphere.uv(ray(distance));
    case ObjectType::bubble:
      return bubble.uv(ray(distance));
    case ObjectType::box:
      return box.uv(ray(distance));
    default:
      return {0.0f, 0.0f};
    }
  }

  HD IntersectionData inter_data(const Rayf &ray, real distance) const {
    switch (type) {
    case ObjectType::sphere:
      return sphere.inter_data(ray, ray(distance));
    case ObjectType::bubble:
      return bubble.inter_data(ray, ray(distance));
    case ObjectType::plane:
      return plane.inter_data(ray, ray(distance));
    case ObjectType::box:
      return box.inter_data(ray, ray(distance));
    case ObjectType::box2:
      return box2.inter_data(ray, ray(distance));
    case ObjectType::pipe:
      return pipe.inter_data(ray, ray(distance));
    default:
      return {};
    }
  }

  HD bool is_in(Vec3f point) const {
    switch (type) {
    case ObjectType::box:
      return box.is_in(point);
    case ObjectType::plane:
      return plane.is_in(point);
    case ObjectType::box2:
      return box2.is_in(point);
    case ObjectType::sphere:
      return sphere.is_in(point);
    case ObjectType::bubble:
      return bubble.is_in(point);
    case ObjectType::pipe:
      return pipe.is_in(point);
    default:
      return false;
    }
  }

  HD void move(Vec3f displ) {
    switch (type) {
    case ObjectType::sphere:
      return sphere.move(displ);
    case ObjectType::bubble:
      return bubble.move(displ);
    default:
      return;
    }
  }

  HD Vec3f pos() {
    switch (type) {
    case ObjectType::sphere:
      return sphere.center;
    case ObjectType::bubble:
      return bubble.center;
    case ObjectType::box:
      return (box.bounds[0] + box.bounds[1]) / 2;
    default:
      return Vec3f{0, 0, 0};
    }
  }
};
