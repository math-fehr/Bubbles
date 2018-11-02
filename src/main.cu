#include "assert_cuda.hpp"
#include "geom.h"
#include "interop_window.hpp"
#include "kernel.cuh"
#include "light.h"
#include "object.h"
#include <cuda_gl_interop.h>
#include <random>

bool create_bubbles = false;
bool move_bubbles = true;
bool move_light = false;
real speed_bubble_grow = 0.1;

using namespace std;

static void show_fps_and_window_size(GLFWwindow *window) {
  // fps counter in static variables
  static double previous_time = 0.0;
  static int frame_count = 0;

  const double current_time = glfwGetTime();
  const double elapsed = current_time - previous_time;

  if (elapsed > 0.5) {
    previous_time = current_time;

    const double fps = (double)frame_count / elapsed;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    char tmp[64];
    sprintf(tmp, "(%u x %u) - FPS: %.2f", width, height, fps);

    glfwSetWindowTitle(window, tmp);

    frame_count = 0;
  }

  frame_count++;
}

void add_skybox(std::vector<Object> *objects) {
  objects->push_back(
      Object(Box(Vec3f{-1e20, -1e20, -1e20}, Vec3f{1e20, 1e20, 1e20}))
          .set(Texture(UniformColor{black}).set(Factors::opaque(0.0f))));
}

void add_scene_box(std::vector<Object> *objects) {
  objects->push_back(Object(Box(Vec3f{-8, -9, -8}, Vec3f{8, -2, 8}))
                         .set(Texture(CheckBoard{white, white * 0.5, 10})
                                  .set(Factors::opaque(0.7f))));
}

void load_final_scene(Scene *scene, std::vector<Object> *objects,
                      Camera *camera) {
  objects->clear();
  // skybox
  add_skybox(objects);

  // scene box
  add_scene_box(objects);

  // The pippy pipe
  objects->push_back(
      Object(Pipe{Vec3f{0.f, -7.f, 0.f}, 8})
          .set(Texture(WoodTexture{}).set(Factors::opaque(0.8))));

  // The scene
  scene->light = PointLight{Vec3f{-7.0f, -5.0f, 0.0f}, Color{1.0f, 1.0f, 1.0f}};
  scene->ambiant_light = AmbiantLight{1.0f, 1.0f, 1.0f};
  scene->n_objects = objects->size();

  // The camera
  camera->set_pos(Vec3f{0, -3.5, 7});
  camera->set_dir(Vec3f{0, -0.3, -1});
}

void load_primitives_scene(Scene *scene, std::vector<Object> *objects,
                           Camera *camera) {
  objects->clear();

  // skybox
  add_skybox(objects);

  // scene_box
  add_scene_box(objects);

  // sphere
  objects->push_back(
      Object(Sphere(Vec3f{-5.5f, -7.f, 0.f}, 0.4f))
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));

  // bubble
  objects->push_back(
      Object(Bubble(Vec3f{-3.5f, -7.f, 0.f}, 0.4f, 5.0f))
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));

  // box
  objects->push_back(
      Object(Box(Vec3f{-1.0f, -7.5f, -0.5f}, Vec3f{-2.f, -6.5f, 0.5f}))
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));

  // boxv2
  objects->push_back(
      Object(Boxv2(Vec3f{0.5f, -7, -0.5f}, (X + Y) * 0.3, (Y + Z) * 0.3,
                   (Z + X) * 0.3))
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));

  // pipe
  objects->push_back(
      Object(Pipe{Vec3f{2.5f, -6.5f, 0.f}, 8})
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));

  // The scene
  scene->light = PointLight{Vec3f{-7.0f, -5.0f, 0.0f}, Color{1.0f, 1.0f, 1.0f}};
  scene->ambiant_light = AmbiantLight{1.0f, 1.0f, 1.0f};
  scene->n_objects = objects->size();

  // The camera
  camera->set_pos(Vec3f{0, -7.5, 7});
  camera->set_dir(Vec3f{0, 0.2, -1});
}

void load_phong_scene(Scene *scene, std::vector<Object> *objects,
                      Camera *camera) {
  objects->clear();

  // skybox
  add_skybox(objects);

  // scene_box
  add_scene_box(objects);

  // box ambiant
  objects->push_back(
      Object(Box(Vec3f{-5.9f, -7.4f, -0.4f}, Vec3f{-5.1f, -6.6f, 0.4f}))
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.0f))));

  // box diffuse
  objects->push_back(
      Object(Box(Vec3f{-3.9f, -7.4f, -0.4f}, Vec3f{-3.1f, -6.6f, 0.4f}))
          .set(Texture(UniformColor{red}).set(Factors::opaque(1.0f))));

  // box specular
  objects->push_back(
      Object(Box(Vec3f{-1.9f, -7.4f, -0.4f}, Vec3f{-1.1f, -6.6f, 0.4f}))
          .set(Texture(UniformColor{red}).set(Factors::opaque(1.0f, 0.5, 5))));
  objects->back().texture.factors.diffuse = 0.0f;

  // box mat
  objects->push_back(
      Object(Box(Vec3f{0.1f, -7.4f, -0.4f}, Vec3f{0.9f, -6.6f, 0.4f}))
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));

  // box metal
  objects->push_back(
      Object(Box(Vec3f{2.1f, -7.4f, -0.4f}, Vec3f{2.9f, -6.6f, 0.4f}))
          .set(Texture(UniformColor{red})
                   .set(Factors::opaque(0.7f, 0.5, 1000))));

  // sphere ambiant
  objects->push_back(
      Object(Sphere(Vec3f{-5.5f, -4.4f, -0.4f}, 0.4))
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.0f))));

  // sphere diffuse
  objects->push_back(
      Object(Sphere(Vec3f{-3.5f, -4.4f, -0.4f}, 0.4))
          .set(Texture(UniformColor{red}).set(Factors::opaque(1.0f))));

  // sphere specular
  objects->push_back(
      Object(Sphere(Vec3f{-1.5f, -4.4f, -0.4f}, 0.4))
          .set(Texture(UniformColor{red}).set(Factors::opaque(1.0f, 0.5, 5))));
  objects->back().texture.factors.diffuse = 0.0f;

  // sphere mat
  objects->push_back(
      Object(Sphere(Vec3f{0.5f, -4.4f, -0.4f}, 0.4))
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));

  // sphere metal
  objects->push_back(
      Object(Sphere(Vec3f{2.5f, -4.4f, -0.4f}, 0.4))
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 50))));

  // The scene
  scene->light = PointLight{Vec3f{-7.0f, -5.0f, 0.0f}, Color{1.0f, 1.0f, 1.0f}};
  scene->ambiant_light = AmbiantLight{1.0f, 1.0f, 1.0f};
  scene->n_objects = objects->size();

  // The camera
  camera->set_pos(Vec3f{0, -7.5, 7});
  camera->set_dir(Vec3f{0, 0.2, -1});
}

void load_many_bubbles_scene(Scene *scene, std::vector<Object> *objects,
                             Camera *camera) {
  objects->clear();

  // skybox
  add_skybox(objects);

  // scene_box
  add_scene_box(objects);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j <= i; ++j) {
      objects->push_back(
          Object(Bubble(Vec3f{-4.5f + i * 2.f, -6.4f, -4.5f + j * 2.f}, 0.5f,
                        0.1f))
              .set(Texture(BubbleTexture{5.0})
                       .set(Factors::full(0.6, 20, 500, 0.1, 0.8, 1.005))));
    }
  }

  // The scene
  scene->light = PointLight{Vec3f{-7.0f, -5.0f, 0.0f}, Color{1.0f, 1.0f, 1.0f}};
  scene->ambiant_light = AmbiantLight{1.0f, 1.0f, 1.0f};
  scene->n_objects = objects->size();

  // The camera
  camera->set_pos(Vec3f{0, -7.5, 7});
  camera->set_dir(Vec3f{0, 0.2, -1});
}

void load_bubble_scene(Scene *scene, std::vector<Object> *objects,
                       Camera *camera) {
  objects->clear();

  // skybox
  add_skybox(objects);

  // scene_box
  add_scene_box(objects);

  // simple bubble
  objects->push_back(
      Object(Sphere(Vec3f{-4.5f, -7.4f, -0.4f}, 0.5f))
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.6, 20, 500))));

  // bubble + texture
  objects->push_back(
      Object(Sphere(Vec3f{-2.5f, -7.4f, -0.4f}, 0.5f))
          .set(Texture(BubbleTexture{5.0}).set(Factors::opaque(0.6, 20, 500))));

  // bubble + texture + refractions + reflections
  objects->push_back(
      Object(Sphere(Vec3f{-0.5f, -7.4f, -0.4f}, 0.5f))
          .set(Texture(BubbleTexture{5.0})
                   .set(Factors::full(0.6, 20, 500, 0.1, 0.8, 1.005))));

  // bubble + texture + refractions + reflections + normal map
  objects->push_back(
      Object(Bubble(Vec3f{1.5f, -7.4f, -0.4f}, 0.5f, 0.1f))
          .set(Texture(BubbleTexture{5.0})
                   .set(Factors::full(0.6, 20, 500, 0.1, 0.8, 1.005))));

  // The scene
  scene->light = PointLight{Vec3f{-7.0f, -5.0f, 0.0f}, Color{1.0f, 1.0f, 1.0f}};
  scene->ambiant_light = AmbiantLight{1.0f, 1.0f, 1.0f};
  scene->n_objects = objects->size();

  // The camera
  camera->set_pos(Vec3f{0, -7.5, 7});
  camera->set_dir(Vec3f{0, 0.2, -1});
}

void load_pipe_scene(Scene *scene, std::vector<Object> *objects,
                     Camera *camera) {
  objects->clear();

  // skybox
  add_skybox(objects);

  // scene_box
  add_scene_box(objects);

  // pipe 1
  objects->push_back(
      Object(Pipe{Vec3f{-2.5f, -6.5f, 0.f}, 1})
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));

  // pipe 2
  objects->push_back(
      Object(Pipe{Vec3f{-0.5f, -6.5f, 0.f}, 2})
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));

  // pipe 4
  objects->push_back(
      Object(Pipe{Vec3f{1.5f, -6.5f, 0.f}, 4})
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));

  // pipe 8
  objects->push_back(
      Object(Pipe{Vec3f{3.5f, -6.5f, 0.f}, 8})
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));
  // The scene
  scene->light = PointLight{Vec3f{-7.0f, -5.0f, 0.0f}, Color{1.0f, 1.0f, 1.0f}};
  scene->ambiant_light = AmbiantLight{1.0f, 1.0f, 1.0f};
  scene->n_objects = objects->size();

  // The camera
  camera->set_pos(Vec3f{0, -7.5, 7});
  camera->set_dir(Vec3f{0, 0.2, -1});
}

void update_camera(Camera &camera, const InteropWindow &win, real time) {
  real speed = 5.0; // in unit per second
  glfwPollEvents();
  if (glfwGetKey(win.window.get(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    speed = 10.0;
  real d = speed * time;
  if (glfwGetKey(win.window.get(), GLFW_KEY_W) == GLFW_PRESS)
    camera.move_front(d);
  if (glfwGetKey(win.window.get(), GLFW_KEY_S) == GLFW_PRESS)
    camera.move_front(-d);
  if (glfwGetKey(win.window.get(), GLFW_KEY_A) == GLFW_PRESS)
    camera.move_lat(-d);
  if (glfwGetKey(win.window.get(), GLFW_KEY_D) == GLFW_PRESS)
    camera.move_lat(d);
  if (glfwGetKey(win.window.get(), GLFW_KEY_SPACE) == GLFW_PRESS)
    camera.move_up(d);
  if (glfwGetKey(win.window.get(), GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    camera.move_up(-d);
}

void update_scene(const InteropWindow &win, Scene *scene,
                  vector<Object> *objects, Camera *camera) {
  if (glfwGetKey(win.window.get(), GLFW_KEY_T) == GLFW_PRESS)
    load_final_scene(scene, objects, camera);
  if (glfwGetKey(win.window.get(), GLFW_KEY_Y) == GLFW_PRESS)
    load_primitives_scene(scene, objects, camera);
  if (glfwGetKey(win.window.get(), GLFW_KEY_U) == GLFW_PRESS)
    load_phong_scene(scene, objects, camera);
  if (glfwGetKey(win.window.get(), GLFW_KEY_I) == GLFW_PRESS)
    load_bubble_scene(scene, objects, camera);
  if (glfwGetKey(win.window.get(), GLFW_KEY_O) == GLFW_PRESS)
    load_many_bubbles_scene(scene, objects, camera);
  if (glfwGetKey(win.window.get(), GLFW_KEY_P) == GLFW_PRESS)
    load_pipe_scene(scene, objects, camera);
}

void update_options(const InteropWindow &win, Camera *camera) {
  if (glfwGetKey(win.window.get(), GLFW_KEY_G) == GLFW_PRESS)
    create_bubbles = true;
  if (glfwGetKey(win.window.get(), GLFW_KEY_V) == GLFW_PRESS)
    create_bubbles = false;
  if (glfwGetKey(win.window.get(), GLFW_KEY_H) == GLFW_PRESS)
    move_bubbles = true;
  if (glfwGetKey(win.window.get(), GLFW_KEY_B) == GLFW_PRESS)
    move_bubbles = false;
  if (glfwGetKey(win.window.get(), GLFW_KEY_J) == GLFW_PRESS) move_light = true;
  if (glfwGetKey(win.window.get(), GLFW_KEY_N) == GLFW_PRESS)
    move_light = false;
  if (glfwGetKey(win.window.get(), GLFW_KEY_K) == GLFW_PRESS)
    speed_bubble_grow = clamp(speed_bubble_grow + 0.1f, 0.1, 2);
  if (glfwGetKey(win.window.get(), GLFW_KEY_M) == GLFW_PRESS)
    speed_bubble_grow = clamp(speed_bubble_grow - 0.1f, 0.1, 2);
  if (glfwGetKey(win.window.get(), GLFW_KEY_L) == GLFW_PRESS)
    camera->gamma += 0.1f;
  if (glfwGetKey(win.window.get(), GLFW_KEY_COMMA) == GLFW_PRESS)
    camera->gamma -= 0.1f;
}

void update_light(const InteropWindow &win, const Camera &camera,
                  Scene *scene) {
  if (glfwGetKey(win.window.get(), GLFW_KEY_X) == GLFW_PRESS)
    scene->light.center = camera.get_pos();
}

void update_bullet(const InteropWindow &win, const Camera &camera,
                   vector<Object> *objects) {
  static int old_right = GLFW_RELEASE;
  if (old_right == GLFW_RELEASE &&
      glfwGetMouseButton(win.window.get(), GLFW_MOUSE_BUTTON_RIGHT) ==
          GLFW_PRESS) {
    objects->push_back(
        Object(Sphere(camera.get_pos(), 0.01f))
            .set(
                Texture(UniformColor{red}).set(Factors::opaque(0.7f, 0.5, 5))));
    objects->back().speed =
        camera.get_ray(camera.screen_width * 0.5, camera.screen_height * 0.5)
            .dir *
        20;
  }
  old_right = glfwGetMouseButton(win.window.get(), GLFW_MOUSE_BUTTON_RIGHT);

  static int old_left = GLFW_RELEASE;
  if (old_left == GLFW_RELEASE &&
      glfwGetMouseButton(win.window.get(), GLFW_MOUSE_BUTTON_LEFT) ==
          GLFW_PRESS) {
    objects->push_back(
                       Object(Bubble(camera.get_pos(), 0.2, 0.1f))
                       .set(Texture(BubbleTexture{5.0})
                            .set(Factors::full(0.6, 20, 500, 0.1, 0.8, 1.005))));

  }
  old_left = glfwGetMouseButton(win.window.get(), GLFW_MOUSE_BUTTON_LEFT);
}

Vec3f gradient(const function<real(Vec3f)> &f, Vec3f pos) {
  real eps = 1e-4;
  return Vec3f{(f(pos + eps * X) - f(pos - eps * X)) / (2 * eps),
               (f(pos + eps * Y) - f(pos - eps * Y)) / (2 * eps),
               (f(pos + eps * Z) - f(pos - eps * Z)) / (2 * eps)};
}

int main(int argc, char *argv[]) {
  Scene scene{nullptr, 0, {}, {}};
  cuda(Malloc(&scene.objects, sizeof(Object) * MAX_OBJECTS));

  InteropWindow interop_window(X_BASE_SIZE, Y_BASE_SIZE);

  std::vector<Object> objects;

  // Camera
  Vec3f camera_pos{3.0f, -10.0f, 0.0f};
  Vec3f camera_dir{0, 0, 1};
  Vec3f camera_up{0, 1, 0};
  Camera camera(camera_pos, camera_dir, camera_up, 51.52f * M_PI / 180.0f,
                X_BASE_SIZE, Y_BASE_SIZE);

  load_final_scene(&scene, &objects, &camera);

  cuda(Memcpy(scene.objects, objects.data(), sizeof(Object) * objects.size(),
              cudaMemcpyHostToDevice));

  interop_window.cursor_callback = [&camera](GLFWwindow *, double xupd,
                                             double yupd) {
    camera.rotate_lat(xupd * 0.0005);
    camera.rotate_up(-yupd * 0.0005);
  };

  double time = glfwGetTime();
  double lasttime = glfwGetTime();

  // Main loop
  while (!glfwWindowShouldClose(interop_window.window.get())) {
    show_fps_and_window_size(interop_window.window.get());

    // Execute the CUDA code
    std::tie(camera.screen_width, camera.screen_height) =
        interop_window.interop_data.get_size();

    kernel_launcher(interop_window.interop_data.get_current_cuda_array(), scene,
                    camera);

    // Event management
    time = glfwGetTime();
    update_camera(camera, interop_window, time - lasttime);
    update_scene(interop_window, &scene, &objects, &camera);
    update_options(interop_window, &camera);
    update_light(interop_window, camera, &scene);
    update_bullet(interop_window, camera, &objects);
    real delta_time = time - lasttime;

    // update physics, simulation, ...

    // update light placing for final scene
    if (move_light) {
      scene.light.center =
          Vec3f{cosf(time * 0.2f) * 7.f, -5.0f, sinf(time * 0.2f) * 7.f};
    }

    // update bullet movement
    for (int i = 0; i < objects.size(); ++i) {
      if (objects[i].type != ObjectType::sphere) {
        continue;
      }
      objects[i].sphere.move(objects[i].speed * delta_time);
      if (abs(objects[i].sphere.center.x) > 30 ||
          abs(objects[i].sphere.center.y) > 30 ||
          abs(objects[i].sphere.center.z) > 30) {
        objects.erase(objects.begin() + i);
        --i;
        continue;
      }
      Object temp_object = objects[i];

      for (int j = 0; j < objects.size(); j++) {
        if (objects[j].type == ObjectType::bubble) {
          real distance2 =
              (objects[i].sphere.center - objects[j].bubble.center).norm2();
          real radius =
              sqrtf(objects[i].sphere.radius2) + objects[j].bubble.radius;
          real radius2 = radius * radius;
          if (distance2 < radius2) {
            objects.erase(objects.begin() + j);
            j--;
          }
        }
      }
    }

    bool has_pipe = false;
    Pipe pipe;
    for (int j = 0; j < objects.size(); ++j) {
      if (objects[j].type == ObjectType::pipe) {
        pipe = objects[j].pipe;
        has_pipe = true;
        break;
      }
    }

    // Update growing bubble
    if (move_bubbles && has_pipe) {
      for (int i = 0; i < objects.size(); ++i) {
        if (objects[i].type != ObjectType::future_bubble) {
          continue;
        }

        FutureBubble &bubble = objects[i].future_bubble;
        real radius = bubble.compute_radius(pipe);
        if (bubble.touch_hole) {
          // If we finish growing
          if (radius > bubble.stop_radius && radius > bubble.radius) {
            Bubble new_bubble = bubble.transform();
            objects[i].type = ObjectType::bubble;
            objects[i].bubble = new_bubble;
            objects[i].speed = Vec3f{0, 1, 0};
          } else {
            bubble.center.y += delta_time * speed_bubble_grow;
            bubble.set_radius(radius);
          }
        } else {
          bubble.center.y += delta_time * speed_bubble_grow;
          bubble.limit_plane += delta_time * speed_bubble_grow;
          if (radius < bubble.radius) {
            bubble.touch_hole = true;
            bubble.set_radius(radius);
            bubble.set_limit_plane(pipe);
          }
        }
      }
    }

    if (create_bubbles && has_pipe) {
      // Maybe add a new bubble
      bool has_future_bubble = false;
      for (int i = 0; i < objects.size(); ++i) {
        if (objects[i].type == ObjectType::future_bubble) {
          has_future_bubble = true;
          break;
        }
      }
      if (objects.size() < (MAX_OBJECTS - 1) && has_future_bubble == false) {
        objects.push_back(
            Object(FutureBubble(pipe))
                .set(Texture(BubbleTexture{5.0})
                         .set(Factors::full(0.6, 20, 500, 0.1, 0.8, 1.005))));
        scene.n_objects++;
      }
    }

    if (move_bubbles) {
      // Update bubbles
      random_device dev;
      normal_distribution<real> dist(0, 0.5);
      real k = 0.1;
      real bubble_mass = 1;

      for (int i = 0; i < objects.size(); ++i) {
        if (objects[i].type != ObjectType::bubble) {
          continue;
        }

        Vec3f grad{0, 0, 0};
        grad += -gradient(
            [&](Vec3f p) {
              return exp(sqrtf(objects[i].sphere.radius2) + objects[1].sdf(p));
            },
            objects[i].pos());
        for (int j = 2; j < objects.size(); ++j) {
          if (j == i) continue;
          grad += -gradient(
              [&](Vec3f p) {
                return exp(sqrtf(objects[i].sphere.radius2) -
                           objects[j].sdf(p));
              },
              objects[i].pos());
        }
        grad += -gradient(
            [&](Vec3f p) { return exp(2 - (p - camera.get_pos()).norm()); },
            objects[i].pos());

        Vec3f force = -k * objects[i].speed +
                      Vec3f{dist(dev), dist(dev), dist(dev)} + grad;

        Vec3f accel = force / bubble_mass;

        objects[i].speed += accel * (time - lasttime);
      }

      for (int i = 0; i < objects.size(); ++i) {
        if (objects[i].type == ObjectType::bubble) {
          objects[i].move(sqrtf(objects[i].sphere.radius2) * objects[i].speed *
                          (time - lasttime));
        }
      }
    }

    lasttime = time;

    scene.n_objects = objects.size();
    // Wait for GPU to finish rendering
    cudaDeviceSynchronize();
    cuda(Memcpy(scene.objects, objects.data(), sizeof(Object) * objects.size(),
                cudaMemcpyHostToDevice));

    // Switch buffers
    interop_window.interop_data.blit_buffer();
    interop_window.interop_data.change_buffer();
    glfwSwapBuffers(interop_window.window.get());
  }

  return EXIT_SUCCESS;
}
