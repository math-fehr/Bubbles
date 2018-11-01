#include "assert_cuda.hpp"
#include "geom.h"
#include "interop_window.hpp"
#include "kernel.cuh"
#include "light.h"
#include "object.h"
#include <cuda_gl_interop.h>
#include <random>

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

void add_scene_box(std::vector<Object> &objects) {
  Object object;
  Texture texture;
  texture.type = TextureType::checkboard;
  texture.checkboard.color1 = white;
  texture.checkboard.color2 = white * 0.5;
  texture.checkboard.n_subdivision = 10.0f;
  texture.factors = Factors::opaque(0.7f);
  Vec3f min_pos = Vec3f{-31.0f, -31.0f, -31.0f};
  Vec3f max_pos = Vec3f{31.0f, 31.0f, 31.0f};
  object.texture = texture;
  object.type = ObjectType::box;
  object.box = Box(min_pos, max_pos);
  objects.push_back(object);
}

void add_mega_scene_box(std::vector<Object> &objects) {
  Object object;
  Texture texture;
  texture.type = TextureType::uniform_color;
  texture.uniform_color.color = black;
  texture.factors = Factors::opaque(0.0f);
  Vec3f min_pos = Vec3f{-3100000.0f, -3100000.0f, -3100000.0f};
  Vec3f max_pos = Vec3f{3100000.0f, 3100000.0f, 3100000.0f};
  object.texture = texture;
  object.type = ObjectType::box;
  object.box = Box(min_pos, max_pos);
  objects.push_back(object);
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

Vec3f gradient(const function<real(Vec3f)> &f, Vec3f pos) {
  real eps = 1e-4;
  return Vec3f{(f(pos + eps * X) - f(pos - eps * X)) / (2 * eps),
               (f(pos + eps * Y) - f(pos - eps * Y)) / (2 * eps),
               (f(pos + eps * Z) - f(pos - eps * Z)) / (2 * eps)};
}

int main(int argc, char *argv[]) {
  std::vector<Object> objects;

  add_mega_scene_box(objects);
  add_scene_box(objects);
  // The bubbly bubble

  objects.push_back(Object(Sphere{Vec3f{-30, 1, 0}, 0.1})
                        .set(Texture(CheckBoard{white, black, 5.0f})
                                 .set(Factors::opaque(0.6f))));

  objects.push_back(Object(Sphere{Vec3f{-30, -1, 0}, 0.1})
                        .set(Texture(CheckBoard{white, black, 5.0f})
                                 .set(Factors::opaque(0.6f))));

  objects.push_back(
      Object(Box{Vec3f{-10, -10, -10}, Vec3f{-5, -5, -5}})
          .set(Texture(UniformColor{red}).set(Factors::opaque(0.7f))));

  objects.push_back(
      Object(Pipe{}).set(Texture(UniformColor{red}).set(Factors::opaque(0.8))));

  int bubbles_start = objects.size();

  for (int i = 0; i < MAX_NUM_BUBBLES; ++i) {
    objects.push_back(
        Object(Sphere{Vec3f{2 * (i % 20) - 20.f, 2.f * (i / 20), 10}, 1})
            .set(Texture(BubbleTexture{})
                     .set(Factors::full(0.7, 0.1, 0.8, 1.005))));

    objects.back().speed = Vec3f{0, 0, -1};
  }

  Object *d_objects = nullptr;
  cuda(Malloc(&d_objects, sizeof(Object) * (bubbles_start + MAX_NUM_BUBBLES)));
  cuda(Memcpy(d_objects, objects.data(), sizeof(Object) * objects.size(),
              cudaMemcpyHostToDevice));

  PointLight light{Vec3f{-30.0f, 0.0f, 0.0f}, Color{1.0f, 1.0f, 1.0f}};
  AmbiantLight ambiant_light{1.0f, 1.0f, 1.0f};

  Scene scene{d_objects, (unsigned)objects.size(), light, ambiant_light};

  unsigned init_width = X_BASE_SIZE;
  unsigned init_height = Y_BASE_SIZE;

  Vec3f camera_pos{0.0f, 0.0f, 0.0f};
  Vec3f camera_dir{1, 1, 1};
  Vec3f camera_up{0, 1, 0};

  Camera camera(camera_pos, camera_dir, camera_up, 51.52f * M_PI / 180.0f,
                init_width, init_height);

  camera.gamma = 1.5;

  InteropWindow interop_window(init_width, init_height);

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

    // update physics, simulation, ...

    random_device dev;
    normal_distribution<real> dist(0, 0.5);
    real k = 0.1;
    real bubble_mass = 1;

    for (int i = bubbles_start; i < objects.size(); ++i) {

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
              return exp(sqrtf(objects[i].sphere.radius2) - objects[j].sdf(p));
            },
            objects[i].pos());
      }
      grad += -gradient(
          [&](Vec3f p) { return exp(2 - (p - camera.get_pos()).norm()); },
          objects[i].pos());

      Vec3f force =
          -k * objects[i].speed + Vec3f{dist(dev), dist(dev), dist(dev)} + grad;

      Vec3f accel = force / bubble_mass;

      objects[i].speed += accel * (time - lasttime);
    }

    for (int i = bubbles_start; i < objects.size(); ++i) {
      objects[i].move(objects[i].speed * (time - lasttime));
    }

    lasttime = time;

    // Wait for GPU to finish rendering
    cudaDeviceSynchronize();
    cuda(Memcpy(d_objects, objects.data(), sizeof(Object) * objects.size(),
                cudaMemcpyHostToDevice));

    // Switch buffers
    interop_window.interop_data.blit_buffer();
    interop_window.interop_data.change_buffer();
    glfwSwapBuffers(interop_window.window.get());
  }

  return EXIT_SUCCESS;
}
