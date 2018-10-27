#include "assert_cuda.hpp"
#include "geom.h"
#include "interop_window.hpp"
#include "kernel.cuh"
#include "light.h"
#include "object.h"
#include <cuda_gl_interop.h>

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
  texture.checkboard.color1 = Color{1.0f, 1.0f, 1.0f};
  texture.checkboard.color2 = Color{0.0f, 0.0f, 0.0f};
  texture.checkboard.n_subdivision = 10.0f;
  texture.diffusion_factor = 0.7f;
  texture.ambiant_factor = 0.3f;
  Vec3f min_pos = Vec3f{-31.0f, -31.0f, -31.0f};
  Vec3f max_pos = Vec3f{31.0f, 31.0f, 31.0f};
  object.texture = texture;
  object.type = ObjectType::box;
  object.box = Box(min_pos, max_pos);
  objects.push_back(object);
}

void update_camera(Camera &camera, const InteropWindow &win, real time) {
  real speed = 1.0; // in unit per second
  real d = speed * time;
  glfwPollEvents();
  if (glfwGetKey(win.window.get(), GLFW_KEY_W) == GLFW_PRESS) camera.move_front(d);
  if (glfwGetKey(win.window.get(), GLFW_KEY_S) == GLFW_PRESS) camera.move_front(-d);
  if (glfwGetKey(win.window.get(), GLFW_KEY_A) == GLFW_PRESS) camera.move_lat(-d);
  if (glfwGetKey(win.window.get(), GLFW_KEY_D) == GLFW_PRESS) camera.move_lat(d);
  if (glfwGetKey(win.window.get(), GLFW_KEY_SPACE) == GLFW_PRESS) camera.move_up(d);
  if (glfwGetKey(win.window.get(), GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    camera.move_up(-d);
}

int main(int argc, char *argv[]) {

  std::vector<Object> objects;
  add_scene_box(objects);
  for (float i = 0.f; i < 19.99f; i += 20.0f) {
    Vec3f min_pos{i - 10.0f - 1.0f, i - 10.0f - 1.0f, -10.0f - 1.0f};
    Vec3f max_pos{i - 10.0f + 1.0f, i - 10.0f + 1.0f, -10.0f + 1.0f};
    Color color{1.0f, 1.0f, 1.0f};
    Object object;
    Texture texture;
    texture.type = TextureType::uniform_color;
    texture.uniform_color.color = color;
    texture.diffusion_factor = 0.1f;
    texture.ambiant_factor = 0.1f;
    texture.refract_factor = 0.8f;
    texture.refract_index = 1.33f;
    object.texture = texture;
    object.type = ObjectType::box;
    object.box = Box{min_pos, max_pos};
    objects.push_back(object);
  }

  Object object;
  Vec3f pos{10.0f, 10.0f, 10.0f};
  object.texture.type = TextureType::checkboard;
  object.texture.ambiant_factor = 0.4f;
  object.texture.diffusion_factor = 0.6f;
  object.texture.checkboard.color1 = Color{1.0f, 1.0f, 1.0f};
  object.texture.checkboard.color2 = Color{0.0f, 0.0f, 0.0f};
  object.texture.checkboard.n_subdivision = 5.0f;
  object.type = ObjectType::sphere;
  object.sphere = Sphere{pos, 1.0f};
  objects.push_back(object);

  Object *d_objects = nullptr;
  cuda(Malloc(&d_objects, sizeof(Object) * objects.size()));
  cuda(Memcpy(d_objects, objects.data(), sizeof(Object) * objects.size(),
              cudaMemcpyHostToDevice));

  PointLight light{Vec3f{-30.0f, 0.0f, 0.0f}, Color{1.0f, 1.0f, 1.0f}};
  AmbiantLight ambiant_light{1.0f, 1.0f, 1.0f};

  Scene scene{d_objects, (unsigned)objects.size(), light, ambiant_light};

  unsigned init_width = X_BASE_SIZE;
  unsigned init_height = Y_BASE_SIZE;

  Vec3f camera_pos{0.0f, 0.0f, 20.0f};
  Vec3f camera_dir{0, 0, -1};
  Vec3f camera_up{0, 1, 0};

  Camera camera(camera_pos, camera_dir, camera_up, 51.52f * M_PI / 180.0f,
                init_width, init_height);

  camera.gamma = 1.5;

  InteropWindow interop_window(init_width, init_height);

  interop_window.cursor_callback = [&camera](GLFWwindow *, double xupd,
                                             double yupd) {
    camera.rotate_lat(xupd * 0.005);
    camera.rotate_up(-yupd * 0.005);
  };

  // Main loop
  while (!glfwWindowShouldClose(interop_window.window.get())) {
    show_fps_and_window_size(interop_window.window.get());

    // Execute the CUDA code
    std::tie(camera.screen_width, camera.screen_height) =
        interop_window.interop_data.get_size();

    kernel_launcher(interop_window.interop_data.get_current_cuda_array(), scene,
                    camera);

    // Get events
    update_camera(camera,interop_window,0.1);
    // update physics, simulation, ...

    cudaDeviceSynchronize();

    // Switch buffers
    interop_window.interop_data.blit_buffer();
    interop_window.interop_data.change_buffer();
    glfwSwapBuffers(interop_window.window.get());
  }

  exit(EXIT_SUCCESS);
}
