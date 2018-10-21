#include "assert_cuda.hpp"
#include "interop_window.hpp"
#include "kernel.cuh"
#include <cuda_gl_interop.h>
#include "geom.h"
#include "object.h"

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

int main(int argc, char *argv[]) {

  std::vector<Object> objects;
  for(float i = 0.f; i < 9.99f; i+=0.1f) {
    Vec3f pos{i - 5.0f, i - 5.0f, -2.0f};
    float radius = 0.1f;
    Color color{1.0f, 1.0f, 1.0f};
    Object object;
    object.color = color;
    object.type = ObjectType::sphere;
    object.sphere = Sphere{pos,radius};
    objects.push_back(object);
  }

  Object* d_objects = nullptr;
  cuda(Malloc(&d_objects, sizeof(Object) * objects.size()));
  cuda(Memcpy(d_objects, objects.data(), sizeof(Object) * objects.size(), cudaMemcpyHostToDevice));

  unsigned init_width = 640;
  unsigned init_height = 480;

  Vec3f camera_pos{0.0f, 0.0f, 20.0f};
  Vec3f camera_to_world_x{1.0f, 0.0f, 0.0f};
  Vec3f camera_to_world_y{0.0f, 1.0f, 0.0f};
  Vec3f camera_to_world_z{0.0f, 0.0f, 1.0f};
  Mat3f camera_to_world{camera_to_world_x,camera_to_world_y,camera_to_world_z};

  Camera camera{camera_pos, camera_to_world, 51.52f * M_PI / 180.0f, init_width, init_height};

  InteropWindow interop_window(init_width, init_height);

  interop_window.key_callbacks.insert(
      {GLFW_KEY_UP, [&camera](GLFWwindow *, int action, int mods) {
                      camera.move_front(0.1);
       }});
  interop_window.key_callbacks.insert(
      {GLFW_KEY_DOWN, [&camera](GLFWwindow *, int action, int mods) {
         camera.move_front(-0.1);
       }});

  // Main loop
  while (!glfwWindowShouldClose(interop_window.window.get())) {
    show_fps_and_window_size(interop_window.window.get());

    // Execute the CUDA code
    std::tie(camera.screen_width, camera.screen_height) = interop_window.interop_data.get_size();
    kernel_launcher(interop_window.interop_data.get_current_cuda_array(),
                    d_objects, objects.size(), camera);

    // Switch buffers
    interop_window.interop_data.blit_buffer();
    interop_window.interop_data.change_buffer();
    glfwSwapBuffers(interop_window.window.get());

    // Get events
    glfwPollEvents();
  }

  exit(EXIT_SUCCESS);
}
