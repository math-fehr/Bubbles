#include "interop.hpp"

#include "assert_cuda.hpp"
#include <cuda_gl_interop.h>

InteropData::InteropData(unsigned n_buffers)
    : n_buffers(n_buffers), current_buffer(0), fb(n_buffers), rb(n_buffers),
      cgr(n_buffers), ca(n_buffers) {
  glCreateRenderbuffers(n_buffers, rb.data());
  glCreateFramebuffers(n_buffers, fb.data());

  // attach rbo to fbo
  for (int i = 0; i < n_buffers; i++) {
    glNamedFramebufferRenderbuffer(fb[i], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                                   rb[i]);
  }
}

InteropData::~InteropData() {
  // unregister CUDA resources
  for (int i = 0; i < n_buffers; ++i) {
    if (cgr[i] != nullptr) {
      cuda(GraphicsUnregisterResource(cgr[i]));
    }
  }

  glDeleteRenderbuffers(n_buffers, rb.data());
  glDeleteFramebuffers(n_buffers, fb.data());
}

std::pair<unsigned, unsigned> InteropData::get_size() {
  return {width, height};
}

void InteropData::set_size(unsigned width, unsigned height) {
  // save new size
  this->width = width;
  this->height = height;

  // resize color buffer
  for (int i = 0; i < n_buffers; ++i) {

    // unregister resources
    if (cgr[i] != nullptr) {
      cuda(GraphicsUnregisterResource(cgr[i]));
    }

    // resize rbo
    glNamedRenderbufferStorage(rb[i], GL_RGBA8, width, height);

    // register rbo
    cuda(GraphicsGLRegisterImage(&cgr[i], rb[i], GL_RENDERBUFFER,
                                 cudaGraphicsRegisterFlagsSurfaceLoadStore |
                                     cudaGraphicsRegisterFlagsWriteDiscard));
  }
  // map graphics resources
  cuda(GraphicsMapResources(n_buffers, cgr.data(), 0));

  // get CUDA array references
  for (int i = 0; i < n_buffers; ++i) {
    cuda(GraphicsSubResourceGetMappedArray(&ca[i], cgr[i], 0, 0));
  }

  cuda(GraphicsUnmapResources(n_buffers, cgr.data(), 0));
}

cudaArray_const_t InteropData::get_current_cuda_array() {
  return ca[current_buffer];
}

void InteropData::clear_buffer(std::array<float, 4> clear_color) {
  glClearNamedFramebufferfv(fb[current_buffer], GL_COLOR, 0,
                            clear_color.data());
}

void InteropData::blit_buffer() {
  glBlitNamedFramebuffer(fb[current_buffer], 0, 0, 0, width, height, 0, height,
                         width, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

void InteropData::change_buffer() {
  current_buffer = (current_buffer + 1) % n_buffers;
}
