all : bubbles

clean :
	rm bubbles

bubbles :
	nvcc -o bubbles src/main.cpp src/interop.cpp src/interop_window.cpp src/kernel.cu glad/glad.c -I glad /lib/libglfw.so
