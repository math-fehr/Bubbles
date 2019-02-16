# Bubbles
Bubbles ? Bubbles ! A small real time ray-tracer/marcher written in CUDA.

This is Thibaut Pérami and Mathieu Fehr project for the M2 class "Computer Graphics and Visualization" in the MPRI.

## Building

You need `cmake` and `cuda` to build the executable.

`ccmake` can be used to change the compilation parameters (to change the initial size of the window for instance).

The executable will be `bin/Bubbles`

## Shortcuts

These are the shortcuts when launching the program (these shortcuts are based on an azerty keyboard):
- `T`, `Y`, `U`, `I`, `O`, `P` load different scenes
- `X` change the position of the light to the position of the camera
- `G` activate and `V` deactivate the creation of bubble in pipes
- `H` activate and `B` deactivate the movement of bubbles
- `J` activate and `N` deactivate the movement of light
- `K` increase and `,` decrease the speed of the bubbles
- `L` increase and `;` decrease the gamma
- `Left Click` fire a bullet from the camera position in the direction of the camera
- `Right Click` creates a bullet in the camera position
