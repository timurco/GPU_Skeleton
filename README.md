# GPU_Skeleton - GPU Based Plugin Template for After Effects

This repository serves as a template for GPU-based plugins for Adobe After Effects. It employs Adobe's macro set and boost postprocessing from [Boost](http://boost.org/) to create universal compute shaders compatible with **Metal**, **Cuda**, and **OpenCL**.

## Getting Started

To turn this template into your own project:
1. Clone this repository into the Adobe After Effects SDK folder: `SDK/Examples/Effects/`.
2. Open the `setup.sh` file and replace the project name:
    ```bash
    # Original and new project names
    OLD_PROJECT_NAME="GPU_Skeleton"
    # Please replace with your project name
    NEW_PROJECT_NAME="NewProjectName"
    ```
3. Run the setup file: `./setup.sh`.
This will replace all instances of the old project name in file names, variables, etc. with the new name. It also removes information about this repository and creates a new one.

## Features

The following features are implemented in this template:
- Display an image (PNG) in the effect window and print debug information inside it using a [custom parameter](https://ae-plugins.docsforadobe.dev/effect-ui-events/custom-ui-and-drawbot.html?highlight=Custom%20UI). Also, use the [stb_image library](https://github.com/nothings/stb/) to load the image into the [global variables buffer](https://ae-plugins.docsforadobe.dev/effect-basics/PF_OutData.html?highlight=global_data#pf-outdata-members).
- Debug a Metal shader by retrieving compilation errors and displaying them in the effect window in the corresponding custom parameter.
- Debug using `Debug.h` and macros to print information in the terminal, as well as to calculate time. This functionality is taken and modified from [ISF4AE](https://github.com/baku89/ISF4AE).

## References

The following resources were used:
- [Adobe After Effects Plugin Documentation](https://ae-plugins.docsforadobe.dev) 
- [ISF4AE](https://github.com/baku89/ISF4AE) - a plugin for creating effects based on OpenGL shaders for Apple.
- [stb](https://github.com/nothings/stb/) - an image conversion library.

## Acknowledgements

Many thanks to [Nate Lovell](https://github.com/NTProductions) for his video about [GPU](https://www.youtube.com/watch?v=Mbfk5jch6UI&t=211s), which inspired the creation of several great plugins and this template to kickstart your projects conveniently.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
