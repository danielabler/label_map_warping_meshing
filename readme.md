# Labelmap warping for mesh creation

Workflow example for warping & meshing arbitrary geometry, including   
- example surface mesh
- image creation from surface mesh
- creation of example displacement field
- warping by displacement field using ANTs
- meshing of final warped image

Functionality depends on
- python vtk
- simpleITK
- ANTs installation
- installation of [MeshTool](https://c4science.ch/diffusion/9312/)

All dependencies are included in this [docker image](https://github.com/danielabler/dockerfiles/tree/master/fenics/2017.2.0_libadjoint_ants_meshtool).
Functionality has been tested with singularity image created from above docker, using ipython3 executable.

# Usage Example

3-Step procedure:
 1. Downsample original OCT image & produce Matlab output:
    - From this output we compute displacement.
 2. Resize original OCT image based on displacement field:
    - Use coarse displacement field to identify approximate dimensions of image after warping.
    - Create image and matlab output for resized image
 3. Warp image based on displacement field
    - create high-res warped image using high-res displacement field

These steps are outlined in any of the example files in <repository/analysis/*.py> .
The procedure can be followed by un/commenting the respective lines of code for each step.

To run python file use the singularity image located at <path_to_singularity>:

- navigate to repository directory
- enter singularity image:
  singularity shell -e <path_to_singularity>
- run respective python file: 
  python3 script.py
- check results in output directory
- to leave the singularity container:
  exit