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
