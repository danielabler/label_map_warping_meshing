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
Functionality has been tested with singularity image created from above docker, using the ipython3 executable.

# Get & Use Singularity Image with dependencies

A singularity image containing all dependencies is available at singularity hub.

It can be downloaded from the shell using:
```shell script
singularity pull shub://danielabler/dockerfiles:libadjoint-2017-2_ants_meshtool
```

To start an interactive session:
```shell script
singularity shell -e <path_to_singularity_image>
```

To leave an interactive singularity session:
```shell script
exit
```


# Usage Example

- Go to folder where git repository was downloaded <base>.
- start singularity session
  ```shell script
  singularity shell -e <path_to_singularity_image>
  ```
- adapt script in analysis folder, eg. `analysis/pptT_prePTA.py`, for one of three steps:
  1. Downsample original OCT image & produce Matlab output:
     - From this output we compute displacement.
  2. Resize original OCT image based on displacement field:
     - Use coarse displacement field to identify approximate dimensions of image after warping.
     - Create image and matlab output for resized image
  3. Warp image based on displacement field
     - create high-res warped image using high-res displacement field

  These steps are outlined in any of the example files in <repository/analysis/*.py> .
  The procedure can be followed by un/commenting the respective lines of code for each step.

- run script from analysis folder, e.g.
    ```shell script
    python3 analysis/pptT_prePTA.py
    ```

- leave an interactive singularity session:
    ```shell script
    exit
    ```