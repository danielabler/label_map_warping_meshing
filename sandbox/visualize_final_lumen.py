import os
import config
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
import vtk
import file_utils as fu


def write_vtk_data(_data, _path_to_file):
    fu.ensure_dir_exists(_path_to_file)
    writer = vtk.vtkXMLDataSetWriter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(_data)
    else:
        writer.SetInputData(_data)
    writer.SetFileName(_path_to_file)
    writer.Update()
    writer.Write()


#-- Reload input image
output_dir = os.path.join(config.output_dir_testing, 'surface_mesh_to_array')
path_image_undef = os.path.join(output_dir, 'input_image_undef.mha')
input_image_undef = sitk.ReadImage(path_image_undef)
input_image_undef_array = sitk.GetArrayFromImage(input_image_undef)

input_image_undef_array = np.swapaxes(input_image_undef_array, 0, 2)  # swap x, z


path_to_final_pos = os.path.join(output_dir, 'FinalNodes.mat')
final_pos_array = sio.loadmat(path_to_final_pos)['TargetNodes']

final_pos_array_lumen = final_pos_array[input_image_undef_array==1]

n_items = final_pos_array_lumen.shape[0]
points = vtk.vtkPoints()
points.SetNumberOfPoints(n_items)

for i in range(n_items):
    points.SetPoint(i, *final_pos_array_lumen[i, :])

vtk_poly = vtk.vtkPolyData()
vtk_poly.SetPoints(points)
path_to_vtp = os.path.join(output_dir, 'final_positions.vtp')
write_vtk_data(vtk_poly, path_to_vtp)

