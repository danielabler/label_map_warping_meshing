import os

import vtk
import math
from vtk.numpy_interface import dataset_adapter as dsa
import SimpleITK as sitk

import numpy as np
import config
import file_utils as fu
import image_registration_utils as reg
import meshing


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


def read_vtk_data(_path_to_file):
    if os.path.exists(_path_to_file):
        extension = fu.get_file_extension(_path_to_file)
        if extension == 'vtk':
            reader = vtk.vtkDataSetReader()
        elif extension == 'vtp':
            reader = vtk.vtkXMLPolyDataReader()
        elif extension == 'stl':
            reader = vtk.vtkSTLReader()
        elif extension == 'vtu':
            reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(_path_to_file)
        reader.Update()
        return reader.GetOutput()
    else:
        print("Path does not exist")



def create_image_from_vtp(vtp_in, spacing=[1,1,1], bounds=None, path_to_image=None, label=1):
    whiteImage = vtk.vtkImageData()
    origin = [0, 0, 0]
    if bounds==None:
        bounds = vtp_in.GetBounds()
        origin[0] = bounds[0] + spacing[0] / 2
        origin[1] = bounds[2] + spacing[1] / 2
        origin[2] = bounds[4] + spacing[2] / 2
    else:
        origin[0] = bounds[0]
        origin[1] = bounds[2]
        origin[2] = bounds[4]
    whiteImage.SetOrigin(origin)
    whiteImage.SetSpacing(spacing)
    # compute dimensions
    dim = [0, 0, 0]
    for i in range(3):
        dim[i] = int(math.ceil((bounds[i * 2 + 1] - bounds[i * 2]) / spacing[i])) + 1
    whiteImage.SetDimensions(dim)
    whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1);
    if vtk.VTK_MAJOR_VERSION <= 5:
        whiteImage.SetScalarTypeToUnsignedChar()
        whiteImage.AllocateScalars()
    else:
        whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    # fill the image with foreground voxels:
    outval = 0
    count = whiteImage.GetNumberOfPoints();
    for i in range(count):
        whiteImage.GetPointData().GetScalars().SetTuple1(i, label)
    # polygonal data --> image stencil:
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    if vtk.VTK_MAJOR_VERSION <= 5:
        pol2stenc.SetInput(vtp_in)
    else:
        pol2stenc.SetInputData(vtp_in)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
    pol2stenc.Update()
    # cut the corresponding white image and set the background:
    imgstenc = vtk.vtkImageStencil()
    if vtk.VTK_MAJOR_VERSION <= 5:
        imgstenc.SetInput(whiteImage)
        imgstenc.SetStencil(pol2stenc.GetOutput())
    else:
        imgstenc.SetInputData(whiteImage)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()
    image = imgstenc.GetOutput()
     # write data
    if path_to_image!=None:
        write_image(image, path_to_image)
        print("Writing seed image to: %s" % (path_to_image))
    return image




def write_image(_data, path_to_image):
    # == make sure data type is unsigned char
    cast = vtk.vtkImageCast()
    if vtk.VTK_MAJOR_VERSION <= 5:
        cast.SetInput(_data)
    else:
        cast.SetInputData(_data)
    cast.SetOutputScalarTypeToUnsignedChar()
    cast.Update()
    _data = cast.GetOutput()
    # == write
    file_ext = fu.get_file_extension(path_to_image)
    if (file_ext=="vti"):
        print("Writing as  '.vti' file.")
        image_writer = vtk.vtkXMLImageDataWriter()
    elif (file_ext=="nii"):
        print("Writing as  '.nii' file.")
        image_writer = vtk.vtkNIFTIImageWriter()
        print("VTK seems to change image orientation of NIFTI. Make sure to check image orientation relative to original image")
    elif (file_ext=="mhd" or file_ext=="mha"):
        print("Writing as .mhd/.raw or .mha image.")
        image_writer = vtk.vtkMetaImageWriter()
        image_writer.SetCompression(False)
    else:
        print("No valid image file extension specified!")
    if vtk.VTK_MAJOR_VERSION <= 5:
        image_writer.SetInput(_data)
    else:
        image_writer.SetInputData(_data)
    image_writer.SetFileName(path_to_image)
    # image_writer.Update()
    image_writer.Write()
    print("Image has been saved as %s " %(path_to_image))


def convert_vti_to_img(vti_in, array_name='material', RGB=False, invert_values=False):
    sim_vtu_wrapped = dsa.WrapDataObject(vti_in)
    field = sim_vtu_wrapped.PointData[array_name]
    _, nx, _, ny, _, nz = vti_in.GetExtent()
    spacing = vti_in.GetSpacing()
    if vti_in.GetDataDimension() == 2:
        shape = (nx + 1, ny + 1)
        spacing = spacing[:2]
    else:
        shape = (nx + 1, ny + 1, nz + 1)
    if invert_values:
        field = - np.array(field)
    # invert input shape: sitk numpy z,y,x -> sitk img x,y,z
    shape_sitk_np = shape[::-1]
    if RGB:
        field_reshaped = np.reshape(field, (*shape_sitk_np, 3))
        img = sitk.GetImageFromArray(field_reshaped, isVector=True)
    else:
        field_reshaped = np.reshape(field, shape_sitk_np)
        img = sitk.GetImageFromArray(field_reshaped)
    img.SetOrigin(vti_in.GetOrigin())
    img.SetSpacing(spacing)
    return img


def get_measures_from_image(sitk_image):
    origin = sitk_image.GetOrigin()
    spacing = sitk_image.GetSpacing()
    height = sitk_image.GetHeight()
    width = sitk_image.GetWidth()
    depts = sitk_image.GetDepth()
    dim = sitk_image.GetDimension()
    # size
    size = np.zeros(dim, dtype=int)
    size[0] = width
    size[1] = height
    if dim==3:
        size[2]=depts
    # extent
    extent = np.zeros((2, dim), dtype=float)
    for i in range(0, dim):
        extent[0, i] = origin[i]
        extent[1, i] = origin[i] + spacing[i] * (size[i]-1) # n-1 distances between n points
    # value dimensionality
    vdim = sitk_image.GetNumberOfComponentsPerPixel()
    return np.array(origin), size, np.array(spacing), extent, dim, vdim


def get_coord_value_array_for_image(image, flat=False):
    origin, size, spacing, extent, dim, vdim = get_measures_from_image(image)
    coord_array = np.zeros( (*size, dim))
    value_array = np.zeros( (*size, vdim))
    if dim==2:
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                index = (i, j)
                coord = image.TransformIndexToPhysicalPoint( index )
                coord_array[i, j, :] = coord
                value = image.GetPixel( index )
                value_array[i, j, :] = value
    elif dim==3:
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                for k in range(0, size[2]):
                    index = (i, j, k)
                    coord = image.TransformIndexToPhysicalPoint(index)
                    coord_array[i, j, k, :] = coord
                    value = image.GetPixel(index)
                    value_array[i, j, k, :] = value
    if flat:
        if dim==2:
            new_shape_first = size[0]*size[1]
        elif dim==3:
            new_shape_first = size[0] * size[1] * size[2]
        coord_array = coord_array.reshape(new_shape_first, dim)
        value_array = value_array.reshape(new_shape_first, vdim)
    return coord_array, value_array

def resample_image(image_in, size_new):
    origin, size, spacing, extent, dim, vdim = get_measures_from_image(image_in)
    print("Downsampling image: %s -> %s" % (size, np.array2string(np.array(size_new))))
    image_new = sitk.Image(size_new, image_in.GetPixelIDValue())
    image_new.SetOrigin(image_in.GetOrigin())
    image_new.SetDirection(image_in.GetDirection())
    image_new.SetSpacing(
        [sz * spc / nsz for nsz, sz, spc in zip(size_new, size, spacing)])
    image_resampled = sitk.Resample(image_in, image_new)
    return image_resampled


def coord_array_to_vtk(coord_array):
    n_points = coord_array.shape[0]
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(n_points)
    lines = vtk.vtkCellArray()
    for i in range(n_points):
        points.SetPoint(i, *coord_array[i, :])
        lines.InsertNextCell(1)  # number of points
        lines.InsertCellPoint(i)
    line = vtk.vtkPolyData()
    line.SetPoints(points)
    line.SetLines(lines)
    return line


#-- Reload input image
image_base_name = 'p1postPTACropped'
output_dir = os.path.join(config.output_dir_testing, '%s_resampled_size-0036-0035-0248'%image_base_name)
#output_dir = os.path.join(config.output_dir_testing, 'p1postPTA_nonisotropic')
spacing = [0.20, 0.20, 0.20]
path_image_undef = os.path.join(output_dir, '%s_resized_resampled_res-%.2f-%.2f-%.2f.mha'% (image_base_name, spacing[0], spacing[1], spacing[2]))
input_image_undef = sitk.ReadImage(path_image_undef)


import scipy.io as sio

print("-- reading displacement field")
path_to_def_field = os.path.join(output_dir, 'disps.mat')
disp_array = sio.loadmat(path_to_def_field)['disps']
disp_array_flat = disp_array.reshape(np.prod(disp_array.shape[:3]), disp_array.shape[3])
disp_array = np.swapaxes(disp_array, 0, 2)  # swap x, z
# compose image
print("-- saving displacement field as image")
disp_image = sitk.GetImageFromArray(disp_array, isVector=True)
disp_image.SetOrigin(input_image_undef.GetOrigin())
disp_image.SetSpacing(input_image_undef.GetSpacing())
path_image_disp_nii = os.path.join(output_dir, 'image_displacement_res-%.2f-%.2f-%.2f.nii'% (spacing[0], spacing[1], spacing[2]))
path_image_disp_mha = os.path.join(output_dir, 'image_displacement_res-%.2f-%.2f-%.2f.mha'% (spacing[0], spacing[1], spacing[2]))
sitk.WriteImage(disp_image, path_image_disp_nii)
sitk.WriteImage(disp_image, path_image_disp_mha)

print("-- constructing OCT centerline")
path_to_centerline = os.path.join(output_dir, 'OCTCenterLine.mat')
centerline_array = sio.loadmat(path_to_centerline)['coordsC']
line = coord_array_to_vtk(centerline_array)
path_to_centerline_vtp = os.path.join(output_dir, 'OCTCenterLine_res-%.2f-%.2f-%.2f.vtp'% (spacing[0], spacing[1], spacing[2]))
write_vtk_data(line, path_to_centerline_vtp)

print("-- constructing X-ray centerline")
path_to_centerline = os.path.join(output_dir, 'XrayCenterLine.mat')
centerline_array = sio.loadmat(path_to_centerline)['A_new']
line = coord_array_to_vtk(centerline_array)
path_to_centerline_vtp = os.path.join(output_dir, 'XrayCenterLine_res-%.2f-%.2f-%.2f.vtp'% (spacing[0], spacing[1], spacing[2]))
write_vtk_data(line, path_to_centerline_vtp)

print("-- inverting displacement field")
filter = sitk.InverseDisplacementFieldImageFilter()
filter.SetOutputOrigin(disp_image.GetOrigin())
filter.SetOutputSpacing(disp_image.GetSpacing())
filter.SetSize(disp_image.GetSize())
disp_image_inv = filter.Execute(disp_image)
path_image_disp_inv_mha = os.path.join(output_dir, 'image_displacement_inv_res-%.2f-%.2f-%.2f.mha'% (spacing[0], spacing[1], spacing[2]))
sitk.WriteImage(disp_image_inv, path_image_disp_inv_mha)
path_image_disp_inv_nii = os.path.join(output_dir, 'image_displacement_inv_res-%.2f-%.2f-%.2f.nii'% (spacing[0], spacing[1], spacing[2]))
sitk.WriteImage(disp_image_inv, path_image_disp_inv_nii)

print("-- applying displacement field")
#-- apply ANTS transformation
path_image_def = os.path.join(output_dir, 'output_image_def_res-%.2f-%.2f-%.2f.mha'% (spacing[0], spacing[1], spacing[2]))
reg.ants_apply_transforms(input_img=path_image_undef, output_file=path_image_def,
                          reference_img=path_image_undef,
                          transforms=[path_image_disp_inv_nii], dim=3, interpolation='GenericLabel')

print("-- creating mesh")
meshing_params = {'global'     : {"cell_radius_edge_ratio": 2.1,
                                "cell_size": 0.1,
                                "facet_angle": 30.0,
                                "facet_distance": 0.1,
                                "facet_size": 0.3}
                  }

path_mesh_def = os.path.join(output_dir, "mesh_3D_res-%.2f-%.2f-%.2f.vtu"% (spacing[0], spacing[1], spacing[2]))
path_mesh_config = os.path.join(output_dir, "meshing_params.xml")

meshing.create_mesh_xml(path_to_image_in=path_image_def,
                        path_to_mesh_out=path_mesh_def,
                        tissues_dict=meshing_params,
                        path_to_xml_file=path_mesh_config)

meshing.mesh_image(path_to_meshtool_bin=config.path_to_meshtool_bin,
                   path_to_meshtool_xsd=config.path_to_meshtool_xsd,
                   path_to_config_file=path_mesh_config)

#-- resize image
path_disp_imag = os.path.join(output_dir,'working_config_fine_mesh', 'output_image_def_res-0.10.mha')
image_ref = sitk.ReadImage(path_image_ref)
image_ref_min_pos, image_ref_max_pos = extract_structure_bounds_from_labelmap(image_ref, 1)
#- load undeformed image & extract bounds
path_image_oct = os.path.join(config.test_data_dir,'p1postPTA.mha')
image_oct = sitk.ReadImage(path_image_oct)
#image_oct_min_pos, image_oct_max_pos = extract_structure_bounds_from_labelmap(image_oct, 255)

image_oct_min_pos = np.array([3.3152397, 2.976531 , 0.])
image_oct_max_pos = np.array([ 8.1187449,  7.9647864, 49.4 ])
all_extents = np.vstack([image_ref_min_pos, image_ref_max_pos, image_oct_min_pos, image_oct_max_pos])
extent_all  = np.vstack([all_extents.min(axis=0), all_extents.max(axis=0)])
