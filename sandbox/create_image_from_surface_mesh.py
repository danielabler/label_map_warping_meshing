import os
import SimpleITK as sitk
import config
import file_utils as fu
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import math
import numpy as np
import scipy.io as sio


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


output_dir = os.path.join(config.output_dir_testing, 'working_config_fine_mesh')
fu.ensure_dir_exists(output_dir)

path_surface_mesh = os.path.join(config.test_data_dir, 'pt1PrePTASmooth.vtp')
input_vtp = read_vtk_data(path_surface_mesh)



for res in [0.5, 0.2, 0.1]:
    path_image_input_vti = os.path.join(output_dir, 'input_image_undef_res-%.2f.vti'%res)
    input_image_from_mesh_vti = create_image_from_vtp(input_vtp,
                                                  spacing=[res, res, res],
                                                  bounds=[2,11,1,9,0,50],
                                                  path_to_image=path_image_input_vti, label=1)

    path_image_undef = os.path.join(output_dir, 'input_image_undef_res-%.2f.mha'%res)
    input_image_undef_sitk = convert_vti_to_img(input_image_from_mesh_vti, array_name='ImageScalars', RGB=False, invert_values=False)
    sitk.WriteImage(input_image_undef_sitk, path_image_undef)

    coord_array, value_array = get_coord_value_array_for_image(input_image_undef_sitk, flat=False)

    sio.savemat(os.path.join(output_dir, 'image_coords_values_res-%.2f.mat'%res),
                                        {'coordinates':coord_array.astype(np.float32),
                                              'values':value_array.astype(np.int8)})