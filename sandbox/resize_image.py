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
import scipy.io as sio

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

def extract_structure_bounds_from_labelmap(image, label_id):
    image_ref_coord_array_flat, image_ref_value_array_flat = get_coord_value_array_for_image(image, flat=True)
    image_ref_sel_lumen = image_ref_value_array_flat == label_id
    image_ref_coord_array_flat_lumen = image_ref_coord_array_flat[image_ref_sel_lumen.T[0]]
    image_ref_min_pos = image_ref_coord_array_flat_lumen.min(axis=0)
    image_ref_max_pos = image_ref_coord_array_flat_lumen.max(axis=0)
    return image_ref_min_pos, image_ref_max_pos


#-- Reload input image
image_base_name = 'p1postPTACropped'
output_dir = os.path.join(config.output_dir_testing, '%s_nonisotropic'%image_base_name)
fu.ensure_dir_exists(output_dir)

#-- use original OCT image
path_image_ref = os.path.join(config.test_data_dir,'%s.mha'%image_base_name)
image_oct_resized = sitk.ReadImage(path_image_ref)

#-- use deformed image to resize original OCT image

# #- load deformed image & extract bounds of deformed lumen
# path_image_ref = os.path.join(config.output_dir_testing,'working_config_fine_mesh', 'output_image_def_res-0.10.mha')
# image_ref = sitk.ReadImage(path_image_ref)
# image_ref_min_pos, image_ref_max_pos = extract_structure_bounds_from_labelmap(image_ref, 1)
# #- load undeformed image & extract bounds
# path_image_oct = os.path.join(config.test_data_dir,'p1postPTA.mha')
# image_oct = sitk.ReadImage(path_image_oct)
# #image_oct_min_pos, image_oct_max_pos = extract_structure_bounds_from_labelmap(image_oct, 255)
#
# image_oct_min_pos = np.array([3.3152397, 2.976531 , 0.])
# image_oct_max_pos = np.array([ 8.1187449,  7.9647864, 49.4 ])
# all_extents = np.vstack([image_ref_min_pos, image_ref_max_pos, image_oct_min_pos, image_oct_max_pos])
# extent_all  = np.vstack([all_extents.min(axis=0), all_extents.max(axis=0)])



#-- use deformed image to resize original OCT image

# #- load deformed image & extract bounds of deformed lumen
# path_image_ref = os.path.join(config.output_dir_testing,'working_config_fine_mesh', 'output_image_def_res-0.10.mha')
# image_ref = sitk.ReadImage(path_image_ref)
# image_ref_min_pos, image_ref_max_pos = extract_structure_bounds_from_labelmap(image_ref, 1)
# #- load undeformed image & extract bounds
# path_image_oct = os.path.join(config.test_data_dir,'p1postPTA.mha')
# image_oct = sitk.ReadImage(path_image_oct)
# #image_oct_min_pos, image_oct_max_pos = extract_structure_bounds_from_labelmap(image_oct, 255)
#
# image_oct_min_pos = np.array([3.3152397, 2.976531 , 0.])
# image_oct_max_pos = np.array([ 8.1187449,  7.9647864, 49.4 ])
# all_extents = np.vstack([image_ref_min_pos, image_ref_max_pos, image_oct_min_pos, image_oct_max_pos])
# extent_all  = np.vstack([all_extents.min(axis=0), all_extents.max(axis=0)])

#
# #- resize oct image with specified margin around minimally sized configuration
# oct_size = image_oct.GetSize()
# oct_index_min = image_oct.TransformPhysicalPointToIndex(extent_all[0,:])
# oct_index_max = image_oct.TransformPhysicalPointToIndex(extent_all[1,:])
# margin = 5
# slices = []
# for i in range(3):
#     slice_min = oct_index_min[i]
#     slice_min_margin = slice_min - margin
#     if slice_min_margin < 0:
#         slice_min_margin = 0
#     slice_max = oct_index_max[i]
#     slice_max_margin = slice_max + margin
#     if slice_max_margin > oct_size[i]:
#         slice_max_margin = oct_size[i]
#     slice_def = slice(slice_min_margin, slice_max_margin)
#     slices.append(slice_def)
#
# image_oct_resized = image_oct[slices[0], slices[1], slices[2]]
#
# path_image_oct_resized = os.path.join(output_dir,'p1prePTA_resized.mha')
# sitk.WriteImage(image_oct_resized, path_image_oct_resized)

# -- compute matlab arrays
origin, size, spacing, extent, dim, vdim = get_measures_from_image(image_oct_resized)
# coord_array, value_array = get_coord_value_array_for_image(image_oct_resized, flat=False)
# matlab_name = '%s_image_coords_values_res-%.2f-%.2f-%.2f.mat' % (image_base_name, spacing[0], spacing[1], spacing[2])
# sio.savemat(os.path.join(output_dir, matlab_name),
#             {'coordinates': coord_array.astype(np.float32),
#              'values': value_array.astype(np.int8)})


#- upsample reduced size image
identity = sitk.Transform(dim, sitk.sitkIdentity)
output_direction = image_oct_resized.GetDirection()
output_origin    = origin

for spacing in [(0.2, 0.2, 0.2)]: #, (0.1, 0.1, 0.1)]:#, (0.05, 0.05, 0.05), (0.02, 0.02, 0.02)]:
    # -- resample image
    output_size = (extent[1, :] - extent[0, :]) / spacing
    output_size = [int(i) for i in list(output_size)]
    print("-- resampling to res-%.2f-%.2f-%.2f" % (spacing[0], spacing[1], spacing[2]))
    print("   -- size: %i-%i-%i" % (output_size[0], output_size[1], output_size[2]))
    image_oct_resized_resampled = sitk.Resample(image_oct_resized, output_size, identity, sitk.sitkNearestNeighbor,
                                                output_origin, spacing, output_direction)
    #-- create dir
    dir_name = '%s_resampled_size-%04d-%04d-%04d'%(image_base_name, output_size[0], output_size[1], output_size[2])
    output_dir = os.path.join(config.output_dir_testing, dir_name)
    fu.ensure_dir_exists(output_dir)
    #--save image
    image_name = '%s_resized_resampled_res-%.2f-%.2f-%.2f.mha'%(image_base_name, spacing[0], spacing[1], spacing[2])
    path_image_oct_resized_resampled = os.path.join(output_dir, image_name)
    sitk.WriteImage(image_oct_resized_resampled, path_image_oct_resized_resampled)
    #-- compute matlab arrays
    path_image_oct_resized_resampled = os.path.join(output_dir, image_name)
    coord_array, value_array = get_coord_value_array_for_image(image_oct_resized_resampled, flat=False)
    matlab_name = '%s_coords_values_res-%.2f-%.2f-%.2f.mat'%(image_base_name, spacing[0], spacing[1], spacing[2])
    sio.savemat(os.path.join(output_dir, matlab_name),
                                        {'coordinates':coord_array.astype(np.float32),
                                              'values':value_array.astype(np.int8)})

