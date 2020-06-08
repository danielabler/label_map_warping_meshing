import SimpleITK as sitk
import numpy as np
import vtk
import file_utils as fu
import scipy.io as sio

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

def resample_image(sitk_img, target_spacing, target_extent=None):
    #- get image dimensions, compute new dimensions
    origin, size, spacing, extent, dim, vdim = get_measures_from_image(sitk_img)
    if target_extent is None:
        target_extent = extent
    output_size = (target_extent[1, :] - target_extent[0, :]) / np.array(target_spacing)
    output_size = [int(i) for i in list(output_size)]
    print("-- resampling to res-%.2f-%.2f-%.2f" % (target_spacing[0], target_spacing[1], target_spacing[2]))
    print("   -- size: %i-%i-%i" % (output_size[0], output_size[1], output_size[2]))
    #- create identity transform
    identity = sitk.Transform(dim, sitk.sitkIdentity)
    output_direction = sitk_img.GetDirection()
    output_origin    = target_extent[0, :]
    # resample image
    image_resampled = sitk.Resample(sitk_img, output_size, identity, sitk.sitkNearestNeighbor,
                                    output_origin, target_spacing, output_direction)
    return image_resampled


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


def write_vtk_data(_data, _path_to_file):
    fu.ensure_dir_exists(_path_to_file)
    file_ext = fu.get_file_extension(_path_to_file)
    if (file_ext == "stl"):
        writer = vtk.vtkSTLWriter()
        writer.SetFileTypeToBinary()
    else:
        writer = vtk.vtkXMLDataSetWriter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(_data)
    else:
        writer.SetInputData(_data)
    writer.SetFileName(_path_to_file)
    writer.Write()

def load_matlab_array(path_to_file, array_names=None):
    matlab_dict = sio.loadmat(path_to_file)
    if not array_names:
        array_names = [key for key in matlab_dict.keys() if not key.startswith('__')]
    arrays = [matlab_dict[array_name] for array_name in array_names if array_name in matlab_dict.keys()]
    if len(arrays)==1:
        arrays=arrays[0]
    elif len(arrays)==0:
        arrays=None
    return arrays

def load_image(path_to_image):
    file_ext = fu.get_file_extension(path_to_image)
    if (file_ext=="vti"):
        print("Opening as  '.vti' file.")
        image_reader = vtk.vtkXMLImageDataReader()
    elif (file_ext=="nii"):
        print("Opening as  '.nii' file.")
        image_reader = vtk.vtkNIFTIImageReader()
    else:
        print("Attempting to load as other vtk image.")
        reader_factory = vtk.vtkImageReader2Factory()
        image_reader = reader_factory.CreateImageReader2(path_to_image)
    image_reader.SetFileName(path_to_image)
    image_reader.Update()
    image = image_reader.GetOutput()
    return image


def create_surfacemesh_from_labelmap(path_to_label_map, path_to_surface_mesh, label_id):
    print("-- creating surface mesh from labelmap '%s'"%path_to_label_map)
    vtk_img = load_image(path_to_label_map)
    filter = vtk.vtkDiscreteMarchingCubes()
    filter.SetInputData(vtk_img)
    filter.GenerateValues(label_id, label_id, label_id)
    filter.Update()
    # filter = vtk.vtkContourFilter()
    # filter.SetInputData(vtk_img)
    # filter.GenerateValues(label_id, label_id, label_id)
    # filter.Update()
    surface_vtp = filter.GetOutput()
    print("-- writing surface mesh to '%s'"%path_to_surface_mesh)
    write_vtk_data(surface_vtp, path_to_surface_mesh)
    return surface_vtp