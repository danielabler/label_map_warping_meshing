import pathlib as pl
import utils
import SimpleITK as sitk
import numpy as np
import scipy.io as sio
import image_registration_utils as reg


class ArteryWarping():

    def __init__(self, output_path):
        self.output_path = pl.Path(output_path)

    def set_oct_image(self, path_to_oct_image):
        self.path_to_oct_image_orig = pl.Path(path_to_oct_image)
        self.oct_base_name = self.path_to_oct_image_orig.name.split('.')[0]

    def _create_file_name(self, description, resolution, ext='mha'):
        name = "%s_%s_res-%.2f-%.2f-%.2f.%s" % (self.oct_base_name, description,
                                                resolution[0], resolution[1], resolution[2], ext)
        return name

    def _create_dir_name(self, description, resolution, create=False):
        name = "%s_%s_res-%.2f-%.2f-%.2f" % (self.oct_base_name, description,
                                                resolution[0], resolution[1], resolution[2])
        path = self.output_path.joinpath(name)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_ref_image(self, target_spacing, description='resampled'):
        dir = self._create_dir_name(description=description, resolution=target_spacing, create=False)
        ref_img_name = self._create_file_name(description=description, resolution=target_spacing, ext='mha')
        path_to_ref_image = dir.joinpath(ref_img_name)
        ref_image = sitk.ReadImage(path_to_ref_image.as_posix())
        return ref_image, path_to_ref_image

    def _get_displacement_array(self, target_spacing, description='resampled'):
        dir = self._create_dir_name(description=description, resolution=target_spacing, create=False)
        displacement_mat_name = 'disps.mat'
        path_to_disp_mat = dir.joinpath(displacement_mat_name)
        print("-- Reading displacement field '%s'" % path_to_disp_mat)
        disp_array = utils.load_matlab_array(path_to_disp_mat.as_posix(), ['disps'])
        return disp_array, path_to_disp_mat

    def _resample_img_create_matlab(self, sitk_img, target_spacing=[0.2, 0.2, 0.2], target_extent=None,
                                    description='resampled', target_label=1):
        # resample image
        img_resampled = utils.resample_image(sitk_img, target_spacing, target_extent)
        # save image
        dir  = self._create_dir_name(description=description, resolution=target_spacing, create=True)
        image_name = self._create_file_name(description=description, resolution=target_spacing, ext='mha')
        path_to_image_resampled = dir.joinpath(image_name)
        print("    -- Saving resampled image as '%s'"%path_to_image_resampled)
        sitk.WriteImage(img_resampled, path_to_image_resampled.as_posix())
        # create matlab array
        print("  -- Creating matlab array")
        coord_array, value_array = utils.get_coord_value_array_for_image(img_resampled, flat=False)
        # recode any non zero value to 1
        value_array[value_array!=0] = target_label
        matlab_array =  {'coordinates': coord_array.astype(np.float32),
                         'values': value_array.astype(np.int8)}
        # save matlab array
        matlab_name = self._create_file_name(description=description+'_coords-values',
                                             resolution=target_spacing, ext='mat')
        path_to_matlab_array = dir.joinpath(matlab_name)
        print("    -- Saving matlab array as '%s'" % path_to_matlab_array)
        sio.savemat(path_to_matlab_array.as_posix(), matlab_array)

    def _create_centerline(self, centerline_name, target_spacing, description='resampled'):
        print("-- Constructing '%s'"%centerline_name)
        dir = self._create_dir_name(description=description, resolution=target_spacing, create=False)
        path_to_centerline = dir.joinpath(centerline_name + '.mat')
        centerline_array = utils.load_matlab_array(path_to_centerline.as_posix())
        if centerline_array is not None:
            line = utils.coord_array_to_vtk(centerline_array)
            centerline_output_name = self._create_file_name(description=description+'_'+centerline_name,
                                                            resolution=target_spacing, ext='vtp')
            path_to_centerline_vtp = dir.joinpath(centerline_output_name)
            print("    -- writing centerline to '%s'"%path_to_centerline_vtp)
            utils.write_vtk_data(line, path_to_centerline_vtp.as_posix())
        else:
            print("    -- Matlab array appears to be empty")

    def _create_displacement_imgs(self, target_spacing, description='resampled'):
        dir = self._create_dir_name(description=description, resolution=target_spacing, create=False)
        disp_array, path_to_disp_mat = self._get_displacement_array(target_spacing, description)
        disp_array = np.swapaxes(disp_array, 0, 2)  # swap x, z
        # compose image
        # -- get reference image
        ref_img, path_to_ref_image = self._get_ref_image(description=description, target_spacing=target_spacing)
        print("-- Reconstructing image from displacement field")
        disp_image = sitk.GetImageFromArray(disp_array, isVector=True)
        disp_image.SetOrigin(ref_img.GetOrigin())
        disp_image.SetSpacing(ref_img.GetSpacing())
        #-- saving disp image
        displacement_img_name = self._create_file_name(description=description + '_displacement',
                                                       resolution=target_spacing, ext='mha')
        path_to_disp_img = dir.joinpath(displacement_img_name)
        print("-- Saving displacement field as image '%s'" % path_to_disp_img)
        sitk.WriteImage(disp_image, path_to_disp_img.as_posix())

        print("-- Inverting displacement field")
        filter = sitk.InverseDisplacementFieldImageFilter()
        filter.SetOutputOrigin(disp_image.GetOrigin())
        filter.SetOutputSpacing(disp_image.GetSpacing())
        filter.SetSize(disp_image.GetSize())
        disp_image_inv = filter.Execute(disp_image)
        displacement_img_inv_name = self._create_file_name(description=description + '_displacement_inv',
                                                       resolution=target_spacing, ext='mha')
        path_to_disp_img_inv = dir.joinpath(displacement_img_inv_name)
        print("-- Saving inverted displacement field as image '%s'" % path_to_disp_img_inv)
        sitk.WriteImage(disp_image_inv, path_to_disp_img_inv.as_posix())
        displacement_img_inv_name = self._create_file_name(description=description + '_displacement_inv',
                                                       resolution=target_spacing, ext='nii')
        path_to_disp_img_inv = dir.joinpath(displacement_img_inv_name)
        print("-- Saving inverted displacement field as image '%s'" % path_to_disp_img_inv)
        sitk.WriteImage(disp_image_inv, path_to_disp_img_inv.as_posix())

    def _create_warped_image(self, target_spacing, description='resampled'):
        dir = self._create_dir_name(description=description, resolution=target_spacing, create=False)
        # -- get reference image
        ref_img_name = self._create_file_name(description=description, resolution=target_spacing, ext='mha')
        path_to_ref_image = dir.joinpath(ref_img_name)
        # -- get inv displacement image
        inv_disp_img_name = self._create_file_name(description=description+ '_displacement_inv',
                                                   resolution=target_spacing, ext='nii')
        path_to_inv_disp_image = dir.joinpath(inv_disp_img_name)
        print("-- Applying displacement field to reference image")
        warped_img_name = self._create_file_name(description=description+"_warped", resolution=target_spacing, ext='mha')
        path_to_warped_img = dir.joinpath(warped_img_name)
        reg.ants_apply_transforms(input_img=path_to_ref_image.as_posix(), output_file=path_to_warped_img.as_posix(),
                                  reference_img=path_to_ref_image.as_posix(),
                                  transforms=[path_to_inv_disp_image.as_posix()], dim=3, interpolation='GenericLabel')

    @staticmethod
    def _extract_structure_bounds_from_coords_and_displacement(coords_array, values_array, displacement_array,
                                                              label_id=1):
        displaced_coords_array = coords_array + displacement_array
        displaced_coords_array_flat = displaced_coords_array.reshape(np.prod(displaced_coords_array.shape[:3]),
                                                                     displaced_coords_array.shape[3])
        values_array_flat = values_array.reshape(np.prod(values_array.shape))
        displaced_coords_array_flat_lumen = displaced_coords_array_flat[np.where(values_array_flat == label_id)]
        image_ref_min_pos = displaced_coords_array_flat_lumen.min(axis=0)
        image_ref_max_pos = displaced_coords_array_flat_lumen.max(axis=0)
        return image_ref_min_pos, image_ref_max_pos

    @staticmethod
    def _extract_structure_bounds_from_labelmap(image, label_id):
        image_ref_coord_array_flat, image_ref_value_array_flat = utils.get_coord_value_array_for_image(image,
                                                                                                       flat=True)
        image_ref_sel_lumen = image_ref_value_array_flat == label_id
        image_ref_coord_array_flat_lumen = image_ref_coord_array_flat[image_ref_sel_lumen.T[0]]
        image_ref_min_pos = image_ref_coord_array_flat_lumen.min(axis=0)
        image_ref_max_pos = image_ref_coord_array_flat_lumen.max(axis=0)
        return image_ref_min_pos, image_ref_max_pos

    def _get_extent_all_from_displacement(self, description='resampled', target_spacing=[0.2, 0.2, 0.2]):
        # get structure bounds in straightened configuration
        ref_img, path_to_ref_image = self._get_ref_image(description=description, target_spacing=target_spacing)
        image_ref_min_pos, image_ref_max_pos = self._extract_structure_bounds_from_labelmap(ref_img, label_id=255)
        # get structure bounds from dispalcement field
        disp_array, path_to_disp_mat = self._get_displacement_array(description=description, target_spacing=target_spacing)
        dir = self._create_dir_name(description=description, resolution=target_spacing, create=False)
        coords_values_name = self._create_file_name(description=description+'_coords-values', resolution=target_spacing, ext='mat')
        path_to_coords_values = dir.joinpath(coords_values_name)
        coords_array, values_array = utils.load_matlab_array(path_to_coords_values.as_posix(),['coordinates', 'values'])
        image_def_min_pos, image_def_max_pos = \
            self._extract_structure_bounds_from_coords_and_displacement(coords_array, values_array, disp_array, label_id=1)
        # combine extents
        all_extents = np.vstack([image_ref_min_pos, image_ref_max_pos, image_def_min_pos, image_def_max_pos])
        extent_all = np.vstack([all_extents.min(axis=0), all_extents.max(axis=0)])
        return extent_all

    def _create_surface_mesh_from_warped_image(self, description='resampled', target_spacing=[0.2, 0.2, 0.2]):
        dir = self._create_dir_name(description=description, resolution=target_spacing, create=False)
        warped_img_name = self._create_file_name(description=description + "_warped", resolution=target_spacing, ext='mha')
        path_to_warped_img = dir.joinpath(warped_img_name)

        surface_mesh_name_vtp = self._create_file_name(description=description+'_warped_surface', resolution=target_spacing, ext='vtp')
        path_surface_mesh_vtp = dir.joinpath(surface_mesh_name_vtp)
        utils.create_surfacemesh_from_labelmap(path_to_label_map=path_to_warped_img.as_posix(),
                                               path_to_surface_mesh=path_surface_mesh_vtp.as_posix(), label_id=255)
        surface_mesh_name_stl = self._create_file_name(description=description + '_warped_surface', resolution=target_spacing, ext='stl')
        path_surface_mesh_stl = dir.joinpath(surface_mesh_name_stl)
        utils.create_surfacemesh_from_labelmap(path_to_label_map=path_to_warped_img.as_posix(),
                                               path_to_surface_mesh=path_surface_mesh_stl.as_posix(), label_id=255)


    def resample_oct_orig(self, target_spacing=[0.2, 0.2, 0.2]):
        oct_img_orig = sitk.ReadImage(self.path_to_oct_image_orig.as_posix())
        self._resample_img_create_matlab(oct_img_orig, target_spacing=target_spacing, description='resampled')

    def create_artifacts(self, target_spacing, description='resampled'):
        self._create_centerline(centerline_name='OCTCenterLine', target_spacing=target_spacing, description=description)
        self._create_centerline(centerline_name='XrayCenterLine', target_spacing=target_spacing, description=description)
        self._create_displacement_imgs(target_spacing=target_spacing, description=description)
        self._create_warped_image(target_spacing=target_spacing, description=description)
        self._create_surface_mesh_from_warped_image(target_spacing=target_spacing, description=description)

    def resize_from_displacement(self, padding=[0.5, 0.5, 0.5],
                                 reference_description='resampled', reference_spacing=[0.2, 0.2, 0.2],
                                 target_description='resized_resampled', target_spacing=[0.2, 0.2, 0.2]):
        extent_all = self._get_extent_all_from_displacement(description=reference_description, target_spacing=reference_spacing)
        img_oct = sitk.ReadImage(self.path_to_oct_image_orig.as_posix())
        padding = np.array(padding)
        extent_all_pad = np.vstack([extent_all[0] - padding, extent_all[1] + padding])
        self._resample_img_create_matlab(img_oct, target_spacing=target_spacing, target_extent=extent_all_pad,
                                        description=target_description)
