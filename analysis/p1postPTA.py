import config
import pathlib as pl
from ArteryWarping import ArteryWarping


base_path = config.output_dir_analysis.joinpath('patient_01', 'postPTA')
aw = ArteryWarping(output_path=base_path)
aw.set_oct_image(path_to_oct_image=config.analysis_data_dir.joinpath('p1postPTA_compressed.mha'))

#%% (1) Downsample original OCT image & produce Matlab output
aw.resample_oct_orig(target_spacing=[0.2, 0.2, 0.2])

#%% (2) Resize original OCT image based on displacement field
# This requires the 'disp.mat' file to be present in the respective folder
# aw.resize_from_displacement(padding=[0.5, 0.5, 0.5],
#                             reference_description='resampled', reference_spacing=[0.2, 0.2, 0.2],
#                             target_description='resized_resampled', target_spacing=[0.1, 0.1, 0.1])
# aw.resize_from_displacement(padding=[0.5, 0.5, 0.5],
#                             reference_description='resampled', reference_spacing=[0.2, 0.2, 0.2],
#                             target_description='resized_resampled', target_spacing=[0.05, 0.05, 0.05])
# aw.resize_from_displacement(padding=[0.5, 0.5, 0.5],
#                             reference_description='resampled', reference_spacing=[0.2, 0.2, 0.2],
#                             target_description='resized_resampled', target_spacing=[0.2, 0.2, 0.2])

#%% (3) Warp image based on displacement field
# This requires 'disp.mat' as well as OCT and Xray centerlines to be present in respective folder
#aw.create_artifacts(target_spacing=[0.2, 0.2, 0.2], description='resampled')
# aw.create_artifacts(target_spacing=[0.1, 0.1, 0.1], description='resized_resampled')
# aw.create_artifacts(target_spacing=[0.05, 0.05, 0.05], description='resized_resampled')

