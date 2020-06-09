import config
import pathlib as pl
from ArteryWarping import ArteryWarping


base_path = config.output_dir_analysis.joinpath('patient_01', 'prePTA')
aw = ArteryWarping(output_path=base_path)
aw.set_oct_image(path_to_oct_image=config.analysis_data_dir.joinpath('p1prePTA_compressed.mha'))

#%% (1) Downsample original OCT image & produce Matlab output
#aw.resample_oct_orig(target_spacing=[0.2, 0.2, 0.2])

#%% (2) Resize original OCT image based on displacement field
# aw.resize_from_displacement(padding=[0.5, 0.5, 0.5],
#                             reference_description='resampled', reference_spacing=[0.2, 0.2, 0.2],
#                             target_description='resized_resampled', target_spacing=[0.1, 0.1, 0.1])

#%% (3) Warp image based on displacement field
#aw.create_artifacts(target_spacing=[0.1, 0.1, 0.1], description='resized_resampled')
