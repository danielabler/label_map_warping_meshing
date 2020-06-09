import pathlib as pl

base_path = pl.Path(__file__).parents[0]

output_dir                              = base_path.joinpath('output')
output_dir.mkdir(exist_ok=True)
output_dir_testing                      = output_dir.joinpath('sandbox')
output_dir_analysis                     = output_dir.joinpath('analysis')
output_dir_analysis.mkdir(exist_ok=True)
output_dir_temp                         = output_dir.joinpath('temp')
output_dir_temp.mkdir(exist_ok=True)


analysis_dir = base_path.joinpath('analysis')
analysis_data_dir = analysis_dir.joinpath('data')

# meshtool settings
path_to_meshtool = pl.Path('/home/fenics/software/MESHTOOL_source')
path_to_meshtool_bin = path_to_meshtool.joinpath('bin', 'MeshTool').as_posix()
path_to_meshtool_xsd = path_to_meshtool.joinpath('src', 'xml-io', 'imaging_meshing_schema.xsd').as_posix()
