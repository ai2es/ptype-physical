import xarray as xr
from os.path import join
import os

data_dir = "/glade/work/dkimpara/ptype-aggs"

def get_datasets(model, res_type):
    paths = [path
           for path in os.listdir(data_dir)
           if ('nc' in path) and (model in path) and ('all' not in path) and (res_type in path)]
    print(paths)
    datasets = [xr.open_dataset(join(data_dir,path), engine='netcdf4') for path in paths]
    return datasets

def concat_results(model, res_type):
    datasets = get_datasets(model, res_type)
    datasets = [ds.drop_duplicates('predtype') for ds in datasets]
    ds = xr.concat(datasets,dim='case_study_day')
    ds.to_netcdf(join(data_dir, f"merged/{model}_all_{res_type}.nc"))

restypes = ['conf79', 'conf35']
models = ['hrrr']
[concat_results(model, res_type) for model in models for res_type in restypes]