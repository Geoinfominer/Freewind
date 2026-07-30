import xarray as xr
import zarr
import os
from dask.diagnostics import ProgressBar

fn = "era5_1979-2020janu_zstd_unified.zarr"
out_dirc = os.path.join(os.path.dirname(fn), 'era5_norm_stats')
if not os.path.exists(out_dirc):
    os.makedirs(out_dirc)

store = zarr.open(fn, mode='r')
print(store.info)
print(store.tree(expand=True))

surface_vars = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature','mean_sea_level_pressure',
]

pressure_vars = [
    'geopotential', 'relative_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind',
]


for var in surface_vars:
    
    ds= xr.open_zarr(fn, group = var)
    print(f'successfully opened {var} dataset')
    ds = ds.sel(time=slice('1979-01-01T00:00', '2015-12-31T23:00'))
    ds = ds[var]
    
    ds = ds.chunk({"time": 20})

    mean_per_level = ds.mean(dim=('time', 'latitude', 'longitude'))
    std_per_level = ds.std(dim=('time', 'latitude', 'longitude'))

    stats = xr.Dataset({"mean": mean_per_level, "std": std_per_level})

    print(f'hey, we are processing {var}')
    with ProgressBar():
        stats = stats.compute()
        
    # save and close the dataset
    our_fn = os.path.join(out_dirc, f"norm_stats_{var}.zarr")
    stats.to_zarr(our_fn,
                  mode='a', 
                  consolidated=True)
    print(f"Processed {var} and saved to {our_fn}")
    
for var in pressure_vars:
    ds= xr.open_zarr(fn, group = var)
    print(f'successfully opened {var} dataset')
    ds = ds.sel(time=slice('1979-01-01T00:00', '2015-12-31T23:00'))
    ds = ds[var]
    
    ds = ds.chunk({"time": 20})

    mean_per_level = ds.mean(dim=('time', 'latitude', 'longitude'))
    std_per_level = ds.std(dim=('time', 'latitude', 'longitude'))

    stats = xr.Dataset({"mean": mean_per_level, "std": std_per_level})

    print(f'hey, we are processing {var}')
    with ProgressBar():
        stats = stats.compute()
        
    # save and close the dataset
    our_fn = os.path.join(out_dirc, f"norm_stats_{var}.zarr")
    stats.to_zarr(our_fn,
                  mode='a', 
                  consolidated=True)
    print(f"Processed {var} and saved to {our_fn}")