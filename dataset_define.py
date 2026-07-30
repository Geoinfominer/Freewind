from imports import *

class ERA_Dataset(Dataset):
    def __init__(
        self,
        dir_ERA: str = "era5_1979-2020janu_zstd_unified.zarr",
        whichPhase: str = "train", # train, validation, testing, prediction
        num_steps: int = 3,        # number of steps >= 2
        resample_scale: int = 1,   # resample scale, 1 means no resample
        dtype = torch.float16,
        *args, **kwargs
    ):
        super().__init__()
        
        # base info
        self.whichPhase = whichPhase
        self.dir_ERA = dir_ERA
        self.num_steps = num_steps
        self.resample_scale = resample_scale
        self.dtype = dtype
        self.delta_time = 6           # hours
        self.vars_pressure = [
            "u", "v", "z", "t", "r",
        ]
        self.vars_surface = [
            "10m_u", "10m_v", "2m_t", "msl",
        ]
        self.vars_ds = {
            "u": xr.open_zarr(self.dir_ERA, group='u_component_of_wind'),
            "v": xr.open_zarr(self.dir_ERA, group='v_component_of_wind'),
            "z": xr.open_zarr(self.dir_ERA, group='geopotential'),
            "t": xr.open_zarr(self.dir_ERA, group='temperature'),
            "r": xr.open_zarr(self.dir_ERA, group='relative_humidity'),
            "10m_u": xr.open_zarr(self.dir_ERA, group='10m_u_component_of_wind'),
            "10m_v": xr.open_zarr(self.dir_ERA, group='10m_v_component_of_wind'),
            "2m_t": xr.open_zarr(self.dir_ERA, group='2m_temperature'),
            "msl": xr.open_zarr(self.dir_ERA, group='mean_sea_level_pressure'),
        }
        
        # !!! our current dataset consists of 2 parts, the first part is from 1979-01-01 to 2009-12-31 with a temporary time step of 6 hours, and the second part is from 2010-01-01 to 2015-12-31 with a temporary time step of 1 hour. !!!
        t0, t1 = self.calculate_init_time('1979-01-01T00:00', '2009-12-31T18:00', self.num_steps, self.delta_time)
        self.init_time_traning_0 = self.vars_ds['10m_u'].sel(time=slice(t0, t1)).time.values
        t0, t1 = self.calculate_init_time('2010-01-01T00:00', '2015-12-31T23:00', self.num_steps, self.delta_time)
        self.init_time_traning_1 = self.vars_ds['10m_u'].sel(time=slice(t0, t1)).time.values
        self.init_time_traning = np.concatenate((self.init_time_traning_0, self.init_time_traning_1), axis=0)
        # 
        t0, t1 = self.calculate_init_time('2016-01-01T00:00', '2017-12-31T23:00', self.num_steps, self.delta_time)
        self.init_time_validation = self.vars_ds['10m_u'].sel(time=slice(t0, t1)).time.values
        # 
        t0, t1 = self.calculate_init_time('2018-01-01T00:00', '2020-12-31T23:00', self.num_steps, self.delta_time)
        self.init_time_testing = self.vars_ds['10m_u'].sel(time=slice(t0, t1)).time.values
         
        # load the normalization parameters of ERA
        self.norm = True
        if self.norm:
            self.norm_ds = {
                "u": xr.open_zarr(os.path.join(os.path.dirname(self.dir_ERA),'era5_norm_stats',f'norm_stats_u_component_of_wind.zarr')),
                "v": xr.open_zarr(os.path.join(os.path.dirname(self.dir_ERA),'era5_norm_stats',f'norm_stats_v_component_of_wind.zarr')),
                "z": xr.open_zarr(os.path.join(os.path.dirname(self.dir_ERA),'era5_norm_stats',f'norm_stats_geopotential.zarr')),
                "t": xr.open_zarr(os.path.join(os.path.dirname(self.dir_ERA),'era5_norm_stats',f'norm_stats_temperature.zarr')),
                "r": xr.open_zarr(os.path.join(os.path.dirname(self.dir_ERA),'era5_norm_stats',f'norm_stats_relative_humidity.zarr')),
                "10m_u": xr.open_zarr(os.path.join(os.path.dirname(self.dir_ERA),'era5_norm_stats',f'norm_stats_10m_u_component_of_wind.zarr')),
                "10m_v": xr.open_zarr(os.path.join(os.path.dirname(self.dir_ERA),'era5_norm_stats',f'norm_stats_10m_v_component_of_wind.zarr')),
                "2m_t": xr.open_zarr(os.path.join(os.path.dirname(self.dir_ERA),'era5_norm_stats',f'norm_stats_2m_temperature.zarr')),
                "msl": xr.open_zarr(os.path.join(os.path.dirname(self.dir_ERA),'era5_norm_stats',f'norm_stats_mean_sea_level_pressure.zarr')),
            }
        
    def calculate_init_time(self, first_time, last_time, num_steps, delta_time):
        t0 = pd.to_datetime(first_time) + pd.Timedelta(hours=(num_steps - 1)*delta_time)
        t0 = t0.strftime("%Y-%m-%dT%H:%M")
        t1 = pd.to_datetime(last_time) - pd.Timedelta(hours=(num_steps - 1 + 1)*delta_time) #  +1 is for target
        t1 = t1.strftime("%Y-%m-%dT%H:%M")
        return t0, t1
        
    def __len__(self):
        if self.whichPhase == "train":
            return len(self.init_time_traning)
        elif self.whichPhase == "validation":
            return len(self.init_time_validation)
        elif self.whichPhase == "testing":
            return len(self.init_time_testing)
    
    def __getitem__(self, idx):       
        
        #@ load input data
        
        # select a init time
        if self.whichPhase == "train":
            init_time = self.init_time_traning[idx]
        elif self.whichPhase == "validation":
            init_time = self.init_time_validation[idx]
        elif self.whichPhase == "testing":
            init_time = self.init_time_testing[idx]
        
        # time series for input, (init_time-delta_time*(num_steps-1), init_time*delta_time)
        time_series = pd.date_range(
            end=init_time,
            periods=self.num_steps,
            freq=f"{self.delta_time}h"
        )
        time_series = time_series.strftime("%Y-%m-%dT%H:%M").values
        
        # loop through the pressure variables to retrieve the data in the time series, lazily
        data_all = []
        for var in self.vars_pressure:
            # select the time series
            da = self.vars_ds[var].sel(time=time_series)
            if 'valid_time' in da.dims:
                da = da.rename({'valid_time': 'time'})
            # very important, you can only concat dataarray instead of dataset,so you have to excute: da = da['variable_name']
            assert len(list(da.var())) == 1, f"More than one variable in {var} dataset"
            da = da[list(da.var())[0]]
            # regularize
            da = da.rename({'level': 'channel'})
            da = da.transpose('time', 'channel', 'latitude', 'longitude')
            da = da.assign_coords(channel = [f'{var}_{i}' for i in da['channel'].values] ) 
            # normalize
            if self.norm:
                assert [f'{var}_{i}' for i in self.norm_ds[var]['mean']['level'].values] == [str(i) for i in da['channel'].values], f"the order of the level in {var} dataset is not the same as the order of the level in norm dataset"
                mean_value = self.norm_ds[var]['mean'].rename({'level': 'channel'}).assign_coords(channel=da.channel)
                std_value = self.norm_ds[var]['std'].rename({'level': 'channel'}).assign_coords(channel=da.channel)
                da = (da - mean_value) / std_value
            # append the dataarray, not dataset, to the list
            data_all.append(da)
        
        # loop through the surface variables to retrieve the data in the time series, lazily
        for var in self.vars_surface:
            # select the time series
            da = self.vars_ds[var].sel(time=time_series)
            if 'valid_time' in da.dims:
                da = da.rename({'valid_time': 'time'})
            # very important, you can only concat dataarray instead of dataset,so you have to excute: da = da['variable_name']
            assert len(list(da.var())) == 1, f"More than one variable in {var} dataset"
            da = da[list(da.var())[0]]
            # regularize
            da = da.expand_dims(dim='channel', axis=1)
            da = da.transpose('time', 'channel', 'latitude', 'longitude')
            da = da.assign_coords(channel = [var])
            # normalize
            if self.norm:
                mean_value = self.norm_ds[var]['mean']
                std_value = self.norm_ds[var]['std']
                da = (da - mean_value) / std_value
            # append the dataarray, not dataset, to the list
            data_all.append(da)

        # combine all the variables
        input_xr = xr.concat(data_all, dim='channel')

        # clip or resample by nearest neighbor before computing
        if self.resample_scale > 1:
            input_xr = input_xr.isel(
            latitude=slice(None, None, self.resample_scale),
            longitude=slice(None, None, self.resample_scale)
            )
        
        # compute
        arr = input_xr.compute().values  # Dask to NumPy
        input_tensor = torch.tensor(
            arr,                       
            dtype=self.dtype,
        )                                # NumPy to Torch
        # input_tensor = input_tensor.unsqueeze(0)

        #@ load the target data, similar to the input data
        
        # add self.delta_time to each element in time_series, and only keep the last element (init_time + delta_time)
        time_series = pd.to_datetime(time_series) + pd.Timedelta(hours=self.delta_time)
        time_series = time_series.strftime("%Y-%m-%dT%H:%M").values[-1:]
        
        # as above
        data_all = []
        for var in self.vars_pressure:
            # select the time series
            da = self.vars_ds[var].sel(time=time_series)
            if 'valid_time' in da.dims:
                da = da.rename({'valid_time': 'time'})
            # very important, you can only concat dataarray instead of dataset,so you have to excute: da = da['variable_name']
            assert len(list(da.var())) == 1, f"More than one variable in {var} dataset"
            da = da[list(da.var())[0]]
            # regularize
            da = da.rename({'level': 'channel'})
            da = da.transpose('time', 'channel', 'latitude', 'longitude')
            da = da.assign_coords(channel = [f'{var}_{i}' for i in da['channel'].values] ) 
            # normalize
            if self.norm:
                assert [f'{var}_{i}' for i in self.norm_ds[var]['mean']['level'].values] == [str(i) for i in da['channel'].values], f"the order of the level in {var} dataset is not the same as the order of the level in norm dataset"
                mean_value = self.norm_ds[var]['mean'].rename({'level': 'channel'}).assign_coords(channel=da.channel)
                std_value = self.norm_ds[var]['std'].rename({'level': 'channel'}).assign_coords(channel=da.channel)
                da = (da - mean_value) / std_value
            # append the dataarray, not dataset, to the list
            data_all.append(da)
        
        for var in self.vars_surface:
            # select the time series
            da = self.vars_ds[var].sel(time=time_series)
            if 'valid_time' in da.dims:
                da = da.rename({'valid_time': 'time'})
            # very important, you can only concat dataarray instead of dataset,so you have to excute: da = da['variable_name']
            assert len(list(da.var())) == 1, f"More than one variable in {var} dataset"
            da = da[list(da.var())[0]]
            # regularize
            da = da.expand_dims(dim='channel', axis=1)
            da = da.transpose('time', 'channel', 'latitude', 'longitude')
            da = da.assign_coords(channel = [var])
            # normalize
            if self.norm:
                mean_value = self.norm_ds[var]['mean']
                std_value = self.norm_ds[var]['std']
                da = (da - mean_value) / std_value
            data_all.append(da)

        target_xr = xr.concat(data_all, dim='channel')

        if self.resample_scale > 1:
            target_xr = target_xr.isel(
            latitude=slice(None, None, self.resample_scale),
            longitude=slice(None, None, self.resample_scale)
            )
        
        arr = target_xr.compute().values
        target_tensor = torch.tensor(
            arr,                       
            dtype=self.dtype,
        )                      
        # target_tensor = target_tensor.unsqueeze(0)
        
        # concat input_tensor[1,1:,C,H,W] with target_tensor[1,1,C,H,W] along the time dimension to get the final target_tensor, for causal attention
        target_tensor = torch.cat(
            [
                input_tensor[1:, :, :, :], 
                target_tensor[:, :, :, :]
            ],
            dim=0
        )
        
        #@ return :[N,T,C,H,W]
        return input_tensor, target_tensor - input_tensor


if __name__ == "__main__":
    
    ds = ERA_Dataset(
        dir_ERA="era5_1979-2020janu_zstd_unified.zarr",
        whichPhase="train",
        num_steps=3,
    )
    for i in range(10):
        input, target = ds[i]
        print(input.shape, target.shape)