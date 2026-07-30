from imports import *
from dataset_define import ERA_Dataset

# the class and fucntions replaced the following steps in ordinary pytorch codes
    # train_ds = ERA_Dataset(...)
    # valid_ds = ERA_Dataset(...)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(...)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(...)
    # train_batch_sampler = torch.utils.data.BatchSampler(...)
    # valid_batch_sampler = torch.utils.data.BatchSampler(...)
    # train_loader = torch.utils.data.DataLoader(...)
    # valid_loader = torch.utils.data.DataLoader(...)

class ERA_Dataset_pl(pl.LightningDataModule):
    def __init__(
        self, 
        dir_ERA: str = "era5_1979-2020janu_zstd_unified.zarr",
        num_steps: int = 3,        # number of steps >= 2
        resample_scale: int = 1,   # resample scale, 1 means no resample
        dtype = torch.float16,
        batch_size: int = 32,
        num_workers: int = 4,
        *args, **kwargs
    ):
        super().__init__()
        self.dir_ERA = dir_ERA
        self.num_steps = num_steps
        self.resample_scale = resample_scale
        self.dtype = dtype
        self.batch_size = batch_size
        self.num_workers = num_workers
        

    def prepare_data(self):
        # there is no need for downloading data
        pass
        
    def setup(self,stage:"fit,validate,test,predict"):
        
        if stage == "fit":
            # including training and validation
            self.train_ds = ERA_Dataset(
                dir_ERA = self.dir_ERA, 
                whichPhase = "train",                   # train, validation, testing, prediction
                num_steps = self.num_steps,             # number of steps >= 2
                resample_scale = self.resample_scale,   # resample scale, 1 means no resample
                dtype = self.dtype,
            )
            print('training set is initialized')
            self.valid_ds = ERA_Dataset(
                dir_ERA = self.dir_ERA, 
                whichPhase = "validation",              # train, validation, testing, prediction
                num_steps = self.num_steps,             # number of steps >= 2
                resample_scale = self.resample_scale,   # resample scale, 1 means no resample
                dtype = self.dtype,
            )
            print('validation set is initialized')
        
        if stage == 'test':
            pass
        
        if stage == 'predict':
            pass

    def train_dataloader(self):
        print('train_dataloader is created')
        return DataLoader(
            self.train_ds,
            batch_size = self.batch_size,
            pin_memory=True,
            num_workers = self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        print('val_dataloader is created')
        return DataLoader(
            self.valid_ds,
            batch_size = self.batch_size,
            pin_memory=True,
            num_workers = self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass