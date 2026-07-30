from imports import *
from model_main import Model_pl
from dataset_main import ERA_Dataset_pl
import configs
import custom_callbacks

if __name__ == "__main__":
    # model, initialize
    model_moudle = Model_pl(
        max_seq_len_per_forward = configs.max_seq_len_per_forward,       
        in_channels = configs.in_channels,           
        in_height = configs.in_height,
        in_width = configs.in_width,
        patch_size = configs.patch_size,
        embedding_dim = configs.embedding_dim,
        transformer_layers = configs.transformer_layers,
        transformer_n_heads = configs.transformer_n_heads,
        out_channels = configs.out_channels,
        out_height = configs.out_height,
        out_width = configs.out_width,  
        vae = configs.vae,
        lr = configs.lr,
    )
    # load checkpoint
    if configs.load_pretrained_weight_file:
        model_moudle = custom_callbacks.load_ckpt(model_moudle, configs.pretrained_weight_file, variable_name='state_dict')

    # dataset
    data_moudle = ERA_Dataset_pl(
        dir_ERA = configs.dir_ERA,
        num_steps = configs.num_steps, 
        resample_scale = configs.resample_scale,
        dtype = configs.dtype,
        batch_size = configs.batch_size,
        num_workers = configs.num_workers,
    )
    
    # logger
    logger=TensorBoardLogger(configs.parent_path + "cps/tb_logs", 
                            name=configs.longname)
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='step') # step or epoch
    # profiler
    if configs.enable_profiler:
        rank = dist.get_rank() if dist.is_initialized() else 0
        tb_log_dir = logger.log_dir
        profiler = PyTorchProfiler(
        dirpath = tb_log_dir,
        filename = f"profiler-{configs.longname}-{rank}",
        record_shapes = True,
        profile_memory = True,
        use_cuda = True,
        export_to_chrome = True,
        record_functions = True, # for specific functions
        schedule=schedule(
            wait=3,      
            warmup=3,  
            active=4,  
            repeat=1,
        ),
        on_trace_ready = tensorboard_trace_handler(
            dir_name = tb_log_dir,
            worker_name = f"worker_{configs.longname}-{rank}",
            use_gzip=True,
        )
        )

    # checkpoint, https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint
    checkpoint_callback0 = callbacks.ModelCheckpoint(
        dirpath= configs.parent_path + "cps/",
        filename= 'latest_' + configs.longname + ',' + '{epoch}',
        save_last=True,
        verbose = True,
        save_weights_only  = False,
        every_n_epochs = 1
    )
    checkpoint_callback1 = callbacks.ModelCheckpoint(
        dirpath=configs.parent_path + "cps/",
        filename='best_' + configs.longname + ',' + '{epoch}',
        monitor="valid_loss_epoch",
        mode="min",
        save_top_k = 1,
        verbose = True,
        save_weights_only  = False,
        every_n_epochs = 1
    )

    # other callbacks
    pass
    
    # trainer, https://pytorch-lightning.readthedocs.io/en/1.2.10/api/pytorch_lightning.trainer.trainer.html#pytorch_lightning.trainer.trainer.Trainer
    trainer = pl.Trainer(
        strategy = 'ddp',
        precision = '16-mixed' if configs.dtype == torch.float16 else 32,          # forward passes in 16-bit, optimizer states in 32-bit 
        logger = logger,
        profiler=profiler if configs.enable_profiler else None,
        accelerator=configs.accelerator,
        devices=configs.devices,
        min_epochs=configs.epoch,
        max_epochs=configs.epoch,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.5,
        callbacks = [
            checkpoint_callback0,
            checkpoint_callback1,
            lr_monitor,
        ],
    )
    if configs.fit_or_predict == 'fit':
        trainer.fit(
            model = model_moudle,
            datamodule = data_moudle,
        )
    elif configs.fit_or_predict == 'predict':
        pass
    