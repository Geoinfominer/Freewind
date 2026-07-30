from imports import *
from model_archi_complete import ModelComplete

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, decay_steps, max_lr, min_lr, last_epoch=-1):
        # override the __init__ function
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        # override the get_lr function
        step = self.last_epoch # should + 1
        if step < self.warmup_steps:
            lr = (step / self.warmup_steps) * self.max_lr
        else:
            progress = (step - self.warmup_steps) / self.decay_steps
            if progress < 1.0:
                # lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.min_lr + cosine_decay * (self.max_lr - self.min_lr)
            else:
                lr = self.min_lr
        return [lr for _ in self.optimizer.param_groups]


class Model_pl(pl.LightningModule):
    def __init__(
        self,
        max_seq_len_per_forward = 5,      # should always >= init_inut_seq_len
        in_channels = 69,                 # input image channel
        in_height = 721,                  # input image height
        in_width = 1440,                  # input image width
        patch_size = (16,32),             # patch size
        embedding_dim = 4096,             # embedding dimension
        transformer_layers = 3,           # number of transformer layers
        transformer_n_heads = 2,          # number of transformer heads
        out_channels = 69,                # output image channel
        out_height = 721,                 # output image height
        out_width = 1440,                 # output image width
        vae = False,                      # whether to use VAE
        lr = 0.0001,                      # learning rate
        *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        # add new parameters
        self.max_seq_len_per_forward = max_seq_len_per_forward
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.transformer_layers = transformer_layers
        self.transformer_n_heads = transformer_n_heads
        self.out_channels = out_channels
        self.out_height = out_height
        self.out_width = out_width
        self.vae = vae
        self.lr = lr
        
        # model
        self.model = ModelComplete(
        max_seq_len_per_forward = self.max_seq_len_per_forward,
        in_channels = self.in_channels,
        in_height = self.in_height,
        in_width = self.in_width,
        patch_size = self.patch_size,
        embedding_dim = self.embedding_dim,
        transformer_layers = self.transformer_layers,
        transformer_n_heads = self.transformer_n_heads,
        out_channels = self.out_channels,
        out_height = self.out_height,
        out_width = self.out_width,
        vae = self.vae,
        )
        
    # override configure_optimizers function
    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=1, verbose=True)
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_loss_epoch'} # monitor is used for scheduler which adjusts lr based on validation performance
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = WarmupCosineScheduler(
            optimizer = optimizer,
            warmup_steps = 3000,
            decay_steps = 297000,
            max_lr = self.lr,
            min_lr = 1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # very important! it's step instead of epoch
            }
        }
        

    # override forward function
    def forward(self, x, past_kv=None):
        return self.model(x, past_kv = past_kv)
    
    def vae_loss(recons_tensor,true_tensor,q_mu,q_log_var,kld_weight=0.01):
        # Computes the VAE loss function.
        # KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        recons_loss =F.mse_loss(recons_tensor, true_tensor)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + q_log_var - q_mu ** 2 - q_log_var.exp(), dim = 1), dim = 0)  
        loss = recons_loss + kld_weight * kld_loss
        return loss
    
    # override training_step function
    def training_step(self, batch, batch_idx):
        
        if batch_idx % 20 == 0 and self.trainer.is_global_zero:
            os.system('nvidia-smi')
        
        # define what are executed in a batch loop during training
        x,y = batch
        result = self.forward(x)
        # compute loss
        if self.vae:
            loss = self.vae_loss(result['result'], y, result['vae_paras'][0], result['vae_paras'][1])
        else:
            loss = F.l1_loss(result['result'], y)
        # progress bar showing training loss of each batch for the main process in real time
        self.log(
            "train_loss", loss,
            on_step=True, on_epoch=False, # true ons_step means no aggregation, and shows in progress bar in real time
            prog_bar=True, logger=False, # show the real time loss in progress bar, but not show in tensorboard
            sync_dist=False, # false sync_dist means only showing the loss of the main process, not all processes
        )
        # tensorboard showing training loss of each epoch for all processes
        self.log(
            "train_loss_epoch", loss,
            on_step=False, on_epoch=True, # true on_epoch means aggregation at the end of each epoch
            prog_bar=False, logger=True, # show the aggregated loss among each batch in tensorboard, but not show in progress bar
            sync_dist=True, # true sync_dist means showing the mean aggregated loss of all processes
        )
        
        return loss # so so so important, otherwise the training_step will return None, and the weights and biases will not be updated !!!
    
    # override validation_step function
    def validation_step(self, batch, batch_idx):
        x,y = batch
        result = self.forward(x)
        if self.vae:
            loss = self.vae_loss(result['result'], y, result['vae_paras'][0], result['vae_paras'][1])
        else:
            loss = F.l1_loss(result['result'], y)
        self.log(
            "valid_loss", loss,
            on_step=True, on_epoch=False, 
            prog_bar=True, logger=False, 
            sync_dist=False,
        )
        self.log(
            "valid_loss_epoch", loss,
            on_step=False, on_epoch=True, 
            prog_bar=False, logger=True, 
            sync_dist=True,
        )
    
    # override predict_step function
    def predict_step(self, batch, batch_idx):
        pass
    

if __name__ == "__main__":
    # model, initialize
    model = Model_pl(
        max_seq_len_per_forward = 5,      # >= init_inut_seq_len
    )
    # get the control_unet
    dtype = torch.float16
    model = model.model.to(dtype=dtype, device='cuda')
    
    # check # of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print('#paras by numel:', total_params)

    #
    batch_size = 1
    init_inut_seq_len = 5 # <= max_seq_len_per_forward
    in_channels = 69
    in_height = 721 // 1
    in_width = 1440 // 1
    source_input = torch.randn(batch_size, init_inut_seq_len, in_channels, in_height, in_width).to(dtype=dtype, device='cuda')
    out = model(source_input)
    out['result'].sum().backward()  # 43061 Mb
    print(out['result'].shape) 