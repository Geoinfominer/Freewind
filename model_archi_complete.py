from imports import *
from model_archi_encoder import calculate_best_embedding_dim
from model_archi_encoder import CompressPatch
from model_archi_GPT import GPT2Model
from model_archi_decoder import DecompressPatch




class ModelComplete(nn.Module):
    def __init__(
        self,
        max_seq_len_per_forward = 3,
        in_channels = 69,
        in_height = 721,
        in_width = 1440,
        patch_size = (16,32),
        embedding_dim = 4096,
        transformer_layers = 3,
        transformer_n_heads = 2,
        out_channels = 69,
        out_height = 721,
        out_width = 1440,
        vae = False,
        ):
        super(ModelComplete, self).__init__()
        
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

        # define encoder
        self.encoder = CompressPatch(
        in_channels = self.in_channels,
        embedding_dim = self.embedding_dim,
        in_height = self.in_height,
        in_width = self.in_width,
        patch_size = self.patch_size,
        conv_type = 'PeriodicConv2d',  # "PeriodicConv2d" or "Conv2d"
        send_skip_connection_to_decoder = True,
        )
        self.encoder.self_introduction(
            input_shape = (self.in_channels, self.in_height, self.in_width),
        )
        
        # define GPT-style transformer
        self.window_size = max_seq_len_per_forward * (self.in_height // self.patch_size[0]) * (self.in_width // self.patch_size[1])
        self.patches_one_step = (self.in_height // self.patch_size[0]) * (self.in_width // self.patch_size[1])
        self.transformer = GPT2Model(
            window_size = self.window_size,         
            n_embd = self.embedding_dim,
            n_layer = self.transformer_layers,
            n_head = self.transformer_n_heads,
            patches_one_step=self.patches_one_step,
            vae=self.vae,
            )
        
        # define decoder, pass in the details from the encoder
        self.decoder = DecompressPatch(
            final_out_channels=self.out_channels,
            final_out_height=self.out_height,
            final_out_width=self.out_width,
            details_encoder = self.encoder.configure(),
            receive_skip_connection_from_encoder = True,
            concat_skip_connection_from_encoder = True,
            conv_type = 'PeriodicConv2d', # "PeriodicConv2d" or "Conv2d"
            deconv_type = "PixelShuffle", # PixelShuffle, ConvTranspose2d
        )
        print('we have created encdoer, transformer and decoder!')
        
    def forward(self, x, past_kv=None):
        
        # x: (batch_size, seq_len, in_channels, in_height, in_width)
        assert len(x.shape) == 5, f"Input tensor must have 5 dimensions, but got {len(x.shape)}"
        batch_size = x.shape[0]
        
        #@ go through encoder
        output_encoder = self.encoder(x)
        
        x = output_encoder['output']
        x = rearrange(x, 'b s p e -> b (s p) e')
        
        #@ go through transformer
        # during training, the input past_kv is None for each forward pass; it will be a list of tensors after the following line, forgetting about it during training
        # during inference, the input past_kv is a None at the very beginning, and then it will be a list of tensors after the first forward pass, and then it will be a list of tensors after each forward pass as input
        x, past_kv, vae_paras = self.transformer(x,past = past_kv)
        
        x = rearrange(x, 'b (s p) e -> (b s) p e', p=self.patches_one_step)
        x = rearrange(x, 'b p e -> b e p')
        x = rearrange(x, 'b e (h w) -> b e h w', h=(self.in_height // self.patch_size[0]) , w=(self.in_width // self.patch_size[1]))
        
        #@ go through decoder
        x = self.decoder(x, output_encoder['skip_connection'])
        
        x = rearrange(x, '(b s) c h w -> b s c h w', b = batch_size)
        
        #
        return {
            'result':x,
            'past_kv_cache':past_kv,
            'vae_paras':vae_paras,
        }
        

if __name__ == '__main__':
    dtype = torch.float16
    batch_size = 1
    init_inut_seq_len = 3
    in_channels = 69
    nearest_neighbor_factor = 1
    in_height = 721 // nearest_neighbor_factor
    in_width = 1440 // nearest_neighbor_factor
    out_channels = in_channels
    out_height = in_height
    out_width = in_width
    patchsize = (16,32)
    
    compression_ratio_for_each_patch2embedding = 0.1
    embedding_dim = calculate_best_embedding_dim(in_channels, patchsize, compression_ratio_for_each_patch2embedding)
    print(f"Best embedding dimension: {embedding_dim}")
    
    model = ModelComplete(
        max_seq_len_per_forward = 5, # should always >= init_inut_seq_len
        in_channels = in_channels,
        in_height = in_height,
        in_width = in_width,
        patch_size = patchsize,
        embedding_dim = embedding_dim,
        transformer_layers = 8,
        transformer_n_heads = 4,
        out_channels = out_channels,
        out_height = out_height,
        out_width = out_width,
        ).to(dtype=dtype, device='cuda')
    
    # randomly create tensors
    patches_along_height = in_height // patchsize[0]
    patches_along_width = in_width // patchsize[1]
    source_input = torch.randn(batch_size, init_inut_seq_len, in_channels, in_height, in_width).to(dtype=dtype, device='cuda')
    print(f"source_input shape: {source_input.shape}")
    
    # number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of parameters (trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # here we go
    #  output = model(source_input)
    #  print(f"output shape: {output['result'].shape}")
    #  print(f"past_kv_cache of 1st transformer layer shape: {output['past_kv_cache'][0].shape}")
    
    
    # inference: 
    prev = source_input
    generations = []
    past_kv = None
    new_steps = 10
    patches_one_step = patches_along_height * patches_along_width
    
    with torch.no_grad():
        for n in trange(new_steps):
            print(f"this is the {n}th forward pass")
            
            # using current input IDs to go trough all transformer blocks
            result = model(prev, past_kv=past_kv)
            output_img, past_kv = result['result'], result['past_kv_cache']
            window_size = model.window_size
            # sliding window
            if (past_kv[0].shape[3] // patches_one_step) == (window_size // patches_one_step):
                for lyr in range(len(past_kv)):
                    # past_kv is a list of [2, B, nh, past_S*patches_one_step, hd]      
                    past_kv[lyr] = rearrange(past_kv[lyr], 'kv b nh (s p) d -> kv b nh s p d', kv=2, p=patches_one_step)
                    past_kv[lyr] = past_kv[lyr][:, :, :, 1:, :, :]
                    past_kv[lyr] = rearrange(past_kv[lyr], 'kv b nh s p d -> kv b nh (s p) d',kv=2,p=patches_one_step)    
            
            # only need the last step of the output embeddings for next prediction
            output_img = output_img[:, -1:, :, :, :]
            prev = output_img
            
            