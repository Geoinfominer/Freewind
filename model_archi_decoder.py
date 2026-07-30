from imports import *

class GeneralizedPixelShuffle(nn.Module):
    def __init__(
        self,
        in_channels = 8192,
        out_channels = 7220,
        kernel_size = 2,
        stride = 2,
        padding = 0,
    ):
        '''
        torch.nn.PixelShuffle(r=p) will reshape an input tensor from (n, p1*p2*c, h, w) into (n, c, p1*h, p2*w).
        It is equivalent to:
        output = einops.rearrange(input, 'n (c p1 p2) h w -> n c (h p1) (w p2)', p1=r, p2=r).
        But in real scenarios, the input tensor may not be able to be rearranged directly, so it needs to be streched first.
        And remember, we do not exchange information between patches in decoder, just like in encoder, we do not exchange information between patches as well.
        So, to stretch the input tensor, we should not use conv2d to adjust the number of channels while remaining the spatial size, but use linear transform to adjust the number of channels.
        '''
        super(GeneralizedPixelShuffle, self).__init__()
        assert kernel_size == stride, "kernel_size must be equal to stride in GeneralizedPixelShuffle !"
        assert padding == (0,0), "padding must be 0 in GeneralizedPixelShuffle !"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.linear = nn.Linear(in_channels, out_channels * kernel_size[0] * kernel_size[1])
        
    def forward(self, x):
        # make sure x has 4 dimensions: (b, C1, h, w)
        assert len(x.shape) == 4, "x must have 4 dimensions: (batchsize, channels, height, width) !"
        hight = x.shape[2]
        # rearange the input tensor from (b,C1,h,w) to (b,h*w,C1)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # apply linear transform to adjust the number of channels, from C1 to C2, C2 = c*k*k
        x = self.linear(x)
        # pixelshuffle: rearange back the output tensor from (b,h*w,c*k*k) to (b,c*k*k,h,w), and then to (b,c,h*k,w*k)
        # these 2 steps may be combined into one step, but for better understanding, we separate them.
        x = rearrange(x, 'b (h w) (c k1 k2) -> b (c k1 k2) h w', k1=self.kernel_size[0], k2=self.kernel_size[1], h=hight)
        x = rearrange(x, 'b (c k1 k2) h w -> b c (h k1) (w k2)', k1=self.kernel_size[0], k2=self.kernel_size[1], h=hight)
        return x
    
class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        """
        A PyTorch nn.Module wrapper around F.interpolate.
        Args:
            size (tuple or None): Output size (H, W).
            scale_factor (float or tuple or None): Multiplicative factor for spatial size.
            mode (str): Algorithm used for upsampling ('nearest', 'bilinear', etc.).
            align_corners (bool): Only relevant for mode='linear', 'bilinear', 'bicubic' or 'trilinear'.
        """
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners
        )

# define a class of PeridicalConv layer inherited from orch.nn.Conv2d
class PeriodicConv2d(torch.nn.Conv2d):
    """
    This class is from Fuxi's source codes.
    A class that implements a periodic convolution layer.
    This layer pads the input tensor with circular padding in the width dimension
    and constant padding in the height dimension.
    The padding is applied before the convolution operation.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize parent Conv2d with given args and kwargs
        super().__init__(*args, **kwargs) 
        # Ensure padding is greater than 0
        assert max(self.padding) >= 0

    def forward(self, x):
        # Input x shape: (batch, channels, height, width) 
        # Pad horizontally with circular mode (periodic in width)
        # (batch, channels, height, width) -> (batch, channels, height, padding[1]+width+padding[1])
        x = F.pad(x, (self.padding[1], self.padding[1], 0, 0), mode="circular")
        # (batch, channels, height, padding[1] + width + padding[1]) -> (batch, channels, padding[0]+height+padding[0], padding[1]+width+padding[1])
        x = F.pad(x, (0, 0, self.padding[0], self.padding[0]), mode="constant", value=0)
        # Apply convolution, removing padding effect via stride and kernel size
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups  
        )
        return x

class PermuteLayerNorm2d(nn.Module):
    """
    LayerNorm for 4D tensors with shape [B, C, H, W].
    Normalizes over the channel (embedding) dimension.
    This layer doesn't view tensor with shape [B, C, H, W] as image, but H*W embeddings with C dimensions.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x

def convert_config_from_encoder_to_decoder(config_conv_encoder):
    # get conv layer config in encoder
    conv_config_encoder = config_conv_encoder[1]
    in_channels_encoder = int(conv_config_encoder.split('(')[1].split(',')[0])
    out_channels_encoder = int(conv_config_encoder.split('(')[1].split(',')[1])
    kernel_size_encoder = int(conv_config_encoder.split('kernel_size=(')[1].split(',')[0]), int(re.search(r'\d+', conv_config_encoder.split('kernel_size=(')[1].split(',')[1]).group())
    stride = int(conv_config_encoder.split('stride=(')[1].split(',')[0]), int(re.search(r'\d+', conv_config_encoder.split('stride=(')[1].split(',')[1]).group())
    if 'padding=' in conv_config_encoder:
        padding = int(conv_config_encoder.split('padding=(')[1].split(',')[0]), int(re.search(r'\d+', conv_config_encoder.split('padding=(')[1].split(',')[1]).group())
    else:
        padding = 0,0
    # convert the config to decoder
    deconv_config_decoder = {}
    deconv_config_decoder['in_channels'] = out_channels_encoder
    deconv_config_decoder['out_channels'] = in_channels_encoder
    deconv_config_decoder['kernel_size'] = kernel_size_encoder
    deconv_config_decoder['stride'] = stride
    deconv_config_decoder['padding'] = padding
    return deconv_config_decoder
    
class DecompressPatch(nn.Module):
    def __init__(
      self, 
      final_out_channels = 69,
      final_out_height = 721,
      final_out_width = 1440,
      details_encoder = None,
      receive_skip_connection_from_encoder = True,
      concat_skip_connection_from_encoder = True,
      conv_type = 'PeriodicConv2d', # "PeriodicConv2d" or "Conv2d"
      deconv_type = "ConvTranspose2d", # PixelShuffle, ConvTranspose2d
    ):
        '''
        DecompressPatch just mirrors the CompressPatch class, but in reverse. 
        Remember the concept of decompression:
            - decompress single embedding to several adjacent embeddings.
            - Because in encoder, we compress several adjacent embeddings to a single patch embedding.
        It takes the output of transformer and the skip connection from encoder, and reconstructs the original image.
        The skip connection from encoder is a list of tensors, each tensor is the output of the encoder at different levels.
        The output of the transformer is a tensor of shape (batch_size, seq_len * patches_along_height * patches_along_width, embedding_dim).
        '''
        super(DecompressPatch, self).__init__()
        #
        self.receive_skip_connection_from_encoder = receive_skip_connection_from_encoder
        self.concat_skip_connection_from_encoder = concat_skip_connection_from_encoder
        self.conv_type = conv_type
        self.deconv_type = deconv_type
        #
        self.final_out_channels = final_out_channels
        self.final_out_height = final_out_height
        self.final_out_width = final_out_width
        #
        self.details_encoder = details_encoder
        self.details_encoder_keys = list(self.details_encoder.keys())
        self.details_encoder_keys.sort()
        self.details_encoder_keys_brach1 = self.details_encoder_keys[1:]
        self.details_encoder_keys_brach0 = self.details_encoder_keys[0]
        # 
        self.concat_to_merge_branches = True
        
        # each deconv for upsampling doesn't need padding, so there is no PeriodicConvTranspose2d and PeriodicPixelShuffle
        if deconv_type == "ConvTranspose2d":
            self.deconv = nn.ConvTranspose2d
        elif deconv_type == "PixelShuffle":
            self.deconv = GeneralizedPixelShuffle
        # some conv layers in decoder may need padding, so we need to define PeriodicConv2d
        if conv_type == "PeriodicConv2d":
            self.conv = PeriodicConv2d
        else:
            self.conv = nn.Conv2d
        
        #-- we have 2 branches in decoder, now, first process branch1 (progressively) and then branch0 (directly) --#
                
        #@ branch 1
        
        self.branch1 = nn.ModuleList()
        for i in range(len(self.details_encoder_keys_brach1)):
            # get the config detail of the encoder at this level, using pop, we define the layers in reverse order
            key = self.details_encoder_keys_brach1.pop()
            config_conv_encoder = self.details_encoder[key]
            # convert the detail to that of decoder
            config_deconv_decoder = convert_config_from_encoder_to_decoder(config_conv_encoder)
            # create a sequential and append to the branch1
            self.branch1.append(nn.Sequential(
                PermuteLayerNorm2d(config_deconv_decoder['in_channels'] * 2 if (receive_skip_connection_from_encoder and concat_skip_connection_from_encoder) else config_deconv_decoder['in_channels']),
                self.deconv(
                    in_channels=config_deconv_decoder['in_channels'] * 2 if (receive_skip_connection_from_encoder and concat_skip_connection_from_encoder) else config_deconv_decoder['in_channels'],
                    out_channels=config_deconv_decoder['out_channels'], 
                    kernel_size=config_deconv_decoder['kernel_size'],
                    stride=config_deconv_decoder['stride'],
                    padding=config_deconv_decoder['padding'],
                ),
                nn.GELU(),
            ))
            
        #@ branch 0
        
        # get the config detail of the encoder at this level
        key = self.details_encoder_keys_brach0
        config_conv_encoder = self.details_encoder[key]
        # convert the detail to the decoder
        config_deconv_decoder = convert_config_from_encoder_to_decoder(config_conv_encoder)
        # create a sequential and append to the branch0
        self.branch0 = nn.Sequential(
            PermuteLayerNorm2d(config_deconv_decoder['in_channels'] * 2 if (receive_skip_connection_from_encoder and concat_skip_connection_from_encoder) else config_deconv_decoder['in_channels']),
            self.deconv(
                in_channels=config_deconv_decoder['in_channels'] * 2 if (receive_skip_connection_from_encoder and concat_skip_connection_from_encoder) else config_deconv_decoder['in_channels'],
                out_channels=config_deconv_decoder['out_channels'], 
                kernel_size=config_deconv_decoder['kernel_size'],
                stride=config_deconv_decoder['stride'],
                padding=config_deconv_decoder['padding'],
            ),
            nn.GELU(),
        )
        
        #@ interpolate
        self.interpolate = Interpolate(
            size=(self.final_out_height, self.final_out_width),
            scale_factor=None,
            mode='bilinear',
            align_corners=False
        )
     
        #@ merge the branches, do not change high and width
        self.merge = nn.Sequential(
            PermuteLayerNorm2d(self.final_out_channels * 2 if self.concat_to_merge_branches else self.final_out_channels),
            self.conv(
                in_channels=self.final_out_channels * 2 if self.concat_to_merge_branches else self.final_out_channels,
                out_channels=self.final_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GELU(),
            PermuteLayerNorm2d(self.final_out_channels),
            self.conv(
                in_channels=self.final_out_channels,
                out_channels=self.final_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
    
    def forward(self, 
                output_transformer,
                skip_connection_from_encoder):
        # before the forward, transformer's output shape shold have been rearranged to (batch_size * seq_len, embedding_dim, patches_along_height, patches_along_width)
        
        x = output_transformer
        if self.receive_skip_connection_from_encoder:
            assert skip_connection_from_encoder is not None, "skip_connection_from_encoder must be provided when receive_skip_connection_from_encoder is True !"
        
        #@ branch1
        
        assert len(self.branch1) == len(skip_connection_from_encoder) - 1, " the length of the defined branch1 of decoder is not equal to the length of the skip_connection_from_encoder - 1 !" 
        # pass into the layers of branch1
        for i in range(len(self.branch1)):
            # get the skip connection from encoder if needed 
            if self.receive_skip_connection_from_encoder:
                skip_connection = skip_connection_from_encoder.pop()
                if self.concat_skip_connection_from_encoder:
                    x = torch.cat((x, skip_connection), dim=1)
                else:
                    x = x + skip_connection
            # pass into the layer of branch1
            x = self.branch1[i](x)
        # interpolate
        x = self.interpolate(x)
        
        #@ branch0
        
        # get the skip connection from encoder if needed
        if self.receive_skip_connection_from_encoder:
            skip_connection = skip_connection_from_encoder.pop()
            if self.concat_skip_connection_from_encoder:
                output_transformer = torch.cat((output_transformer, skip_connection), dim=1)
            else:
                output_transformer = output_transformer + skip_connection
        # pass into the layer of branch0
        y = self.branch0(output_transformer)
        # interpolate
        y = self.interpolate(y)
        
        #@ merge the branches
        
        if self.concat_to_merge_branches:
            x = torch.cat((x, y), dim=1)
        else:
            x = x + y
        x = self.merge(x)
        
        return x
    

if __name__ == "__main__":
    
    from model_archi_encoder import CompressPatch
    from model_archi_encoder import calculate_best_embedding_dim
    
    dtype = torch.float16
    batch_size = 1
    seq_len = 2
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
    
    # create encoder
    encoder = CompressPatch(
        in_channels=in_channels,
        embedding_dim=embedding_dim,
        in_height = in_height,
        in_width = in_width,
        patch_size=patchsize,
        conv_type='PeriodicConv2d',  # "PeriodicConv2d" or "Conv2d"
        send_skip_connection_to_decoder=True,
    ).to(dtype=dtype, device='cuda')
    encoder.self_introduction()
    
    # create a decoder, pass in the details from the encoder
    decoder = DecompressPatch(
        final_out_channels=out_channels,
        final_out_height=out_height,
        final_out_width=out_width,
        details_encoder = encoder.configure(),
        receive_skip_connection_from_encoder = True,
        concat_skip_connection_from_encoder = True,
        conv_type = 'PeriodicConv2d', # "PeriodicConv2d" or "Conv2d"
        deconv_type = "PixelShuffle", # PixelShuffle, ConvTranspose2d
    ).to(dtype=dtype, device='cuda')
    
    # randomly create tensors
    patches_along_height = in_height // patchsize[0]
    patches_along_width = in_width // patchsize[1]
    source_input = torch.randn(batch_size, seq_len, in_channels, in_height, in_width).to(dtype=dtype, device='cuda')
    output_transformer = torch.randn(batch_size, seq_len * patches_along_height * patches_along_width, embedding_dim)
    output_transformer = rearrange(output_transformer, 'b (s h w) d -> (b s) d h w', h=patches_along_height, w=patches_along_width).to(dtype=dtype, device='cuda')
    print(f"source_input shape: {source_input.shape}")
    print(f"output_transformer shape: {output_transformer.shape}")
    
    # here we go
    output_encoder = encoder(source_input)
    print(f"output_encoder shape: {output_encoder['output'].shape}")
    output_decoder = decoder(output_transformer, output_encoder['skip_connection'])
    print(f"output_decoder shape: {output_decoder.shape}")
    
    
    
    # batch_size = 2      
        # for sequence length == 3:
            # if there is no skip connection from encoder
                # intermidiate_patch_size = 4
                    # we use ConvTranspose2d, then memory usage is 22,525MiB
                    # we use PixelShuffle, then memory usage is 21,917MiB
                # intermidiate_patch_size = 2
                    # we use PixelShuffle, then memory usage is 26,047MiB
        
        # for sequence length == 5:
            # if there is no skip connection from encoder
                # intermidiate_patch_size = 4
                    # we use PixelShuffle, then memory usage is 34,897MiB
                # intermidiate_patch_size = 2
                    # we use PixelShuffle, then memory usage is 42,143MiB