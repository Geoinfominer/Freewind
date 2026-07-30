from imports import *


def calculate_best_embedding_dim( 
    channels: int,
    patchsize: tuple,
    compression_ratio: float,
) -> int:
    """
    Calculate the best embedding dimension for the model for a patch of size (channels, patchsize, patchsize).
    The best embedding dimension is defined as the following:
    1. it should be exactly the power of 2
    2. it should be close to channels * patchsize[0] * patchsize[1] * compression_ratio
    Args:
        channels (int): Number of input channels.
        patchsize (tuple): Size of the patches.
        compression_ratio (float): Compression ratio.
    Returns:
        int: Best embedding dimension.
    """
    # Calculate the target embedding dimension
    target_embedding_dim = int(channels * patchsize[0] * patchsize[1] * compression_ratio)

    # Find the closest power of 2 to the target embedding dimension
    best_embedding_dim = 1
    while best_embedding_dim < target_embedding_dim:
        best_embedding_dim *= 2

    # If the best embedding dimension is greater than the target, check the previous power of 2
    if best_embedding_dim > target_embedding_dim:
        prev_embedding_dim = best_embedding_dim // 2
        if abs(prev_embedding_dim - target_embedding_dim) < abs(best_embedding_dim - target_embedding_dim):
            best_embedding_dim = prev_embedding_dim

    return best_embedding_dim
   
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

# define a class of CompressPatch layer using Conv2d, kernel_size=patchsize, stride=patchsize
class CompressPatch(nn.Module):
    """
    The layer reshapes the input tensor into patches of the specified size.
    it is composed of 2 branches:
        - 1. the first branch is a convolutional layer with kernel size and stride equal to the patch size, e.g., 32, followed by Norm and activation layers.
            it directly compress input patches (channels, patchsize, patchsize) into (embedding_dim, 1, 1).
        - 2. the second branch is gradually compress input patches (channels, patchsize, patchsize) into (embedding_dim, 1, 1), using the combination of Conv2d, Norm and activation layers.
        - 3. add the outputs of the two branches together.
    """
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        in_height: int,
        in_width: int,
        patch_size: tuple,
        conv_type: str = "PeriodicConv2d", # "PeriodicConv2d" or "Conv2d"
        send_skip_connection_to_decoder: bool = True,
        first_increase_channels: bool = False,
    ):
        super(CompressPatch, self).__init__()
        
        # if conv_type == "PeriodicConv2d" then use PeriodicConv2d, otherwise use Conv2d
        if conv_type == "PeriodicConv2d":
            self.conv = PeriodicConv2d
        else:
            self.conv = nn.Conv2d
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.first_increase_channels = first_increase_channels
        self.send_skip_connection_to_decoder = send_skip_connection_to_decoder
        self.configurations = {}
        
        #@ Branch 0: Directly compress input patches into (embedding_dim, 1, 1)
        
        self.branch0 = nn.Sequential(
            PermuteLayerNorm2d(in_channels),
            self.conv(
                in_channels,
                embedding_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=(0,0),
            ),
            nn.GELU(),
        )
        self.configurations['branch0'] = str(self.branch0)
        
        #@ Branch 1: Progressively compress input patches into (embedding_dim, 1, 1)
        
        self.convs_brach1 = nn.ModuleList()
        
        # increase dimension first (optional, it only increases the channels, do not change the height and width)
        if self.first_increase_channels:
            increased_channels = calculate_best_embedding_dim(
                channels = in_channels * 2,
                patchsize = 1,
                compression_ratio = 1,
            )
            self.increase_channels_layer = nn.Sequential(
                PermuteLayerNorm2d(in_channels),
                self.conv(          # can be 1*1 conv, or 3*3 conv, just for increasing the channels
                    in_channels,
                    increased_channels,
                    kernel_size=3,
                    stride=1,
                    padding=(1, 1),
                ),
                nn.GELU(),
            )
            print(f"here we use increase_channels_layer for branch1, so in_channels changes from {self.in_channels} to {increased_channels}")
            self.in_channels = increased_channels
            
            # if first_increase_channels is True, then add the increase_channels_layer to the convs
            self.convs_brach1.append(self.increase_channels_layer)
            
            self.configurations['branch1-0-increase_channels_layer'] = str(self.increase_channels_layer)
        
        # intermediate_patch_sizes are combined to be equal to the patch_size, we use a greedy logic to find the intermediate_patch_sizes comination
        intermediate_patch_size = (2,2) # for simplicity, can you set 2*2, 4*4, 8*8. it is no need to set 1st element and 2nd element to be different.
        # calculate the intermediate_patch_size along height
        intermediate_patch_size_h = intermediate_patch_size[0]
        intermediate_patch_sizes_h = []
        greedy_h = self.patch_size[0] # it is a tuple, like (16, 32)
        while greedy_h > 1:
            greedy_lost_h = greedy_h % intermediate_patch_size_h
            if greedy_lost_h == 0:
                intermediate_patch_sizes_h.append(intermediate_patch_size_h)
                greedy_h = greedy_h // intermediate_patch_size_h
            else:
                greedy_h = greedy_lost_h
                intermediate_patch_size_h //= 2
        # calculate the intermediate_patch_size along width
        intermediate_patch_size_w = intermediate_patch_size[1]
        intermediate_patch_sizes_w = []
        greedy_w = self.patch_size[1] # it is a tuple, like (16, 32)
        while greedy_w > 1:
            greedy_lost_w = greedy_w % intermediate_patch_size_w
            if greedy_lost_w == 0:
                intermediate_patch_sizes_w.append(intermediate_patch_size_w)
                greedy_w = greedy_w // intermediate_patch_size_w
            else:
                greedy_w = greedy_lost_w
                intermediate_patch_size_w //= 2
        # combine the intermediate_patch_sizes_h and intermediate_patch_sizes_w
        len_intermediate_patch_sizes_h = len(intermediate_patch_sizes_h)
        len_intermediate_patch_sizes_w = len(intermediate_patch_sizes_w)
        if len_intermediate_patch_sizes_h < len_intermediate_patch_sizes_w:
            intermediate_patch_sizes_h.append(1)
        elif len_intermediate_patch_sizes_h > len_intermediate_patch_sizes_w:
            intermediate_patch_sizes_w.append(1)
        intermediate_patch_sizes = list(zip(intermediate_patch_sizes_h, intermediate_patch_sizes_w))
         
        # prepare convolution configuration. how many dimensions we need to reduce totally
        dimension_deduction = self.in_channels * self.in_height * self.in_width - self.embedding_dim * (self.in_height // self.patch_size[0]) * (self.in_width // self.patch_size[1])
        # how many dimensions we need to reduce in each step, we obey a linear rule to reduce the dimensions at each step
        dimension_deduction_each_step = dimension_deduction // len(intermediate_patch_sizes)
        # loop over the intermediate_patch_sizes to create the conv layers
        convs_configs_brach1 = {}
        in_channels_this_step = self.in_channels
        hight_this_step = self.in_height
        width_this_step = self.in_width
        for idx, intermediate_patch_size in enumerate(intermediate_patch_sizes):
            # calculate the number of patches in this step
            n_patches_this_step_along_height = hight_this_step // intermediate_patch_size[0]
            n_patches_this_step_along_width = width_this_step // intermediate_patch_size[1]
            n_patches_this_step = n_patches_this_step_along_height * n_patches_this_step_along_width
            # in this step, how many dimensions we need to reduce
            minus_channels_this_step = dimension_deduction_each_step // n_patches_this_step
            # calculate the number of channels in this step, which accounts for the dimension deduction
            out_channels_this_step = intermediate_patch_size[0] * intermediate_patch_size[1] * in_channels_this_step - minus_channels_this_step
            # out_channels_this_step = calculate_best_embedding_dim(  # be careful, this may lead to error
            #     channels = out_channels_this_step,
            #     patchsize = 1,
            #     compression_ratio = 1,
            # )
            if idx == len(intermediate_patch_sizes) - 1:
                out_channels_this_step = self.embedding_dim
            # define the config for the conv layer
            convs_configs_brach1[idx] = {
                "in_channels": in_channels_this_step,
                "out_channels": out_channels_this_step,
                "kernel_size": intermediate_patch_size,
                "stride": intermediate_patch_size,
                "padding": (0, 0),
            }
            # update
            in_channels_this_step = out_channels_this_step
            hight_this_step = n_patches_this_step_along_height
            width_this_step = n_patches_this_step_along_width
        print(f"convs_configs for branch1: {convs_configs_brach1}")
        
        # using the module list to store the conv layers according to the configs, loop over the convs_configs_brach1 to create the conv layers
        for idx, config in convs_configs_brach1.items():
            self.convs_brach1.append(
                nn.Sequential(
                    PermuteLayerNorm2d(config["in_channels"]),
                    self.conv(
                        config["in_channels"],
                        config["out_channels"],
                        kernel_size=config["kernel_size"],
                        stride=config["stride"],
                        padding=config["padding"],
                    ),
                    nn.GELU(),
                )
            )
            self.configurations[f'branch1-{idx+1 if self.first_increase_channels else idx}-conv'] = str(self.convs_brach1[-1])
        
        print(f'finished creating Encoder, the in_channels is {self.in_channels}, the embedding_dim is {self.embedding_dim}, the patch_size is {self.patch_size}')
    
    def configure(self):
        config_to_decoder = {}
        for k, v in self.configurations.items():
            v = v.replace(" ", "").replace("\n", "")
            pattern = r"\(\d+\):(.*?)(?=\(\d+\):|$)"
            matches = re.findall(pattern, v)
            config_to_decoder[k] = matches
        return config_to_decoder
    
    def self_introduction(self,
                          input_shape = (69, 721, 1440),
                          ):
        # ignore batch size and seq_len
        def conv_shape_calculation(
            out_channels: int,
            in_height: int,
            in_width: int,
            kernel_size: tuple,
            stride: tuple,
            padding: tuple,
        ): # Calculate the output shape of a convolutional layer 
            return (
                out_channels,
                (in_height + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,
                (in_width + 2 * padding[1] - kernel_size[1]) // stride[1] + 1,
            )
        print(f"{'-'*20}Hi there, I am Encoder, below is my self-introduction:{'-'*20}")
        my_config = self.configure()
        keys = list(my_config.keys())
        keys.sort()
        key_branch0 = keys[0]
        key_branch1 = keys[1:]

        #@ branch0
        print(f"{'-'*20}branch0{'-'*20}")
        print(f"input_shape: {input_shape}")
        for lyr in my_config[key_branch0]:
            print(lyr)
        conv_layer = my_config[key_branch0][1]
        # 
        out_channels = int(conv_layer.split('(')[1].split(',')[1])
        kernel_size = int(conv_layer.split('kernel_size=(')[1].split(',')[0]), int(re.search(r'\d+', conv_layer.split('kernel_size=(')[1].split(',')[1]).group())
        stride = int(conv_layer.split('stride=(')[1].split(',')[0]), int(re.search(r'\d+', conv_layer.split('stride=(')[1].split(',')[1]).group())
        if 'padding=' in conv_layer:
            padding = int(conv_layer.split('padding=(')[1].split(',')[0]), int(re.search(r'\d+', conv_layer.split('padding=(')[1].split(',')[1]).group())
        else:
            padding = 0,0
        out_shape = conv_shape_calculation(
            out_channels=out_channels,
            in_height=input_shape[1],
            in_width=input_shape[2],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        print(f"output_shape: {out_shape}")
        
        #@ branch1
        print(f"{'-'*20}branch1{'-'*20}")
        input_shape_branch1 = input_shape
        for idx, key in enumerate(key_branch1):
            if idx == 0:
                print(f"input_shape: {input_shape_branch1}")
            for lyr in my_config[key]:
                print(lyr)
            conv_layer = my_config[key][1]
            # 
            out_channels = int(conv_layer.split('(')[1].split(',')[1])
            kernel_size = int(conv_layer.split('kernel_size=(')[1].split(',')[0]), int(re.search(r'\d+', conv_layer.split('kernel_size=(')[1].split(',')[1]).group())
            stride = int(conv_layer.split('stride=(')[1].split(',')[0]), int(re.search(r'\d+', conv_layer.split('stride=(')[1].split(',')[1]).group())
            if 'padding=' in conv_layer:
                padding = int(conv_layer.split('padding=(')[1].split(',')[0]), int(re.search(r'\d+', conv_layer.split('padding=(')[1].split(',')[1]).group())
            else:
                padding = 0,0
            out_shape = conv_shape_calculation(
                out_channels=out_channels,
                in_height=input_shape_branch1[1],
                in_width=input_shape_branch1[2],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            print(f"output_shape: {out_shape}")
            input_shape_branch1 = out_shape
        print(f"{'-'*20}Thanks for having me, I am Encoder. The Decoder just mirrors me, in addition to the skip connection (concat or add){'-'*20}")
    
    def forward(self, x):
        
        # print(f"input x shape: {x.shape}")
        # Reshape the input tensor 
        batch_size, seq_len, channels, height, width = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        
        # Branch 0
        x1 = self.branch0(x)
        if self.send_skip_connection_to_decoder:
            skip_connection = [x1]
        
        # Branch 1
        x2 = x
        for layer in self.convs_brach1:
            x2 = layer(x2)
            if self.send_skip_connection_to_decoder:
                skip_connection.append(x2)
                
        # add the outputs of the two branches together
        x = x1 + x2
        
        # Reshape the output tensor
        x = rearrange(x, "(b s) c h w -> b s (h w) c", b=batch_size, s=seq_len)
        # print(f"Encoder output x shape: {x.shape}")
        
        return {
            'output': x,
            'skip_connection': skip_connection if self.send_skip_connection_to_decoder else None,
        }

if __name__ == "__main__":
    
    # dateset config
    dtype = torch.float16
    batch_size = 2
    seq_len = 5
    channels = 69
    nearest_neighbor_factor = 1
    height = 721 // nearest_neighbor_factor
    width = 1440 // nearest_neighbor_factor
    x = torch.randn(batch_size, seq_len, channels, height, width).to(dtype=dtype, device='cuda') # (2, 5, 69, 721, 1440) is about 3189 MB at FT32, and 1823 MB at FT16
    print(x.shape)
    
    # encoder config
    patchsize = (16,32) # 16 pixels along the height, 32 pixels along the width per patch
                        # number of pixels along the height can be 2,4,8,16. should not => 32. because 721 / 32 = 22.53, then in decoder 22 * 32 = 704, we need to pad 17 pixels! that is not good.
    compression_ratio_for_each_patch2embedding = 0.1
    embedding_dim = calculate_best_embedding_dim(channels, patchsize, compression_ratio_for_each_patch2embedding)
    print(f"Best embedding dimension: {embedding_dim}")
    
    # create the encoder
    encoder = CompressPatch(
        in_channels=channels,
        embedding_dim=embedding_dim,
        in_height = height,
        in_width = width,
        patch_size=patchsize,
        conv_type="PeriodicConv2d", # "PeriodicConv2d" or "Conv2d"
    ).to(dtype=dtype, device='cuda')                                                             # the model itself is about  2416 MB for FT16 without input 
    
    details_encoder = encoder.configure()
    print(f"encoder config: {details_encoder}")
    
    encoder.self_introduction()
    
    # forward
    output = encoder(x)
    print(f"output shape: {output['output'].shape}")                                                         # 16,233MiB total allocated
    print(f"skip_connection: {len(output['skip_connection'])}")