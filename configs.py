from imports import *
from model_archi_encoder import calculate_best_embedding_dim

#@ hyperparameters for model

max_seq_len_per_forward = 2                                                                                           # >= init_inut_seq_len
in_channels = 69                                                                                                      # input image channel
resample_scale = 2                                                                                                    # resample scale, 1 means no resample
in_height = 721 // resample_scale + 721 % resample_scale                                                              # input image height  
in_width = 1440 // resample_scale + 1440 % resample_scale                                                             # input image width
patch_size = (8,16)                                                                                                  # patch size
compression_ratio_for_each_patch2embedding = 0.1                                                                     # compression ratio for each patch to embedding
embedding_dim = calculate_best_embedding_dim(in_channels, patch_size, compression_ratio_for_each_patch2embedding)     # embedding dimension
transformer_layers = 6                                                                                               # number of transformer layers
transformer_n_heads = 8                                                                                               # number of transformer heads
out_channels = 69                                                                                                     # output image channel
out_height = in_height                                                                                                # output image height
out_width = in_width                                                                                                  # output image width
vae = False                                                                                                           # whether to use VAE layer 


#@ hyperparameters for dataset and dataloader

dir_ERA = "era5_1979-2020janu_zstd_unified.zarr"
num_steps = max_seq_len_per_forward         # number of steps input to the model, should be >= 2 and <= max_seq_len_per_forward
batch_size = 1 
num_workers = 8

#@ hyperparameters for training

lr = 1.5e-4 # from Megatron-LM, 0.0001 is also ok

accelerator = 'gpu' 

epoch = 100

devices = [0, 1, 2, 3, 4, 5, 6, 7]                                                                                          # GPU devices to use

parent_path = '/ExploreLWM/codes/'
if not os.path.exists(parent_path + 'cps/'):
    os.makedirs(parent_path + 'cps/')

dtype = torch.float16

fit_or_predict = 'fit'                          # fit, test, predict

enable_profiler = False

load_pretrained_weight_file = False
if load_pretrained_weight_file:
    pretrained_weight_file = None


longname_model = f'model[{max_seq_len_per_forward}maxlenperfw_{in_channels}inc_{in_height}inh_{in_width}inw_{patch_size[0]}x{patch_size[1]}patsize_{embedding_dim}embdim_{transformer_layers}trlyr_{transformer_n_heads}trh]-'
longname_dataset = f'dset[{num_steps}instp_{resample_scale}rescale]-'
longname_train = f'train[{batch_size}bs_{num_workers}nwo_{lr:.0e}lr_{epoch}ep_{str(devices).replace(" ", "")}dev_{str(dtype)[-2:]}dt]'
addition = f'profile' if enable_profiler else ''
longname = longname_model + longname_dataset + longname_train + addition