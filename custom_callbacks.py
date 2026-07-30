from imports import *

def load_ckpt(pl_model, dirc_ckpt, variable_name='state_dict'):
    state = torch.load(dirc_ckpt, map_location=lambda storage, loc: storage)
    print(pl_model.load_state_dict(state[variable_name], strict=True)) # model_state_dict is for ckpt saved by vanilla pytorch, while state_dict is for ckpt saved by pytorch-lightning
    print("previous weights are loaded: \n", dirc_ckpt)
    return pl_model