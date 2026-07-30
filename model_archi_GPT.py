# GPT2 PyTorch Implementation (by graykode) - Annotated in Detail
# Based on OpenAI GPT-2 and Hugging Face's PyTorch implementation

from imports import *

# Rotary embedding helper
def apply_rotary_pos_emb(x, sin, cos):
    # excute rotary computation
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated.flatten(-2)

def build_rotary_embeddings(abs_pos, dim, patches_one_step):
    # theta for the lowest dimension is always 1, and for the highest dimension is always 1/100 (if base is 100, and 1/10000 if base is 10000), and exponentially decaying in between
    # the key to choose the base is to make sure that the highest dimension does not exceed a full rotation
    theta = 100 ** (-2 * (torch.arange(0, dim//2,device=abs_pos.device, dtype=abs_pos.dtype) ) / dim)
    # for each step, evenly rotate, here absolute position is crucial, because we will slide the window
    freqs = torch.einsum("i,j->ij", abs_pos, theta)
    # each step has many patches, so we need to repeat the frequency for each patch
    freqs = freqs.repeat_interleave(patches_one_step, dim=0)
    # compute sin and cos
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, current_S*patches_one_step, dim//2]
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, current_S*patches_one_step, dim//2]
    return sin, cos

# Multi-head masked self-attention
class Attention(nn.Module):
    def __init__(self, 
                 n_embd = 4096, 
                 window_size = 2*45*45,
                 patches_one_step = 45*45,
                 n_head = 4, 
                 scale=False):
        super(Attention, self).__init__()
        # 
        self.window_size = window_size
        self.patches_one_step = patches_one_step
        self.n_head = n_head
        self.n_embd = n_embd
        
        assert self.n_embd % self.n_head == 0
        
        # bias is used to mask the attention weights
        # self.register_buffer("bias", torch.tril(torch.ones(self.window_size, self.window_size)).view(1, 1, self.window_size, self.window_size))  # [1, 1, self.window_size, self.window_size]
        steps = self.window_size // self.patches_one_step
        bias = torch.zeros(window_size, window_size)
        for s in range(steps):
            for p in range(s+1):
                bias[s*self.patches_one_step:(s+1)*self.patches_one_step, p*self.patches_one_step:(p+1)*self.patches_one_step] = 1
        self.register_buffer("bias", bias.view(1, 1, self.window_size, self.window_size))  # [1, 1, self.window_size, self.window_size]
        # register_buffer instead of regular tensor (self.bias =): moved to GPU with the model, saved/loaded with the model’s state, not treated as a parameter to optimize
        
        self.scale = scale
        self.c_attn = nn.Linear(self.n_embd, self.n_embd*3)  # projects to query, key, value
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)  # projects to output

    def _attn(self, q, k, v): 
        # q: [B, nh, current_S*patches_one_step, hd]
        # k: [B, nh, hd, (past_S+current_S)*patches_one_step]
        # v: [B, nh, (past_S+current_S)*patches_one_step, hd]
        
        w = torch.matmul(q, k)  # w: [B, nh, current_S*patches_one_step, (past_S+current_S)*patches_one_step]
        
        # scaled dot-product attention
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        
        # extract mask from a complete attention mask (self.bias)
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]                             # b: [1, 1,  current_S*patches_one_step, (past_S+current_S)*patches_one_step]
        
        # apply mask (masked positions get very negative value)
        w = w * b - 1e10 * (1 - b)                                     # w: [B, nh, current_S*patches_one_step, (past_S+current_S)*patches_one_step]
        
        w = nn.Softmax(dim=-1)(w)                                      # w: [B, nh, current_S*patches_one_step, (past_S+current_S)*patches_one_step]
        
        return torch.matmul(w, v)                                      # [B, nh, current_S*patches_one_step, hd], the same shape as q

    def merge_heads(self, x):
        # x: [B, nh, current_S, hd]
        x = x.permute(0, 2, 1, 3).contiguous()                    # [B, nh, current_S, hd] → [B, current_S, nh, hd]
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)  # 
        return x.view(*new_x_shape)                               # [B, current_S, nh * hd] ==> [B, current_S, D]

    def split_heads(self, x, k=False):
        # x: [B, current_S, D]
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        # x: [B, current_S, nh, hd]
        if k:
            return x.permute(0, 2, 3, 1)  # [B, nh, hd, current_S]
        else:
            return x.permute(0, 2, 1, 3)  # [B, nh, current_S, hd]

    def forward(self, x, layer_past=None):
        # x: [B, current_S*patches_one_step, D]
        # layer_past: None at the very beginning, it will be [2, B, nh, past_S*patches_one_step, hd] after the first forward pass for each layer
        
        # create multi-head QKV
        x = self.c_attn(x)                                   # x: [B, current_S*patches_one_step, D] → c: [B, current_S*patches_one_step, 3D], linear projection
        query, key, value = x.split(self.n_embd, dim=2)      # 3-way split: [B, current_S*patches_one_step, 3D] → [B, current_S*patches_one_step, D], [B, current_S*patches_one_step, D], [B, current_S*patches_one_step, D]
        query = self.split_heads(query)                      # [B, current_S*patches_one_step, D] → [B, nh, current_S*patches_one_step, hd]
        key = self.split_heads(key, k=True)                  # [B, current_S*patches_one_step, D] → [B, nh, hd, current_S*patches_one_step]
        value = self.split_heads(value)                      # [B, current_S*patches_one_step, D] → [B, nh, current_S*patches_one_step, hd]

        # RoPE
        RoPE = True
        if RoPE:
            # get absolute positions
            past_S = layer_past.shape[-2] // self.patches_one_step if layer_past is not None else 0
            current_S = key.shape[-1] // self.patches_one_step
            abs_pos = torch.arange(past_S, past_S + current_S, device=key.device, dtype=key.dtype)
            # rotate
            sin,cos = build_rotary_embeddings(abs_pos, key.shape[-2],self.patches_one_step)
            query = apply_rotary_pos_emb(query, sin, cos)
            key = apply_rotary_pos_emb(key.transpose(-2, -1), sin, cos).transpose(-2, -1)
        
        # combine past key/value with current key/value
        if layer_past is not None:
            # separate past key/value into key and value
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # past_key:   [B, nh, hd, past_S*patches_one_step]
                                                                                   # past_value: [B, nh, past_S*patches_one_step, hd]
            # combine past key/value with current key/value
            key = torch.cat((past_key, key), dim=-1)                               # past + current key: [B, nh, hd, (past_S+current_S)*patches_one_step]
            value = torch.cat((past_value, value), dim=-2)                         # past + current value: [B, nh, (past_S+current_S)*patches_one_step, hd]
        
        # save past + current key/value for next time step
        present = torch.stack((key.transpose(-2, -1), value))  # [2, B, nh, (past_S+current_S)*patches_one_step, hd]

        # apply attention
        a = self._attn(query, key, value)                      # a: [B, nh, current_S*patches_one_step, hd]
        a = self.merge_heads(a)                                # a: [B, nh, current_S*patches_one_step, hd] → [B, current_S*patches_one_step, nh*hd=D]
        a = self.c_proj(a)                                     # a: [B, current_S*patches_one_step, D], linear projection
        
        return a, present                                      # a: [B, current_S, D], present (key/value pairs for this one layer): [2, B, nh, past_S + current_S, hd]

# Feedforward layer with GELU
class MLP(nn.Module):
    def __init__(self, n_state, n_embd):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd, n_state)
        self.c_proj = nn.Linear(n_state, n_embd)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, current_S, D]
        h = self.act(self.c_fc(x))  # [B, current_S, 4D]
        h2 = self.c_proj(h)         # [B, current_S, D]
        return h2

# Transformer Block
class Block(nn.Module):
    def __init__(self,
                 window_size = 2*45*45, 
                 patches_one_step = 45*45,
                 n_embd=4096,
                 n_layer=4,
                 n_head=4,
                 scale=True,
                 ):
        super(Block, self).__init__()
        #
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.window_size = window_size
        self.patches_one_step = patches_one_step
        self.window_size = window_size
        # 
        self.ln_1 = nn.LayerNorm(self.n_embd)
        self.attn = Attention(self.n_embd, self.window_size, self.patches_one_step, self.n_head, scale)
        self.ln_2 = nn.LayerNorm(self.n_embd)
        self.mlp = MLP(4 * self.n_embd, self.n_embd)

    def forward(self, x, layer_past=None):
        # x         : [B, current_S*patches_one_step, D], current_S is 1 in inference, and window_size//patches_one_step in training
        # layer_past: list of L * [None] at the beginning, it will be a list of [2, B, nh, past_S, hd] after the first forward pass for each layer
        
        # layer_norm -> attention
        a, present = self.attn(self.ln_1(x), layer_past=layer_past) # a: [B, current_S*patches_one_step, D], present (key/value pairs for this one layer): [2, B, nh, current_S*patches_one_step, hd]
        
        # residual connection
        x = x + a                                                   # x: [B, current_S*patches_one_step, D]
        
        # layer_norm -> MLP
        m = self.mlp(self.ln_2(x))                                  # m: [B, current_S*patches_one_step, D] 
        
        # residual connection
        x = x + m  
        
        return x, present                                           # x: [B, current_S*patches_one_step, D], present (key/value pairs for this one layer): [2, B, nh, current_S*patches_one_step, hd]

# GPT2Model implementation: core transformer body
class GPT2Model(nn.Module):
    def __init__(self,
                window_size=2*45*45,               # The context window size, i.e., how many tokens the model can process at once (input length)
                n_embd=4096,                       # embedding size
                n_layer=2,                         # number of transformer blocks
                n_head=2,                          # number of attention heads
                patches_one_step=45*45,            # number of patches in one step
                vae = False,                       # whether to use VAE
                max_steps_sliding_window_autoregressive = 120,
                ):
        super(GPT2Model, self).__init__()
        self.n_head = n_head
        self.patches_one_step = patches_one_step
        self.n_layer = n_layer                   
        self.n_embd = n_embd                      
        self.window_size = window_size           
        self.vae = vae  

        # Positional embedding for space [patches_one_step, D]
        self.wpe = nn.Embedding(self.patches_one_step, self.n_embd)

        # One transformer block
        block = Block(window_size = self.window_size, 
                      patches_one_step=self.patches_one_step,
                      n_embd=self.n_embd,
                      n_layer=self.n_layer,
                      n_head=self.n_head,
                      scale=True)
              
        # Stack of L blocks
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(self.n_layer)])  
        
        # vae layer
        if self.vae:
            self.vae_layer = nn.Sequential(
                nn.LayerNorm(self.n_embd),
                nn.Linear(self.n_embd, 2 * self.n_embd),
            )

    def forward(self, inputs_embeddings, past=None):
        # inputs_embeddings: [B, current_S * patches_one_step, D]; current_S maybe 1 in inference, and window_size/patches_one_step in training
        # past is None at the beginning, it wil be a list of L (layer) * [2, B, nh, past_S * patches_one_step, hd] after the first forward pass

        # past is None at the very beginning in both training and inference
        if past is None:
            past = [None] * len(self.h)        # Create empty cache for each layer → List[None] * L

        #@ add position embeddings of space
        
        # retrieve all position embeddings for space （just one step）
        position_embeds = self.wpe(
            torch.arange(self.patches_one_step, device=inputs_embeddings.device, dtype=torch.long)
            ) # [patches_one_step, D]

        # inputs_embeddings.shape: [B, current_S*patches_one_step, D], position_embeds.shape:[patches_one_step, D]
        current_S = inputs_embeddings.size(1) // self.patches_one_step
        inputs_embeddings = rearrange(inputs_embeddings, 'b (s p) d -> (b s) p d', s=current_S)
        # add input embeddings and position embeddings of space by broadcasting
        hidden_states = inputs_embeddings + position_embeds
        hidden_states = rearrange(hidden_states, '(b s) p d -> b (s p) d', s=current_S)  # [B, current_S * patches_one_step, D]

        #@ go through all transformer layers for this step
        
        # store key/value caches for each layer
        presents = []
        # Iterate through each transformer block
        for block, layer_past in zip(self.h, past):
            # self.h is a list of transformer blocks, each block is a layer
            # layer_past is a list of past key/value pairs for each layer
            hidden_states, present = block(hidden_states, layer_past)  # hidden_states: [B, current_S*patches_one_step, D]
                                                                       # layer_past: None at the very beginning, it will be [2, B, nh, past_S*patches_one_step, hd] after the first forward pass for each layer
                                                                       # present: (key/value pairs for this one layer): [2, B, nh, (current_S+past_S)*patches_one_step, hd]
            # save each layer's key/value pairs as a list
            presents.append(present)
        
        # use vae layer to add noise
        if self.vae:
            # apply VAE layer
            hidden_states = self.vae_layer(hidden_states)
            # split the last dimension into two parts: mean and logvar
            mean, logvar = hidden_states.split(self.n_embd, dim=-1)
            std = torch.exp(0.5 * logvar)
            # sample a random noise from a normal distribution
            least_randomness = False   # all patches share the same noise
            if least_randomness:
                eps = torch.randn( [1,1,std.shape[-1]], dtype=inputs_embeddings.dtype, device=inputs_embeddings.device)
            else:
                eps = torch.randn_like(std, dtype=inputs_embeddings.dtype, device=inputs_embeddings.device)
            # re-parameterization trick
            hidden_states = mean + eps * std 
        
        # we don't need the original final layernorm which serves for the last output_embedding layer(linear projection from embedding_dim to vocab_size)
        # each block in our model has its own layernorm in the first layer of the block
        
        # [B, current_S*patches_one_step, D]; List of Tensor[key, value], each one is [2, B, nh, current_S*patches_one_step, hd]
        return hidden_states, presents, (mean, logvar) if self.vae else None
        

    
if __name__ == '__main__':
    
    # data configurations
    dtype = torch.float16
    batch_size = 1
    seq_len = 3
    in_channels = 69
    nearest_neighbor_factor = 1
    in_height = 721 // nearest_neighbor_factor
    in_width = 1440 // nearest_neighbor_factor
    patchsize = (16,32)
    patches_along_height = in_height // patchsize[0]
    patches_along_width = in_width // patchsize[1]
    embedding_dim = 4096
    
    # model configurations
    window_size = 3*45*45
    patches_one_step = 45*45
    n_embd = 4096
    n_layer = 2
    n_head = 2
    model = GPT2Model(window_size=window_size,         
                      n_embd=n_embd,
                      n_layer=n_layer,     
                      n_head=n_head,       
                      patches_one_step=patches_one_step,).to(dtype=dtype, device='cuda')
    
    # randomly create tensors
    input_transformer = torch.randn(batch_size, seq_len * patches_along_height * patches_along_width, embedding_dim).to(dtype=dtype, device='cuda')
    print(f"input_transformer shape: {input_transformer.shape}")
    
    
    '''
    # in this script, remember the following terms:
            # initial_S is the length of the input sequence by the user
            # current_S is the length of the sequence we are generating, it's either 1 during inference or max_window_size during training
            # past_S is the length of the sequence we have generated K/V cache for it. Actually, it is the length of KV cache.
            # So, before the first forward pass, past_S = 0 even though initial_S is larger than 0.
            # after the first forward pass, past_S = initial_S. Then it increases by 1 for each forward pass.
    '''
    
    
    # inference: 
    prev = input_transformer
    generations = []
    past_kv = None
    new_steps = 10
    with torch.no_grad():
        for n in trange(new_steps):
            print(f"this is the {n}th forward pass")
            
            # using current input IDs to go trough all transformer blocks
            output_embeddings, past_kv, _ = model(prev, past=past_kv)
            
            # sliding window
            if (past_kv[0].shape[3] // patches_one_step) == (window_size // patches_one_step):
                for lyr in range(len(past_kv)):
                    # past_kv is a list of [2, B, nh, past_S*patches_one_step, hd]      
                    past_kv[lyr] = rearrange(past_kv[lyr], 'kv b nh (s p) d -> kv b nh s p d', kv=2, p=patches_one_step)
                    past_kv[lyr] = past_kv[lyr][:, :, :, 1:, :, :]
                    past_kv[lyr] = rearrange(past_kv[lyr], 'kv b nh s p d -> kv b nh (s p) d',kv=2,p=patches_one_step)    
            
            # only need the last step of the output embeddings for next prediction
            output_embeddings = rearrange(output_embeddings, 'b (s p) d -> b s p d', p=patches_one_step)
            output_embeddings = output_embeddings[:, -1, :, :]
            prev = output_embeddings
            
            