"""
1) Do all the resource accounting exercises
2) Go through the implementation function by functuon and justuify the design choices made
3) Try building a computational graph
"""


import pickle
import jaxtyping
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import rearrange, einsum, repeat
import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.std = (2/(self.in_features+self.out_features))**0.5
        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, mean=0, std=self.std,a=-3*self.std, b=3*self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "... out in, ... in -> ... out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids):
        return self.embedding[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.G = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x):
        og_dtype = x.dtype
        x = x.to(torch.float32)
        self.G.to(torch.float32)
        Squares = (x**2).mean(dim=-1, keepdim=True)
        RMS = (Squares + self.eps)**0.5
        return einsum(x/RMS, self.G, "... d_model, d_model -> ... d_model").to(og_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype = None):
        super().__init__()
        if d_ff is None:
            self.d_ff = int((8/3 * d_model)//64)*64
        else:
            self.d_ff = d_ff
        self.W1 = nn.Parameter(torch.empty(self.d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model, self.d_ff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(self.d_ff, d_model, device=device, dtype=dtype))
        std = (2/(self.d_ff+d_model))**0.5 
        nn.init.trunc_normal_(self.W1,mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.W2,mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.W3,mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        y = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff") #W_1 x
        SiLU = y * torch.sigmoid(y) #Silu(W_1 x)
        z = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff") * SiLU 
        return einsum(self.W2, z, "d_model d_ff, ... d_ff -> ... d_model")



class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.max_seq = max_seq_len
        self.theta = theta
        self.d = d_k

        if d_k%2 !=0:
            raise ValueError

        def rotation_block(i, k):
            angle = torch.tensor(i/(self.theta**((2*k-2)/d_k)))
            return torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]], device=device, dtype=dtype)

        def rotation(i):
            return torch.block_diag(*[rotation_block(i, k) for k in range(1, int(self.d//2) +1)])

        self.R = torch.stack([rotation(i) for i in range(self.max_seq)])

    def forward(self, x, token_positions):
        if token_positions.max() >= self.max_seq or token_positions.min() < 0:
            raise IndexError("sequence position out of bounds")
        rotations = self.R.to(device=x.device, dtype=x.dtype)[token_positions.to(x.device)]
        return einsum(rotations, x, "... sequence out_coord in_coord, ... sequence in_coord -> ... sequence out_coord")

def Softmax(x, dim):
    maxi = x.max(dim=dim, keepdim=True).values
    expo = (x-maxi).exp()
    return expo/(expo.sum(dim=dim, keepdim=True))


def Scaled_dot_product_attention(Q:Float[Tensor, "queries d_k"], K: Float[Tensor, "keys d_k"], V: Float[Tensor, "values d_v"], mask=None):
    pre_mask = 1/(Q.size(-1)**0.5) * einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        post_mask = pre_mask.masked_fill(~mask, -torch.inf)
    else:
        post_mask = pre_mask
    return einsum(Softmax(post_mask, -1), V, "... queries keys, ... keys d_v -> ... queries d_v")


class multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None,  theta=None, max_seq_len=None):
        super().__init__()
        if d_model%num_heads != 0:
            raise ValueError("size of model not factor of head count")
        self.h = num_heads
        self.d_model = d_model
        self.d_k = int(d_model/num_heads)
        self.d_v = int(d_model/num_heads)

        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = None

        if theta is not None:
            self.rope = RoPE(theta=theta, d_k=d_model/num_heads, max_seq_len=max_seq_len, device=device, dtype=dtype)



    def forward(self, x: Float[Tensor, "... sequences d_model"], token_positions=None):
        rows, cols = x.shape[-2], x.shape[-2]
        i, j = torch.meshgrid(torch.arange(rows, device=x.device), torch.arange(cols, device=x.device), indexing='ij')
        mask = j <= i

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        heads = []
        for i in range(self.h):
            q = Q[..., i*self.d_k: (i+1)*self.d_k]
            k = K[..., i*self.d_k: (i+1)*self.d_k]
            v = V[..., i*self.d_k: (i+1)*self.d_k]

            if self.rope is not None:
                q = self.rope(q, token_positions)
                k = self.rope(k, token_positions)

            heads.append(Scaled_dot_product_attention(q, k, v, mask=mask))

        multihead = torch.cat(heads, dim=-1)

        return self.W_O(multihead)


class transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, device=None, dtype=None, theta=None, token_positions=None, max_seq_len=None):
        super().__init__()
        self.token_positions = token_positions
            
        self.multihead_self_attention_call = multihead_self_attention(d_model, num_heads, device=device, dtype=dtype, theta=theta, max_seq_len=max_seq_len)
        self.FFN = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... batch sequence d_model"]):
        if self.token_positions is None:
            token_positions = torch.tensor([i for i in range(x.shape[-2]) ], device=x.device)
        else:
            token_positions = self.token_positions.to(x.device)

        y = x + self.multihead_self_attention_call(self.rms_norm1(x), token_positions=token_positions)

        return y + self.FFN(self.rms_norm2(y))



class transformer_lm(nn.Module):
    def __init__(self, vocab_size, d_model, context_length, num_layers, num_heads, d_ff, device=None, dtype=None, theta=None, token_positions=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([transformer_block(d_model, num_heads, d_ff, device=device, dtype=dtype, theta=theta, max_seq_len=context_length, token_positions=token_positions) for _ in range(num_layers)])
        self.output_embedding = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.rms_norm = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x):
        out = self.embedding(x)

        for block in self.blocks:
            out = block(out)

        return self.output_embedding(self.rms_norm(out))



#Investigate why subtracting the largest element causes more stability
def cross_entropy_loss(logits, x):
    """
    log(exp[ o_i[x_{i+1}] ]  / \sum_a exp[ o_i[a] ]) = o_i[x_{i+1}] - log(\sum_a exp[ o_i[a] ])
    """
    selected_terms = logits[torch.arange(logits.size(0)), x]
    biggest_terms = torch.max(logits, dim=-1, keepdim=True).values
    norming_terms = torch.exp(logits-biggest_terms).sum(dim=-1) 

    return (torch.log(norming_terms) + biggest_terms.squeeze(-1) - selected_terms).mean()



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3,  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0 or betas[0]<0 or betas[1]<0 or eps<0 or weight_decay<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"][0], group["betas"][1]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1) 
                m = state.get("m", 0)
                v = state.get("v", 0)

                grad = p.grad.data

                p -= lr * weight_decay * p

                lr_t = lr * ((1-beta2 ** t) ** 0.5) / (1-beta1 ** t)

                m = beta1 * m + (1-beta1) * grad 
                v = beta2 * v + (1-beta2) * (grad**2)
                state["m"] = m
                state["v"] = v

                p -= lr_t * (m / (v**0.5 + eps))
                
                state["t"] = t+1

        return None


def learning_rate_schedule(alp_min, alp_max, t, warmup, post_anneal):
    if t < warmup:
        return (t / warmup) * alp_max
    elif t <= post_anneal:
        return alp_min + 0.5 * (1 + math.cos( (t - warmup) / (post_anneal - warmup) *  math.pi)) * (alp_max - alp_min)
    else:
        return alp_min


@torch.no_grad()
def gradient_clipping(params, max_l2_norm, eps=1e-6):
    grads = torch.cat([
    p.grad.reshape(-1)
    for p in params
    if p.grad is not None
    ])

    l2_norm = torch.linalg.vector_norm

    l2_norm = torch.linalg.vector_norm(grads)

    if l2_norm > max_l2_norm:
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * (max_l2_norm / (l2_norm + eps))

if __name__ == "__main__":
    #Learning rate experiment:
    # weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # opt = torch.optim.SGD([weights], lr=1e3)

    # for t in range(10):
    #     opt.zero_grad()
    #     loss = (weights**2).mean()
    #     print(loss.cpu().item())
    #     loss.backward()
    #     opt.step()

    """
    We test the optimizer on learning rates of 1e1, 1e2, 1e3 on 10 runs.
    FOr 1e1, it converges quite nicely to 0.4123
    For 1e2, the loss does not move much at all, it starts at 24.651573181152344, stays very close to it all 10 runs and
    ends at 24.651567459106445.
    For 1e3, it quickly blows up to an asctronomical valye of 2.63e+24.
    """
