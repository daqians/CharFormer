from functools import partial
import torch
from torch import nn, einsum
from einops import rearrange
from FusedAttention import FusedAttentionBlock

List = nn.ModuleList


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, depth=1):
    return val if isinstance(val, tuple) else (val,) * depth


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim, 1)
        self.project_out = nn.Sequential(
            nn.Conv2d(dim, out_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, 1)
        )

    def forward(self, x):
        x = self.project_in(x)
        return self.project_out(x)


# SELF-ATTENTION with windows and skip-connections
class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, window_size=16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x, skip=None):
        h, w, b = self.heads, self.window_size, x.shape[0]

        q = self.to_q(x)

        kv_input = x

        # Cancate the skip-connection with k and v
        if exists(skip):
            kv_input = torch.cat((kv_input, skip), dim=0)

        k, v = self.to_kv(kv_input).chunk(2, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) (x w1) (y w2) -> (b h x y) (w1 w2) c', w1=w, w2=w, h=h),
                      (q, k, v))

        if exists(skip):
            k, v = map(lambda t: rearrange(t, '(r b) n d -> b (r n) d', r=2), (k, v))

        # Matrix to Matrix multiplication, output self-attention reuslts
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h x y) (w1 w2) c -> b (h c) (x w1) (y w2)', b=b, h=h, y=x.shape[-1] // w, w1=w, w2=w)

        return self.to_out(out)


class GSNB(nn.Module):
    def __init__(self, dim, out_dim, depth=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(dim*2, dim, 1)
        if depth == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.Conv2d(out_dim, dim, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            )

    def forward(self, x, skip = None):
        if exists(skip):
            x = torch.cat((x, skip), dim=1)
            x = self.upsample(x)

        return self.conv(x) + x

class RSAB(nn.Module):
    def __init__(self, dim, depth, out_dim=128, dim_head=64, heads=8, window_size=16):
        super().__init__()
        self.dim = dim

        # self.pos_emb = AxialRotaryEmbedding(dim_head) if rotary_emb else None
        # self.pos_embedding = nn.Parameter(torch.randn(1, dim, dim))

        self.layers = List([])
        for _ in range(depth):
            self.layers.append(List([
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, window_size=window_size)),
                PreNorm(dim, FeedForward(dim, out_dim))
            ]))
        self.outer_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x, skip=None):
        current_depth = 0
        for attn, ff in self.layers:
            if current_depth == 0:
                x = attn(x, skip=skip) + x
            else:
                x = attn(x) + x
            current_depth += 1

            x = ff(x) + x

        return self.outer_conv(x)+x

class CFB(nn.Module):
    def __init__(self, dim=64, reduction=16, depth_RSAB=2, depth_GSNB = 2, dim_head=64,window_size=16,heads=8):

        super().__init__()

        self.fusedAtt = FusedAttentionBlock(dim, reduction)
        self.RSAB = RSAB(dim, depth=depth_RSAB, dim_head=dim_head, heads=heads, window_size=window_size)
        self.GSNB = GSNB(dim, dim*4, depth=depth_GSNB)

    def forward(self, input_rsab, input_gsnb, skip_rsab = None, skip_gsnb = None):
        RSAB_r = self.RSAB(input_rsab, skip_rsab)
        GSNB_r = self.GSNB(input_gsnb, skip_gsnb)

        # Two options to obtain the combined feature, we process simple add function (or can be concat as commented)
        # c = torch.cat((RSAB_r,GSNB_r),dim=1)
        fused_att = self.fusedAtt(RSAB_r + GSNB_r)

        return RSAB_r + fused_att, GSNB_r + fused_att

# classes

class CharFormer(nn.Module):
    def __init__(
            self,
            dim=64,
            channels=3,
            stages=4,
            depth_RSAB=2,
            depth_GSNB=2,
            dim_head=64,
            window_size=16,
            heads=8,
            input_channels=None,
            output_channels=None
    ):
        super().__init__()
        input_channels = default(input_channels, channels)
        output_channels = default(output_channels, channels)

        self.project_in = nn.Sequential(
            nn.Conv2d(input_channels, dim, 3, padding=1),
            nn.GELU()
        )

        self.project_out = nn.Sequential(
            nn.Conv2d(dim, output_channels, 3, padding=1),
        )

        self.feature_corrector = nn.Sequential(
            nn.Conv2d(dim, output_channels, 3, padding=1),
        )

        self.downs = List([])
        self.ups = List([])

        heads, window_size, dim_head, depth_RSAB, depth_GSNB = map(partial(cast_tuple, depth=stages),
                                                       (heads, window_size, dim_head, depth_RSAB, depth_GSNB))

        for ind, heads, window_size, dim_head, depth_RSAB, depth_GSNB in zip(range(stages), heads, window_size, dim_head,
                                                                 depth_RSAB, depth_GSNB):
            is_last = ind == (stages - 1)
            self.downs.append(List([
                CFB(dim, depth_RSAB=depth_RSAB, depth_GSNB = depth_GSNB, dim_head=dim_head, heads=heads, window_size=window_size),
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1)
            ]))

            self.ups.append(List([
                nn.ConvTranspose2d(dim * 2, dim, 2, stride=2),
                CFB(dim, depth_RSAB=depth_RSAB, depth_GSNB = depth_GSNB, dim_head=dim_head, heads=heads, window_size=window_size)
            ]))

            dim *= 2

            if is_last:
                self.mid = CFB(dim=dim, depth_RSAB=depth_RSAB, depth_GSNB = depth_GSNB, dim_head=dim_head, heads=heads, window_size=window_size)

    def forward(self, x):

        rsab = self.project_in(x)
        gsnb = torch.clone(rsab)

        skip_rsab = []
        skip_gsnb = []

        for block, downsample in self.downs:
            rsab, gsnb = block(rsab,gsnb)

            skip_rsab.append(rsab)
            skip_gsnb.append(gsnb)

            rsab = downsample(rsab)
            gsnb = downsample(gsnb)

        rsab,gsnb = self.mid(rsab,gsnb)

        for (upsample, block), skip1, skip2 in zip(reversed(self.ups), reversed(skip_rsab), reversed(skip_gsnb)):
            rsab = upsample(rsab)
            gsnb = upsample(gsnb)

            rsab,gsnb = block(rsab,gsnb, skip_rsab=skip1, skip_gsnb=skip2)

        rsab = self.project_out(rsab)
        gsnb = self.feature_corrector(gsnb)

        return rsab,gsnb


if __name__ == "__main__":
    model = CharFormer(
        dim=64,  # initial dimensions after input projection, which increases by 2x each stage
        stages=3,  # number of stages
        depth_RSAB=2,  # number of transformer blocks per RSAB
        depth_GSNB=1,  # number of Conv2d blocks per GSNB
        window_size=16,  # set window size (along one side) for which to do the attention within
        dim_head=32,
        heads=2,
    ).cuda()

    x = torch.randn(1, 3, 256, 256).cuda()
    output,add_feature = model(x)  # (1, 3, 256, 256)

    print(output.shape)
    print(add_feature.shape)

