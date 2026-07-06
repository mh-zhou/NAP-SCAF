import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models.densenet import _DenseBlock
from einops import rearrange
sys.path.append(os.path.realpath('..'))

class _Transition(nn.Sequential):

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.Upsample(scale_factor=2.0))

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        num_init_features = 64
        self.encoder = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.encoder.features[0] = nn.Conv2d(4, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Conv2d(1024, 2 * 1024, 3, 1, 1)

    def forward(self, x):
        x = self.encoder.features[0](x)
        x = self.encoder.features[1](x)
        x = self.encoder.features[2](x)
        x = self.encoder.features[3](x)
        x = self.encoder.features[4](x)
        x = self.encoder.features[5](x)
        x = self.encoder.features[6](x)
        x = self.encoder.features[7](x)
        x = self.encoder.features[8](x)
        x = self.encoder.features[9](x)
        x = self.encoder.features[10](x)
        x = self.encoder.features[11](x)
        x = self.fc(x)
        mu, logvar = x.chunk(2, dim=1)
        return (mu, logvar)

def reparameterize(mu, logvar):
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std

class Decoder(nn.Module):

    def __init__(self, cdim=4, zdim=1024):
        super().__init__()
        self.fc = nn.Sequential(nn.Conv2d(zdim, 1024, 3, 1, 1), nn.ReLU(True))
        self.decoder = nn.Sequential(_Transition(1024, 512), _DenseBlock(num_layers=16, num_input_features=512, bn_size=4, growth_rate=32, drop_rate=0), _Transition(1024, 256), _DenseBlock(num_layers=8, num_input_features=256, bn_size=4, growth_rate=32, drop_rate=0), _Transition(512, 128), _DenseBlock(num_layers=4, num_input_features=128, bn_size=4, growth_rate=32, drop_rate=0), _Transition(256, 64), _DenseBlock(num_layers=2, num_input_features=64, bn_size=4, growth_rate=32, drop_rate=0), _Transition(128, 32), nn.Conv2d(32, cdim, 3, 1, 1))

    def forward(self, x):
        x = self.fc(x)
        x = self.decoder(x)
        return x

class Conv3x3(nn.Module):

    def __init__(self, inc, outc, stride):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, 3, stride, 1)
        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class NAP_Path(nn.Module):

    def __init__(self, inc, midc, stages):
        super().__init__()
        self.inc, self.stages = (inc, stages)
        self.vae = nn.ModuleDict({'enc': Encoder(), 'dec': Decoder()})
        self.inconv = Conv3x3(4, midc, 1)
        self.enc = nn.ModuleList()
        stagec = midc
        for _ in range(stages):
            self.enc.append(Conv3x3(stagec, stagec * 2, 2))
            stagec *= 2
        self.dec = nn.ModuleList()
        for _ in range(stages):
            self.dec.append(nn.ModuleList([nn.ConvTranspose2d(stagec, stagec // 2, 2, 2), Conv3x3(stagec, stagec // 2, 1)]))
            stagec //= 2

    @torch.no_grad()
    def reconstruct(self, x):
        mu, logvar = self.vae['enc'](x)
        z = reparameterize(mu, logvar)
        rec = self.vae['dec'](z)
        return rec

    def forward(self, x):
        rec = self.reconstruct(x)
        x = self.inconv(rec)
        enc_feas = []
        for layer in self.enc:
            x = layer(x)
            enc_feas.append(x)
        fea = enc_feas[-1]
        dec_feas = []
        for i, (up, merge) in enumerate(self.dec):
            fea = up(fea)
            if i < len(enc_feas) - 1:
                fea = merge(torch.cat([fea, enc_feas[-2 - i]], 1))
            dec_feas.append(fea)
        return dec_feas

class Attention(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        self.dim = dim

    def forward(self, x_in):
        b, c, h, w = x_in.shape
        x = x_in.permute(0, 2, 3, 1).reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inp, k_inp, v_inp))
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = k @ q.transpose(-2, -1) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c).permute(0, 3, 1, 2)
        out_p = self.pos_emb(x_in)
        return out_c + out_p

class FeedForward(nn.Module):

    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim, dim * mult, 1, 1, bias=False), nn.GELU(), nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult), nn.GELU(), nn.Conv2d(dim * mult, dim, 1, 1, bias=False))

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.ff = FeedForward(dim=dim)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class CERA(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.local1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.local2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels, bias=False)
        self.local3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3, groups=in_channels, bias=False)
        self.bg = nn.AvgPool2d(7, stride=1, padding=3)
        self.fuse = nn.Sequential(nn.Conv2d(in_channels * 4, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels), nn.Sigmoid())

    def forward(self, x):
        b = self.bg(x)
        c = (x - b).abs()
        f = torch.cat([self.local1(x), self.local2(x), self.local3(x), c], dim=1)
        att = self.fuse(f)
        return x * att + x

class SWG(nn.Module):

    def __init__(self, gch, xch, midc, alpha_init=0.1):
        super().__init__()
        self.q = nn.Conv2d(gch, midc, 1, bias=False)
        self.k = nn.Conv2d(xch, midc, 1, bias=False)
        init = torch.log(torch.tensor(alpha_init) / (1 - torch.tensor(alpha_init)))
        self.logit_alpha = nn.Parameter(init.view(1, 1, 1, 1))

    def forward(self, g, x):
        q = F.normalize(self.q(g), dim=1, eps=1e-06)
        k = F.normalize(self.k(x), dim=1, eps=1e-06)
        cos_sim = (q * k).sum(dim=1, keepdim=True)
        sim = (cos_sim + 1.0) * 0.5
        alpha = torch.sigmoid(self.logit_alpha)
        w = 1.0 - alpha * sim
        return x * w

class SWC_Gate(nn.Module):

    def __init__(self, gch, xch, midc):
        super().__init__()
        self.gconv = nn.Conv2d(gch, midc, 1, bias=False)
        self.xconv = nn.Conv2d(xch, midc, 1, bias=False)
        self.psi = nn.Conv2d(midc, 1, 1, bias=False)
        self.cera = CERA(xch)
        self.swg = SWG(gch, xch, midc, alpha_init=0.1)

    def forward(self, g, x):
        g1 = self.gconv(g)
        x1 = self.xconv(x)
        hint = torch.relu(-g1 * x1)
        hint = torch.sigmoid(self.psi(hint))
        x_enh = self.cera(x)
        x_soft = self.swg(g, x_enh)
        out = x_soft * hint
        return (out, (g1, x1))

class NAP_SCAF(nn.Module):

    def __init__(self, inc, outc, midc=16, stages=4, gate_passes: int=2):
        super().__init__()
        assert gate_passes >= 2, '建议 gate_passes>=2，才能体现前后双阶段校准'
        self.inc, self.stages = (inc, stages)
        self.gate_passes = gate_passes
        self.nap_path = NAP_Path(inc, midc, stages)
        self.inconvs = nn.ModuleList([Conv3x3(1, midc, 1) for _ in range(inc)])
        self.encs = nn.ModuleList()
        for _ in range(inc):
            enc = nn.ModuleList()
            stagec = midc
            for _ in range(stages):
                enc.append(Conv3x3(stagec, stagec * 2, 2))
                stagec *= 2
            self.encs.append(enc)
        self.fusion = nn.ModuleList()
        stagec = midc
        for _ in range(stages):
            stagec = stagec * 2
            self.fusion.append(nn.Sequential(Transformer(dim=inc * stagec, dim_head=midc, heads=stagec // midc), Conv3x3(inc * stagec, stagec, 1)))
        self.dec = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.convouts = nn.ModuleList()
        for k in range(stages):
            self.dec.append(nn.ModuleList([nn.ConvTranspose2d(stagec, stagec // 2, 2, 2), Conv3x3(stagec, stagec // 2, 1)]))
            stagec //= 2
            self.gates.append(SWC_Gate(gch=stagec, xch=stagec, midc=stagec // 2))
            self.convouts.append(nn.Conv2d(stagec, outc, 1, 1, 0))

    def forward(self, x):
        nap_feats = self.nap_path(x)
        xin = torch.chunk(x, self.inc, 1)
        enc_stacks = []
        for i in range(self.inc):
            feas = []
            xi = self.inconvs[i](xin[i])
            for layer in self.encs[i]:
                xi = layer(xi)
                feas.append(xi)
            enc_stacks.append(feas)
        fused = []
        for k in range(self.stages):
            feas = [feas[k] for feas in enc_stacks]
            fea = torch.cat(feas, 1)
            fea = self.fusion[k](fea)
            fused.append(fea)
        outs = []
        fea = fused[-1]
        for i, (up, merge) in enumerate(self.dec):
            fea = up(fea)
            gfeat = nap_feats[i]
            fea, _ = self.gates[i](gfeat, fea)
            if i < len(fused) - 1:
                fea = merge(torch.cat([fea, fused[-2 - i]], 1))
            for _ in range(self.gate_passes - 1):
                fea, _ = self.gates[i](gfeat, fea)
            outs.append(self.convouts[i](fea))
        return outs[-1]
if __name__ == '__main__':
    input_channels = 4
    output_channels = 4
    mid_channels = 16
    stages = 4
    nap_scaf_model = NAP_SCAF(inc=input_channels, outc=output_channels, midc=mid_channels, stages=stages)
    batch_size = 16
    image_height, image_width = (256, 256)
    input_tensor = torch.randn(batch_size, input_channels, image_height, image_width)
    output_train = nap_scaf_model(input_tensor)
    print('Train Segmentation Output Shape:', tuple(output_train.shape))
