import os
import math
from abc import abstractmethod
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
from tqdm import tqdm
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, t):
        pass
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, t):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        return x
def norm_layer(channels):
    return nn.GroupNorm(32, channels)
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x, t):
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)
    def forward(self, x):
        return self.op(x)
class UNetModel(nn.Module):
    def __init__(self,num_classes=10,in_channels=3,model_channels=128, out_channels=3, num_res_blocks=2, attention_resolutions=(8, 16),dropout=0,channel_mult=(1, 2, 2, 2),conv_resample=True,num_heads=4):
        super().__init__()
        self.num_classes=num_classes
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.label_embed=nn.Sequential(
            nn.Embedding(num_classes,model_channels),
            nn.Linear(model_channels,time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim,time_embed_dim)
        )
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )
    def forward(self, x, timesteps,y_label):
        t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        y_label=y_label.long()
        y_emb=self.label_embed(y_label)
        emb=t_emb+y_emb

        hs=[]
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    x = steps / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas

class GaussianDiffusion:
    def __init__(self,timesteps=1000,beta_schedule='linear'):
        self.timesteps = timesteps
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule=='cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    def p_mean_variance(self, model, x_t, t,y_label, clip_denoised=True):
        pred_noise = model(x_t, t,y_label)
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
    @torch.no_grad()
    def p_sample(self, model, x_t, t, y_label, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,y_label, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    @torch.no_grad()
    def p_sample_loop(self, model, shape,y_label):
        batch_size = shape[0]
        device = next(model.parameters()).device
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long),y_label)
            imgs.append(img.cpu().numpy())
        return imgs
    @torch.no_grad()
    def sample(self, model, image_size,y_label, batch_size=8, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size),y_label=y_label)
    def train_losses(self, model, x_start, t, y_label):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, y_label)
        loss = F.mse_loss(noise, predicted_noise)
        return loss

batch_size = 64

from torchvision.transforms import transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

from torch.utils.data import Dataset
transforms1=torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor()
    ]
)
class Data(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)
        for i in self.image_path:
            if not i.endswith('.png'):
                self.image_path.remove(i)
        self.class_to_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                             '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                             }
    def __getitem__(self, idx):
        image_name=self.image_path[idx]
        image_item_path=os.path.join(self.root_dir,self.label_dir,image_name)
        img=transforms1(Image.open(image_item_path).convert('L'))
        return img,torch.tensor(self.class_to_idx[image_name.strip('.png').split('_')[-1]])

    def __len__(self):
        return len(self.image_path)

all_train=[Data(root_dir='Number',label_dir=f'{i}') for i in range(0,10)]
all_train1=torch.utils.data.ConcatDataset(all_train)
train_loader = torch.utils.data.DataLoader(all_train1, batch_size=batch_size, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

grid_params={"timesteps":[500,1000],"beta_schedule":['linear','cosine']}
#网格搜索的超参数优化
best_loss=float('inf')
best_params=None
learning_rate=0.0001
num_a=1
for timesteps in grid_params['timesteps']:
    for beta_schedule in grid_params['beta_schedule']:
        print(f"Training with timesteps={timesteps}, beta_schedule={beta_schedule}")
        diffusion = GaussianDiffusion(timesteps=timesteps,beta_schedule=beta_schedule)
        model = UNetModel(
            num_classes=10,
            in_channels=1,
            model_channels=96,
            out_channels=1,
            channel_mult=(1, 2, 2),
            attention_resolutions=[]
        )
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

        num_epochs=100
        loss_for_epoch=[]
        for epoch in range(num_epochs):
            total_loss=0.0
            for step,(images,labels) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_size=images.shape[0]
                images1=images.to(device)
                labels1=labels.to(device).long()
                t=torch.randint(0,timesteps,(batch_size,),device=device)
                loss=diffusion.train_losses(model,images1,t,y_label=labels1)
                total_loss+=loss.item()
                loss.backward()
                optimizer.step()
            loss_for_epoch.append(total_loss)
            if (epoch+1)%10==0:
                print(f"Epoch :{epoch+1},loss={total_loss}")

        avg_loss = sum(loss_for_epoch) / len(loss_for_epoch)
        print(f"Average loss for timesteps={timesteps}, beta_schedule={beta_schedule}: {avg_loss}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {
                'timesteps': timesteps,
                'beta_schedule': beta_schedule
            }
            torch.save(model.state_dict(), f"best_model.pth")
        torch.save(model.state_dict(),f"model{num_a}.pth")
        num_a+=1