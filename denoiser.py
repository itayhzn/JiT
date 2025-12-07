import torch
import torch.nn as nn
import torch.fft
import math
from model_jit import JiT_models


def get_dct_matrix(N):
    # Returns the DCT-II matrix of shape (N, N)
    # C[k, n] = cos(pi * k * (2n + 1) / (2N))
    # X[k] = sum_n x[n] * C[k, n] * scale
    
    n = torch.arange(N).float()
    k = torch.arange(N).float()
    n, k = torch.meshgrid(n, k, indexing='ij') # (N, N)
    
    k, n = torch.meshgrid(torch.arange(N).float(), torch.arange(N).float(), indexing='ij')
    M = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
    
    M[0, :] *= math.sqrt(1 / N)
    M[1:, :] *= math.sqrt(2 / N)
    
    return M

def dct_2d(x, norm='ortho'):
    """
    2D Discrete Cosine Transform (Type II) via Matrix Multiplication
    """
    # x: (..., H, W)
    H, W = x.shape[-2:]
    device = x.device
    
    M_h = get_dct_matrix(H).to(device)
    M_w = get_dct_matrix(W).to(device)
    
    dct_h = torch.matmul(M_h, x)
    dct = torch.matmul(dct_h, M_w.t())
    
    return dct

def idct_2d(x, norm='ortho'):
    """
    2D Inverse Discrete Cosine Transform via Matrix Multiplication
    """
    # x: (..., H, W)
    H, W = x.shape[-2:]
    device = x.device
    
    M_h = get_dct_matrix(H).to(device)
    M_w = get_dct_matrix(W).to(device)
    
    idct_h = torch.matmul(M_h.t(), x)
    idct = torch.matmul(idct_h, M_w)
    
    return idct

def spectral_transform(x, transform_type='dct'):
    if transform_type == 'dft':
        # x: (N, 3, H, W)
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft = torch.view_as_real(x_fft) # (N, 3, H, W, 2)
        x_fft = x_fft.permute(0, 1, 4, 2, 3).contiguous() # (N, 3, 2, H, W)
        x_fft = x_fft.view(x.size(0), -1, x.size(2), x.size(3)) # (N, 6, H, W)
        return x_fft
    elif transform_type == 'dct':
        return dct_2d(x, norm='ortho')
    else:
        raise NotImplementedError(f"Transform {transform_type} not implemented")

def inverse_spectral_transform(x, transform_type='dct'):
    if transform_type == 'dft':
        # x: (N, 6, H, W)
        B, C, H, W = x.shape
        x = x.view(B, 3, 2, H, W)
        x = x.permute(0, 1, 3, 4, 2).contiguous() # (N, 3, H, W, 2)
        x = torch.view_as_complex(x)
        x = torch.fft.ifft2(x, norm='ortho')
        return x.real
    elif transform_type == 'dct':
        return idct_2d(x, norm='ortho')
    else:
        raise NotImplementedError(f"Transform {transform_type} not implemented")

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.spectral = getattr(args, 'spectral', False)
        self.transform_type = getattr(args, 'transform_type', 'dft')
        self.transform_type = getattr(args, 'transform_type', 'dct')
        
        in_channels = 3
        if self.spectral:
            if self.transform_type == 'dft':
                in_channels = 6
            elif self.transform_type == 'dct':
                in_channels = 3
            else:
                raise NotImplementedError(f"Transform {self.transform_type} not implemented")

        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=in_channels,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        # x is expected to be in the correct domain (pixel or spectral)
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        
        in_channels = 3
        if self.spectral:
            if self.transform_type == 'dft':
                in_channels = 6
            elif self.transform_type == 'dct':
                in_channels = 3
        
        z = self.noise_scale * torch.randn(bsz, in_channels, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        
        # Return z in the domain it was generated (pixel or spectral)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)


