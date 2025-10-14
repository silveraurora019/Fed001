import torch
import torch.nn as nn
import torch.fft as fft
from typing import Tuple

class FFT2DDecompose(nn.Module):
    """
    将 BCHW 的实数特征做 2D FFT，输出 (real, imag) 两个实数张量，形状均为 BCHW
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_fft = fft.fft2(x, dim=(-2, -1))
        real = x_fft.real
        imag = x_fft.imag
        return real, imag

class IFFT2DReconstruct(nn.Module):
    """
    将 (real, imag) 两个实数张量重组成复数，再做 2D IFFT，返回实部（BCHW）
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        x_complex = torch.complex(real, imag)
        x_ifft = fft.ifft2(x_complex, dim=(-2, -1))
        return x_ifft.real

class FrequencyEnhanceBlock(nn.Module):
    """
    频域增强块：空间域→FFT→(实/虚拼接)→逐通道1x1变换→拆分实/虚→IFFT→空间域
    """
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.fft = FFT2DDecompose()
        self.conv = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=in_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=in_channels * 2
        )
        self.ifft = IFFT2DReconstruct()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = self.fft(x)
        freq_cat = torch.cat([real, imag], dim=1)
        freq_out = self.conv(freq_cat)
        c = freq_out.shape[1] // 2
        real_new = freq_out[:, :c, :, :]
        imag_new = freq_out[:, c:, :, :]
        x_out = self.ifft(real_new, imag_new)
        return x_out