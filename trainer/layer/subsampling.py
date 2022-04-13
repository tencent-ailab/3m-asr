#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""Subsampling layer definition."""

import torch


class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1
        self.in_ch = 1

    def trans_3d_to_4d(self, x):
        assert x.dim() == 3 and x.size(2) % self.in_ch == 0
        b, t, f = x.size()
        trans_x = x.view(b, t, self.in_ch, f // self.in_ch).transpose(1, 2).contiguous()
        return trans_x


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, idim: int, odim: int, dropout_rate: float = 0.0):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-12),
            torch.nn.Dropout(dropout_rate),
        )
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .

        """
        x = self.out(x)
        return x, x_mask


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension
        odim (int): Output dimension.

    """
    def __init__(self, idim: int, odim: int, in_ch: int = 1):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) / 2 * stride  * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) / 2 * 2 * 1 + (3 - 1) / 2 * 2 * 2
        self.right_context = 6
        self.in_ch = in_ch

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.

        """
        x = self.trans_3d_to_4d(x)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
    """
    def __init__(self, idim: int, odim: int, in_ch: int = 1):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3),
                                      odim)
        # 14 = (3 - 1) / 2 * 2 * 1 + (5 - 1) / 2 * 3 * 2
        self.subsampling_rate = 6
        self.right_context = 14

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
        """
        x = self.trans_3d_to_4d(x)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.

    """
    def __init__(self, idim: int, odim: int, in_ch: int = 1):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.subsampling_rate = 8
        # 14 = (3 - 1) / 2 * 2 * 1 + (3 - 1) / 2 * 2 * 2 + (3 - 1) / 2 * 2 * 4
        self.right_context = 14

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
        """
        x = self.trans_3d_to_4d(x)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
