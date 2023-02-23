# encoding: utf-8
"""Attention U-net model [1] for segmentation

References
----------
[1] https://arxiv.org/abs/1804.03999
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, num_input_ch: int, num_output_ch: int, dropout_prob: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_input_ch, num_output_ch, kernel_size=(3, 3), padding=1),
            nn.Dropout2d(dropout_prob),
            nn.BatchNorm2d(num_output_ch),
            nn.ReLU(),
            nn.Conv2d(num_output_ch, num_output_ch, kernel_size=(3, 3), padding=1),
            nn.Dropout2d(dropout_prob),
            nn.BatchNorm2d(num_output_ch),
            nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, num_input_ch: int, num_output_ch: int, dropout_prob: float):
        super().__init__()
        self.deconv = nn.Sequential(
            (nn.ConvTranspose2d(num_input_ch, num_output_ch, kernel_size=(2, 2), stride=(2, 2))),
            nn.Dropout2d(dropout_prob),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.deconv(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, num_input_ch: int, ch_skip: int, num_output_ch: int, dropout_prob: float):
        super().__init__()
        self.W_skip = nn.Sequential(
            nn.Conv2d(ch_skip, num_output_ch, kernel_size=(1, 1), padding=0),
            nn.Dropout2d(dropout_prob),
            nn.BatchNorm2d(num_output_ch),
        )

        self.W_in = nn.Sequential(
            nn.Conv2d(num_input_ch, num_output_ch, kernel_size=(1, 1), padding=0),
            nn.Dropout2d(dropout_prob),
            nn.BatchNorm2d(num_output_ch),
        )

        self.relu = nn.ReLU()

        self.psi = nn.Sequential(
            nn.Conv2d(num_output_ch, 1, kernel_size=(1, 1), padding=0),
            nn.Dropout2d(dropout_prob),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_skip(skip)
        x = self.W_in(x)
        psi = self.relu(g + x)
        psi = self.psi(psi)
        return skip * psi


class Encoder(nn.Module):
    def __init__(self, num_input_ch: int, num_initial_filters: int, dropout_prob: float, depth: int = 4):
        super().__init__()
        self._num_input_ch = num_input_ch
        self._num_initial_filters = num_initial_filters
        self._dropout_rate = dropout_prob
        self._depth = depth

        first_block = ConvBlock(
            num_input_ch=self._num_input_ch, num_output_ch=self._num_initial_filters, dropout_prob=self._dropout_rate
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_blocks = nn.ModuleList([first_block])

        for idx in range(0, self._depth):
            num_channels = self._num_initial_filters * 2**idx
            self.conv_blocks.append(
                ConvBlock(num_input_ch=num_channels, num_output_ch=num_channels * 2, dropout_prob=self._dropout_rate)
            )

    def forward(self, x) -> list[torch.Tensor]:
        output = []
        x_in = self.conv_blocks[0](x)
        output.append(x_in)
        for idx in range(0, self._depth):
            x_out = self.max_pool(x_in)
            x_in = self.conv_blocks[idx + 1](x_out)
            output.append(x_in)

        return output


class Decoder(nn.Module):
    def __init__(self, num_initial_filters: int, dropout_prob: float, depth: int = 4):
        super().__init__()

        self.depth = depth

        self.conv_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()

        for divisor in range(0, depth):
            convolution_channels = num_initial_filters * 16 // 2**divisor
            self.conv_layers.append(
                ConvBlock(
                    num_input_ch=convolution_channels,
                    num_output_ch=convolution_channels // 2,
                    dropout_prob=dropout_prob,
                )
            )
            self.deconv_layers.append(
                DeconvBlock(
                    num_input_ch=convolution_channels,
                    num_output_ch=convolution_channels // 2,
                    dropout_prob=dropout_prob,
                )
            )

            attention_channels = num_initial_filters * 8 // 2**divisor
            self.attention_layers.append(
                AttentionBlock(
                    num_input_ch=attention_channels,
                    ch_skip=attention_channels,
                    num_output_ch=attention_channels // 2,
                    dropout_prob=dropout_prob,
                )
            )

    def forward(self, skip_output) -> torch.Tensor:
        d5 = self.deconv_layers[0](skip_output[-1])
        s4 = self.attention_layers[0](x=d5, skip=skip_output[-2])
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.conv_layers[0](d5)

        d4 = self.deconv_layers[1](d5)
        s3 = self.attention_layers[1](x=d4, skip=skip_output[-3])
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.conv_layers[1](d4)

        d3 = self.deconv_layers[2](d4)
        s2 = self.attention_layers[2](x=d3, skip=skip_output[-4])
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.conv_layers[2](d3)

        d2 = self.deconv_layers[3](d3)
        s1 = self.attention_layers[3](x=d2, skip=skip_output[-5])
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.conv_layers[3](d2)

        return d2


class OutputLayer(nn.Module):
    def __init__(self, num_initial_filters: int, num_output_ch: int):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv2d(
                in_channels=num_initial_filters,
                out_channels=num_output_ch,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
            ),
        )

    def __call__(self, x):
        output_layer = self.output(x)
        return output_layer


class AttentionUnet(nn.Module):
    def __init__(
        self,
        num_input_ch: int,
        num_classes: int,
        num_initial_filters: int,
        depth: int,
        dropout_prob: float,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_input_ch=num_input_ch,
            num_initial_filters=num_initial_filters,
            dropout_prob=dropout_prob,
            depth=depth,
        )
        self.decoder = Decoder(num_initial_filters=num_initial_filters, dropout_prob=dropout_prob, depth=depth)
        self.output_layer = OutputLayer(
            num_initial_filters=num_initial_filters,
            num_output_ch=num_classes,
        )
        self.num_classes = num_classes

    def forward(self, x) -> torch.Tensor:
        skip_features = self.encoder(x)
        decoder_output = self.decoder(skip_features)
        output = self.output_layer(decoder_output)
        return output
