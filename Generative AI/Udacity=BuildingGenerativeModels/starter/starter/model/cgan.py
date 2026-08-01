"""
cgan.py
-------
Defines Generator and Discriminator architectures for Conditional GAN.
"""

import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.z_dim = z_dim
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(num_classes, num_classes)
        num_g_filters = 64
        num_channels = 1
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=z_dim + num_classes,
                out_channels=num_g_filters * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=num_g_filters * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=num_g_filters * 8,
                out_channels=num_g_filters * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=num_g_filters * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=num_g_filters * 4,
                out_channels=num_g_filters * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=num_g_filters * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=num_g_filters * 2,
                out_channels=num_g_filters,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=num_g_filters),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=num_g_filters,
                out_channels=num_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        labels = self.label_emb(labels)
        labels = labels.view(labels.shape[0], -1, 1, 1)
        x = torch.cat((noise, labels), dim=1)
        img = self.model(x)
        return img
        #return img.view(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.uplabel_embed = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 64*64)
        )
        
        num_channels = 2
        num_d_filters = 64
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_d_filters,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(
                in_channels=num_d_filters,
                out_channels=num_d_filters * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=num_d_filters * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(
                in_channels=num_d_filters * 2,
                out_channels=num_d_filters * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=num_d_filters * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(
                in_channels=num_d_filters * 4,
                out_channels=num_d_filters * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=num_d_filters * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(
                in_channels=num_d_filters * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            #nn.Sigmoid()
        )

    def forward(self, img, labels):
        labels = self.label_emb(labels)
        labels = self.uplabel_embed(labels)
        labels = labels.view(labels.shape[0], 1, 64, 64)
        x = torch.cat((img, labels), dim=1)
        logits = self.model(x)
        return logits