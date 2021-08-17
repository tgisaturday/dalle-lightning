import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_g_loss(logits_fake):
    return torch.mean(F.relu(1. - logits_fake))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        patch_size,
        n_res_blocks,
        in_channel,
        channel,
        n_res_block,
        n_res_channel,
    ):
        super().__init__()
        blocks = [
            K.RandomCrop((patch_size, patch_size), cropping_mode='resample'),
            nn.Conv2d(in_channel, channel, 3, padding=1),
        ]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.extend([
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1),
            nn.AvgPool2d(patch_size),
            nn.Flatten(),
            nn.Linear(channel, 1)
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

    def d_loss(self, reals, fakes):
        real_preds = self.forward(reals)
        fake_preds = self.forward(fakes)

        return hinge_d_loss(real_preds, fake_preds)

    def g_loss(self, fakes):
        return hinge_g_loss(self.forward(fakes))


class PatchReconstructionDiscriminator(PatchDiscriminator):
    def __init__(
        self,
        patch_size,
        n_res_blocks,
        in_channel,
        channel,
        n_res_block,
        n_res_channel,
    ):
        super().__init__(
            patch_size,
            n_res_blocks,
            in_channel * 2,
            channel,
            n_res_block,
            n_res_channel,
        )

    def forward(self, x, y):
        assert x.shape == y.shape
        input = torch.stack([x, y], dim=1)
        return super().forward(input)

    def d_loss(self, reals, fakes):
        assert reals.shape == fakes.shape
        reals_1, reals_2 = reals.view(2, reals.size(0) // 2, -1, -1, -1)
        fakes_1, fakes_2 = fakes.view(2, fakes.size(0) // 2, -1, -1, -1)
        logits_real = self.forward(reals_1, fakes_1)
        logits_fake = self.forward(fakes_1, reals_1)
        return hinge_d_loss(logits_real, logits_fake)

    def g_loss(self, reals, fakes):
        assert reals.shape == fakes.shape
        reals_1, reals_2 = reals.view(2, reals.size(0) // 2, -1, -1, -1)
        fakes_1, fakes_2 = fakes.view(2, fakes.size(0) // 2, -1, -1, -1)
        logits_real = self.forward(reals_1, fakes_1)
        logits_fake = self.forward(fakes_1, reals_1)
        return hinge_d_loss(logits_fake, logits_real)
