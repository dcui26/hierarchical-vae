import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoderconv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1) #32x32->16x16
        self.encoderbn1 = nn.BatchNorm2d(64)
        self.encoderconv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1) #16x16->8x8
        self.encoderbn2 = nn.BatchNorm2d(128)
        self.encoderconv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1) #8x8->4x4
        self.encoderbn3 = nn.BatchNorm2d(256)
        self.encoder_mu = nn.Linear(256*4*4, 256)
        self.encoder_logvar = nn.Linear(256*4*4, 256)
        self.decoder_fc = nn.Linear(256, 256*4*4)
        self.decoderconv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.decoderbn1 = nn.BatchNorm2d(128)
        self.decoderconv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.decoderbn2 = nn.BatchNorm2d(64)
        self.decoderconv3 = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def encode(self, x):
        x = self.relu(self.encoderbn1(self.encoderconv1(x)))
        x = self.relu(self.encoderbn2(self.encoderconv2(x)))
        x = self.relu(self.encoderbn3(self.encoderconv3(x)))
        x = x.view(-1, 256*4*4)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.relu(self.decoder_fc(z))
        x = x.view(-1, 256, 4, 4)
        x = self.relu(self.decoderbn1(self.decoderconv1(x)))
        x = self.relu(self.decoderbn2(self.decoderconv2(x)))
        x = self.tanh(self.decoderconv3(x))
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        epsilon = torch.randn_like(logvar)
        std = torch.exp(0.5*logvar)
        z = mu + std * epsilon
        pred = self.decode(z)
        return pred, mu, logvar
    
    @staticmethod
    def beta_schedule(epoch):
        if epoch < 20:
            beta = epoch / 20.0
        elif epoch < 30:
            beta = 1.0
        else:
            beta = max(0.4, 1.0 - (epoch - 20) / 100)
        return beta

    @staticmethod
    def compute_loss(pred, target, mu, logvar):
        recon_loss = F.mse_loss(pred, target, reduction='sum')
        KL_loss = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
        return recon_loss, KL_loss
    