import torch 
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,in_channels=3): #in_channels set to 3 for RGB
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)   # (1,8,8) → (8,8,8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)  # (8,4,4) → (16,4,4)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 + relu + pool → (8,4,4)
        x = self.pool(F.relu(self.conv2(x)))  # conv2 + relu + pool → (16,2,2)
        return x  # latent shape: (16, 2, 2)
    
class Decoder(nn.Module):
    def __init__(self,in_channels=16):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=8, kernel_size=2, stride=2)   # (16,2,2) → (8,4,4)
        self.deconv2 = nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=2, stride=2)  # (8,4,4) → (1,8,8)

    def forward(self, x):
        x = F.relu(self.deconv1(x))  # conv1 + relu + pool → (8,4,4)
        x = F.sigmoid(self.deconv2(x))  # conv2 + relu + pool → (1,8,8)
        return x
    
class ConvAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(ConvAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
