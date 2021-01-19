import torch.nn.fnctional as F
import torch

def conv_vae_loss(recon_x , x , mu , logvar):
    BCE = F.binary_cross_entropy(recon_x ,x , reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return torch.mean(BCE + KLD)