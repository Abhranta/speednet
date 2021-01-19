import torch.nn as nn
import torch
from utils import Flatten , Unflatten , weights_init , down_conv , up_conv

class Net(nn.Module):
    def __init__(self , num_layers , img_dim , in_chan , act_func , latent_vector_size):
        
        super(Net , self).__init__()
        assert act_func in ("ReLU" , "LeakyReLU") , "Activation function that can be used now are ReLU and LeakyReLU"
        assert img_dim % (2**(num_layers)) >= 0 , "Latent vector driven to 0, please increase image size or decreasenumber of layers"
        
        self.act_func = act_func
        self.in_chan = in_chan
        self.num_layers = num_layers
        self.latent_vector_size = latent_vector_size
        self.in_chan2 = self.in_chan
        self.encoder_net_layers = []
        self.decoder_net_layers = []
        
        self.out_chan = 2**5
        for _ in range(num_layers):
            
            self.encoder_net_layers.append(down_conv(self.in_chan , self.act_func , self.out_chan))
            self.in_chan = self.out_chan*2
            self.out_chan = self.out_chan*4
        self.encoder = nn.Sequential(*self.encoder_net_layers , 
                                     Flatten() , 
                                     nn.Linear(((self.out_chan//2)*((img_dim//(2 ** num_layers))**2)) , self.latent_vector_size*4) , 
                                     nn.ReLU(),
                                     nn.Linear(self.latent_vector_size*4 , self.latent_vector_size*2) , 
                                     nn.ReLU()
                                    )
        self.mu = nn.Linear(self.latent_vector_size*2 , self.latent_vector_size)
        self.logvar = nn.Linear(self.latent_vector_size*2 , self.latent_vector_size)
        self.out_chan2 = self.out_chan
        
        for _ in range(num_layers):
            self.decoder_net_layers.append(up_conv(self.out_chan2//2 , self.act_func , self.out_chan2//4))
            self.out_chan2 = self.out_chan2//4
        
        
        
        self.decoder = nn.Sequential(nn.Linear(self.latent_vector_size , self.latent_vector_size*4) , 
                                    nn.ReLU() ,  
                                    nn.Linear(self.latent_vector_size*4 , ((self.out_chan//2)*((img_dim//(2 ** num_layers))**2))) , 
                                    nn.ReLU() , 
                                    Unflatten(self.out_chan//2 , (img_dim//(2 ** num_layers)) , (img_dim//(2 ** num_layers)) ) , 
                                    *self.decoder_net_layers , 
                                    nn.ConvTranspose2d(self.out_chan2//2 , self.in_chan2 , 3 , 1 , 1))
        
    def encode(self , input_tensor):
        encoded_vector = self.encoder(input_tensor)
        mu , logvar = self.mu(encoded_vector) , self.logvar(encoded_vector)
        return mu , logvar
    
    def reparameterize(self , mu , logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + std*eps
        return latent
    
    def decode(self , latent):
        decoded_vector = self.decoder(latent)
        return decoded_vector
    
    def forward(self , input_tensor):
        mu , logvar = self.encode(input_tensor)
        latent_space = self.reparameterize(mu , logvar)
        return self.decode(latent_space) , mu , logvar