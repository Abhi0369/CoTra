import torch
import math
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from configs import config
from einops import rearrange




class PatchEmbeddings(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches from the image size and patch size

   

        self.projection = nn.Linear(self.patch_size * self.num_channels, self.hidden_size,bias=False)




    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)

        p = self.patch_size 
    
        

        x = rearrange(x, 'b (c) (h p1) -> b (h) (p1 c)  ', p1 = p)

    
        x = self.projection(x)

        return x

class Pos_Embed(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        max_len = 625
        embed_dim=config["hidden_size"]
        pe=torch.zeros(max_len,embed_dim)
        position=torch.arange(0,max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,embed_dim,2)*-(math.log(100000.0)/embed_dim))
        
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x=Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return x
    
class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]

        self.num_patches = (self.image_size)// (self.patch_size) 
        self.pos_embed= nn.Parameter(torch.randn(1, 8, config["hidden_size"]))


    def forward(self, x):
        x = self.patch_embeddings(x)

        x = x * self.pos_embed


        return x




class Correlator(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.batch_size = config['batch_size']
        self.num_corr  = config["num_corr"]
        
        
#         self.mlp = MLP(config)

        self.num_patches = ((self.image_size) // (self.patch_size)) ** 2

        self.q_proj = nn.Linear(self.hidden_size,self.hidden_size,bias=config['qkv_bias'])
        self.k_proj = nn.Linear(self.hidden_size,self.hidden_size,bias=config['qkv_bias'])
        self.v_proj = nn.Linear(self.hidden_size,self.hidden_size,bias=config['qkv_bias'])
   
        self.Pos_embed = Pos_Embed(config)

    def forward(self,x,x_1,output_attentions=False):

        query = self.q_proj(x_1)
        
        key = self.k_proj(x)
#         v= self.proj(self.pos_embed(h))
        value = self.v_proj(self.Pos_embed(x))
        pos = self.Pos_embed(x)
        # print(pos)
        x2 = torch.matmul(query,key.transpose(-1,-2))
     
        all_attention = x2

        x2 = torch.matmul(x2,value)/math.sqrt(self.hidden_size)
       

        if not output_attentions:
            return (x2, x_1,all_attention)
        else:
            return (x2, x_1 ,all_attention)



class Encoder(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.num_corr = config["num_corr"]
        
        for _ in range(self.num_corr-1):
            block = Correlator(config)
            self.blocks.append(block)

    def forward(self,x, output_attention=True):
        op = x
        # inp=x
        outp = [op.mean((-1,-2))]
        all_attention =[]

#         outp = torch.empty((64,self.num_corr))
        for (i,block) in enumerate(self.blocks):
                       
            op,inp,attention= block(op,x,output_attentions=output_attention)

            outp.append(op.mean((-1,-2)))
            all_attention.append(torch.unsqueeze(attention,1))

    
        op=torch.stack(outp,dim=1)
                       
        if not output_attention:
            return (op,None)
        else:
            return (op,all_attention)

        
class ViT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.patch_size = config["patch_size"]
        self.num_classes = config["num_classes"]
        self.num_corr = config["num_corr"]
        # Create the embedding module
        self.embedding = Embeddings(config)


        self.encoder = Encoder(config)

        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.num_corr, self.num_classes,bias=True)



    def forward(self, x, output_attentions=False):
        # Calculate the embedding output

        embedding_output = self.embedding(x)


        # Calculate the encoder's output
        op,all_attention= self.encoder(embedding_output,output_attentions)

        self.output = op
#         self.output = self.bn(op)
        

        logits = self.classifier(self.output)

        if not output_attentions:
            return (logits,None, None)
        else:
            return (logits, all_attention,None)
