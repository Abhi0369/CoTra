from opt_einsum import contract
import numpy as np
from helper_functions import load_experiment
import torch
from configs import config

def corr3(p_no):
    config, model, train_losses, test_losses, accuracies = load_experiment("my_exp")
    d = config["hidden_size"]
    w = model.embedding.patch_embeddings.projection.weight
    wp=np.asarray(w.cpu().detach().numpy()).transpose()

    pe = model.embedding.pos_embed.cpu().detach().numpy()
    pe = np.squeeze(pe)

    Pe = np.stack((pe[p_no,:],pe.transpose()[:,p_no]))


        
    q1 = model.encoder.blocks[0].q_proj.weight
    k1 = model.encoder.blocks[0].k_proj.weight
    v1 = model.encoder.blocks[0].v_proj.weight
    q2 = model.encoder.blocks[1].q_proj.weight
    k2 = model.encoder.blocks[1].k_proj.weight
    v2 = model.encoder.blocks[1].v_proj.weight


    wq1 = np.asarray(q1.cpu().detach().numpy()).transpose()
    wk1 = np.asarray(k1.cpu().detach().numpy()).transpose()           
    wv1 = np.asarray(v1.cpu().detach().numpy()).transpose()
    wq2 = np.asarray(q2.cpu().detach().numpy()).transpose()
    wk2 = np.asarray(k2.cpu().detach().numpy()).transpose()           
    wv2 = np.asarray(v2.cpu().detach().numpy()).transpose()
    p = model.encoder.blocks[0].Pos_embed(torch.zeros((1,8,d))).squeeze(0).detach().numpy()
    

    op = contract('a,k,l,ea,kr,sl,mk,ac,lm,cb,bn,n->ers',Pe[0],Pe[1],Pe[0],wp,wp.transpose(),wp,wq1.transpose(),wq2,wk1,wk2.transpose(),wv1.transpose(),p.transpose()[:,p_no])
    

    return op,accuracies[-1]
