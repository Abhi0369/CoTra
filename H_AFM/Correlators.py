from opt_einsum import contract
import numpy as np
from helper_functions import load_experiment
import torch
def corr2(p_no):
    config, model, train_losses, test_losses, accuracies= load_experiment("my_exp")
    
    w = model.embedding.patch_embeddings.projection.weight
    wp=np.asarray(w.cpu().detach().numpy()).transpose()

    pe = model.embedding.pos_embed.cpu().detach().numpy()
    pe = np.squeeze(pe)
    # print(pe.shape)
    Pe = np.stack((pe[p_no,:],pe.transpose()[:,p_no]))

        
    q = model.encoder.blocks[0].q_proj.weight
    k = model.encoder.blocks[0].k_proj.weight
    v = model.encoder.blocks[0].v_proj.weight


    cl = model.classifier.weight.cpu().detach().numpy()


    wq = np.asarray(q.cpu().detach().numpy()).transpose()
    wk = np.asarray(k.cpu().detach().numpy()).transpose()           
    wv = np.asarray(v.cpu().detach().numpy()).transpose()
     
    op = contract('k,l,rk,ls,km,ml->rs',Pe[0],Pe[1],wp,wp.transpose(),wq,wk.transpose())
    # op = contract('rk,ls,km,ml->rs',wp,wp.transpose(),wq,wk.transpose())

    return op,accuracies[-1]

