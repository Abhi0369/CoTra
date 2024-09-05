from opt_einsum import contract
import numpy as np
from helper_functions import load_experiment
import torch

def corr4(p_no):
    config, model, train_losses, test_losses, accuracies = load_experiment("my_exp")
    
    w = model.embedding.patch_embeddings.projection.weight                                               
    wp=np.asarray(w.cpu().detach().numpy()).transpose()
    d = config["hidden_size"]
#     pe = model.embedding.pos_embed.cpu().detach().numpy()
#     pe = np.squeeze(pe)
#     print(pe.shape)
#     Pe = np.stack((pe[0,:],pe.transpose()[:,0]))
        
    q1 = model.encoder.blocks[0].q_proj.weight
    k1 = model.encoder.blocks[0].k_proj.weight
    v1 = model.encoder.blocks[0].v_proj.weight
    q2 = model.encoder.blocks[1].q_proj.weight
    k2 = model.encoder.blocks[1].k_proj.weight
    v2 = model.encoder.blocks[1].v_proj.weight
    q3 = model.encoder.blocks[2].q_proj.weight
    k3 = model.encoder.blocks[2].k_proj.weight
    v3 = model.encoder.blocks[2].v_proj.weight


    cl = model.classifier.weight.cpu().detach().numpy()


    wq1 = np.asarray(q1.cpu().detach().numpy()).transpose()
    wk1 = np.asarray(k1.cpu().detach().numpy()).transpose()           
    wv1 = np.asarray(v1.cpu().detach().numpy()).transpose()
    wq2 = np.asarray(q2.cpu().detach().numpy()).transpose()
    wk2 = np.asarray(k2.cpu().detach().numpy()).transpose()           
    wv2 = np.asarray(v2.cpu().detach().numpy()).transpose()
    p1 = model.encoder.blocks[0].pos_embed(torch.randn((1,64,6))).squeeze(0).detach().numpy()

    wq3 = np.asarray(q3.cpu().detach().numpy()).transpose()
    wk3 = np.asarray(k3.cpu().detach().numpy()).transpose()           
    wv3 = np.asarray(v3.cpu().detach().numpy()).transpose()
    p2 = model.encoder.blocks[1].pos_embed(torch.randn((1,64,d))).squeeze(0).detach().numpy()
    
   
    op = contract('hp,ae,rk,ls,km,ca,pt,ml,cb,tq,nb,qo,n,o->hers',wp,wp.transpose(),wp,wp.transpose(),
                  wq1,wq2.transpose(),wq3,wk1.transpose(),wk2,wk3.transpose(),wv1,wv2.transpose(),p2[p_no,:],p2.transpose()[:,p_no])
    return op,accuracies[-1]



def corr3(p_no):
    config, model, train_losses, test_losses, accuracies = load_experiment("my_exp")
    
    w = model.embedding.patch_embeddings.projection.weight
    wp=np.asarray(w.cpu().detach().numpy()).transpose()

    # pe = model.embedding.pos_embed.cpu().detach().numpy()
    # pe = np.squeeze(pe)
    # print(pe.shape)
    # Pe = np.stack((pe[0,:],pe.transpose()[:,0]))


        
    q1 = model.encoder.blocks[0].q_proj.weight
    k1 = model.encoder.blocks[0].k_proj.weight
    v1 = model.encoder.blocks[0].v_proj.weight
    q2 = model.encoder.blocks[1].q_proj.weight
    k2 = model.encoder.blocks[1].k_proj.weight
    v2 = model.encoder.blocks[1].v_proj.weight

    cl = model.classifier.weight.cpu().detach().numpy()


    wq1 = np.asarray(q1.cpu().detach().numpy()).transpose()
    wk1 = np.asarray(k1.cpu().detach().numpy()).transpose()           
    wv1 = np.asarray(v1.cpu().detach().numpy()).transpose()
    wq2 = np.asarray(q2.cpu().detach().numpy()).transpose()
    wk2 = np.asarray(k2.cpu().detach().numpy()).transpose()           
    wv2 = np.asarray(v2.cpu().detach().numpy()).transpose()
    # p = model.encoder.blocks[1].pos_embed(torch.zeros((8,16))).squeeze(0).detach().numpy()
    p2 = model.encoder.blocks[1].pos_embed(torch.randn((1,64,28))).squeeze(0).detach().numpy()

   
    op = contract('ea,kr,sl,mk,ac,lm,cb,bn,n->ers',wp,wp.transpose(),wp,wq1.transpose(),wq2,wk1,wk2.transpose(),wv1.transpose(),p2.transpose()[:,p_no])

    return op,accuracies[-1]



def corr2():
    config, model, train_losses, test_losses, accuracies= load_experiment("my_exp")

    
    w = model.embedding.patch_embeddings.projection.weight
    wp=np.asarray(w.cpu().detach().numpy()).transpose()
        
    q = model.encoder.blocks[0].q_proj.weight
    k = model.encoder.blocks[0].k_proj.weight
    v = model.encoder.blocks[0].v_proj.weight

    cl = model.classifier.weight.cpu().detach().numpy()


    wq = np.asarray(q.cpu().detach().numpy()).transpose()
    wk = np.asarray(k.cpu().detach().numpy()).transpose()           
    wv = np.asarray(v.cpu().detach().numpy()).transpose()
     
    op = contract('rk,ls,km,ml->rs',wp,wp.transpose(),wq,wk.transpose())
    return op,accuracies[-1]
