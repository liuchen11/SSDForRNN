import sys
sys.path.insert(0,'util/')
sys.path.insert(0,'models/')
import numpy as np
import copy

import math
import gradient
import softmax
import sharp
import batchProduct
import RNNs

'''
>>> Calculate the gradient of each parameters matrix
>>> model: multi-layer RNN
>>> states: Input states. 2-D array of shape S*N where S is the sequence lendth and N the input dimension
>>> ground_truth: Labels. 2-D array of shape S*H or 1*H where H is the number of labels
>>> optimizer: optimizer. optimize strategy
>>> final_label: Boolean, optional. If True the sequence has only one label otherwise each token is assigned one label
'''
def sgd(model,states,ground_truth,optimizer,final_label=False):
    hidden_layers=model.hidden_layers
    neurons=model.size

    # Global states
    hidden_states=copy.copy(model.s)            #hidden states, should be deep copy
    lamb=copy.copy(model.s)                     #lambda, size shape of hidden states
    err=0.0

    # gradient used by iteration
    dSdU=[[] for i in xrange(hidden_layers)]    #dSdU[n1,n2]=dS[n1]/dU[n2], n1>=n2
    dSdW=[[] for i in xrange(hidden_layers)]    #dSdW[n1,n2]=dS[n1]/dW[n2], n1>=n2
    dSds=[[] for i in xrange(hidden_layers)]    #dSds[n1,n2]=dS[n1]/ds[n2], n1>=n2

    for n1 in xrange(hidden_layers):
        for n2 in xrange(n1+1):
            dSdU_n1n2=np.zeros([neurons[n2+1],neurons[n2],neurons[n1+1]])       #dSdU[n1,n2][i,j,k]=dS[n1][k]/dU[n2][i,j]
            dSdW_n1n2=np.zeros([neurons[n2+1],neurons[n2+1],neurons[n1+1]])      #dSdW[n1,n2][i,j,k]=dS[n1][k]/dW[n2][i,j]
            dSds_n1n2=np.zeros([neurons[n1+1],neurons[n2+1]]) if n1!=n2 else np.eye(neurons[n1+1])       #dSds[n1,n2][i,j]=dS[n1][i]/ds[n2][j]
            dSdU[n1].append(dSdU_n1n2)
            dSdW[n1].append(dSdW_n1n2)
            dSds[n1].append(dSds_n1n2)

    weight=1.0 if final_label else 1.0/len(states)
    for idx,token in enumerate(states):
        # Save the value of old hidden states, useful to update the gradient
        hidden_states_old=copy.copy(hidden_states)

        # Forward Propagation
        linear_comb=np.dot(model.U[0],token)+np.dot(model.W[0],hidden_states[0])
        hidden_states[0]=model.activation[0](linear_comb)
        lamb[0]=model.dactivation[0](linear_comb)
        for i in xrange(1,hidden_layers):
            linear_comb=np.dot(model.U[i],hidden_states[i-1])+np.dot(model.W[i],hidden_states[i])
            hidden_states[i]=model.activation[i](linear_comb)
            lamb[i]=model.dactivation[i](linear_comb)
        
        # R[n][i,j]=dS[n][i]_t/dS[n-1][j]_t
        # S[n][i,j]=dS[n][i]_t/dS[n][j]_{t-1}
        R=[];S=[]
        for i in xrange(hidden_layers):
            Ri=np.dot(np.diag(lamb[i]),model.U[i])
            Si=np.dot(np.diag(lamb[i]),model.W[i])
            R.append(Ri)
            S.append(Si)

        for n1 in xrange(hidden_layers):
            for n2 in xrange(n1):
                dSdU[n1][n2]=batchProduct.nXone(dSdU[n1-1][n2],R[n1].T)+batchProduct.nXone(dSdU[n1][n2],S[n1].T)
                dSdW[n1][n2]=batchProduct.nXone(dSdW[n1-1][n2],R[n1].T)+batchProduct.nXone(dSdW[n1][n2],S[n1].T)
                dSds[n1][n2]=np.dot(dSds[n1-1][n2],R[n1].T)+np.dot(dSds[n1][n2],S[n1].T)

            dSdU[n1][n1]=batchProduct.nXone(dSdU[n1][n1],model.W[n1])
            dSdW[n1][n1]=batchProduct.nXone(dSdW[n1][n1],model.W[n1])
            for i in xrange(neurons[n1+1]):
                for j in xrange(neurons[n1+1]):
                    dSdW[n1][n1][i,j,i]+=hidden_states_old[n1][j]
                for j in xrange(neurons[n1]):
                    if n1>0:
                        dSdU[n1][n1][i,j,i]+=hidden_states[n1-1][j]
                    else:
                        dSdU[n1][n1][i,j,i]+=token[j]
            dSdU[n1][n1]=batchProduct.nXone(dSdU[n1][n1],np.diag(lamb[n1]))
            dSdW[n1][n1]=batchProduct.nXone(dSdW[n1][n1],np.diag(lamb[n1]))
            dSds[n1][n1]=np.dot(np.diag(lamb[n1]),np.dot(model.W[n1],dSds[n1][n1]))

        # Have supervised signal -> update the gradient
        if idx==len(states)-1 or not final_label:
            proj=np.dot(model.V,hidden_states[-1])
            soft=softmax.softmax(proj)
            logsoft=np.log(soft)

            token_truth=ground_truth[0] if final_label else ground_truth[idx]
            err-=np.dot(token_truth,logsoft)*weight

            # Update V
            dEdV=np.dot((soft-token_truth).reshape(neurons[-1],1),hidden_states[-1].reshape(1,neurons[-2]))
            model.gV+=dEdV*weight
            model.dV+=np.div(dEdV,optimizer.DV)*weight if optimizer.name!='const' else dEdV*weight

            # Update U,W,s
            dEdS=np.dot(model.V.T,(soft-token_truth).reshape(neurons[-1],1))
            for n in xrange(hidden_layers):
                dEdUi=batchProduct.nXone(dSdU[-1][n],dEdS).squeeze()
                dEdWi=batchProduct.nXone(dSdW[-1][n],dEdS).squeeze()
                dEdsi=np.dot(dSds[-1][n].T,dEdS).squeeze()

                model.gU[n]+=dEdUi*weight
                model.dU[n]+=np.div(dEdUi,optimizer.DU[n])*weight if optimizer.name!='const' else dEdUi*weight
                model.gW[n]+=dEdWi*weight
                model.dW[n]+=np.div(dEdWi,optimizer.DW[n])*weight if optimizer.name!='const' else dEdWi*weight
                model.gs[n]+=dEdsi*weight
                model.ds[n]+=np.div(dEdsi,optimizer.Ds[n])*weight if optimizer.name!='const' else dEdsi*weight

    model.buffer+=1
    return err


'''
>>> Calculate the spectral gradient of each parameters matrix
>>> model: multi-layer RNN
>>> states: Input states. 2-D array of shape S*N where S is the sequence lendth and N the input dimension
>>> ground_truth: Labels. 2-D array of shape S*H or 1*H where H is the number of labels
>>> optimizer: optimizer. optimize strategy
>>> final_label: Boolean, optimal. If True the sequence has only one label otherwise each token is assigned one label
'''
def ssd(model,states,ground_truth,optimizer,final_label=False):
    hidden_layers=model.hidden_layers
    neurons=model.size

    # Global states
    hidden_states=copy.copy(model.s)            #hidden states, should be deep copy
    lamb=copy.copy(model.s)                     #lambda, size shape of hidden states
    err=0.0
    exception_num=0

    # gradient used by iteration
    dSdU=[[] for i in xrange(hidden_layers)]    #dSdU[n1,n2]=dS[n1]/dU[n2], n1>=n2
    dSdW=[[] for i in xrange(hidden_layers)]    #dSdW[n1,n2]=dS[n1]/dW[n2], n1>=n2
    dSds=[[] for i in xrange(hidden_layers)]    #dSds[n1,n2]=dS[n1]/ds[n2], n1>=n2

    for n1 in xrange(hidden_layers):
        for n2 in xrange(n1+1):
            dSdU_n1n2=np.zeros([neurons[n2+1],neurons[n2],neurons[n1+1]])     #dSdU[n1,n2][i,j,k]=dS[n1][k]/dU[n2][i,j]
            dSdW_n1n2=np.zeros([neurons[n2+1],neurons[n2+1],neurons[n1+1]])    #dSdW[n1,n2][i,j,k]=dS[n1][k]/dW[n2][i,j]
            dSds_n1n2=np.zeros([neurons[n1+1],neurons[n2+1]]) if n1!=n2 else np.eye(neurons[n1+1])       #dSds[n1,n2][i,j]=dS[n1][i]/ds[n2][j]
            dSdU[n1].append(dSdU_n1n2)
            dSdW[n1].append(dSdW_n1n2)
            dSds[n1].append(dSds_n1n2)

    weight=1.0 if final_label else 1.0/len(states)
    for idx,token in enumerate(states):
        # Save the value of old hidden states, useful to update the gradient
        hidden_states_old=copy.copy(hidden_states)

        # Forward Propagation
        linear_comb=np.dot(model.U[0],token)+np.dot(model.W[0],hidden_states[0])
        hidden_states[0]=model.activation[0](linear_comb)
        lamb[0]=model.dactivation[0](linear_comb)
        for i in xrange(1,hidden_layers):
            linear_comb=np.dot(model.U[i],hidden_states[i-1])+np.dot(model.W[i],hidden_states[i])
            hidden_states[i]=model.activation[i](linear_comb)
            lamb[i]=model.dactivation(linear_comb)
        
        # R[n][i,j]=dS[n][i]_t/dS[n-1][j]_t
        # S[n][i,j]=dS[n][i]_t/dS[n][j]_{t-1}
        R=[];S=[]
        for i in xrange(hidden_layers):
            Ri=np.dot(np.diag(lamb[i]),model.U[i])
            Si=np.dot(np.diag(lamb[i]),model.W[i])
            R.append(Ri)
            S.append(Si)

        for n1 in xrange(hidden_layers):
            for n2 in xrange(n1):
                dSdU[n1][n2]=batchProduct.nXone(dSdU[n1-1][n2],R[n1].T)+batchProduct.nXone(dSdU[n1][n2],S[n1].T)
                dSdW[n1][n2]=batchProduct.nXone(dSdW[n1-1][n2],R[n1].T)+batchProduct.nXone(dSdW[n1][n2],S[n1].T)
                dSds[n1][n2]=np.dot(dSds[n1-1][n2],R[n1].T)+np.dot(dSds[n1][n2],S[n1].T)

            dSdU[n1][n1]=batchProduct.nXone(dSdU[n1][n1],model.W[n1])
            dSdW[n1][n1]=batchProduct.nXone(dSdW[n1][n1],model.W[n1])
            for i in xrange(neurons[n1+1]):
                for j in xrange(neurons[n1+1]):
                    dSdW[n1][n1][i,j,i]+=hidden_states_old[n1][j]
                for j in xrange(neurons[n1]):
                    if n1>0:
                        dSdU[n1][n1][i,j,i]+=hidden_states[n1-1][j]
                    else:
                        dSdU[n1][n1][i,j,i]+=token[j]
            dSdU[n1][n1]=batchProduct.nXone(dSdU[n1][n1],np.diag(lamb[n1]))
            dSdW[n1][n1]=batchProduct.nXone(dSdW[n1][n1],np.diag(lamb[n1]))
            dSds[n1][n1]=np.dot(np.diag(lamb[n1]),np.dot(model.W[n1],dSds[n1][n1]))

        # Have supervised signal -> update the gradient
        if idx==len(states)-1 or not final_label:
            proj=np.dot(model.V,hidden_states[-1])
            soft=softmax.softmax(proj)
            logsoft=np.log(soft)

            token_truth=ground_truth[0] if final_label else ground_truth[idx]
            err-=np.dot(token_truth,logsoft)*weight

            # Update V
            dEdV=np.dot((soft-token_truth).reshape(neurons[-1],1),hidden_states[-1].reshape(1,neurons[-2]))
            model.gV+=dEdV*weight
            model.dV+=np.div(dEdV,optimizer.DV)*weight if optimizer.name!='const' else  dEdV*weight

            # Update U,W,s
            dEdS=np.dot(model.V.T,(soft-token_truth).reshape(neurons[-1],1))
            for n in xrange(hidden_layers):
                dEdUi=batchProduct.nXone(dSdU[-1][n],dEdS).squeeze()
                dEdWi=batchProduct.nXone(dSdW[-1][n],dEdS).squeeze()
                dEdsi=np.dot(dSds[-1][n].T,dEdS).squeeze()

                model.gU[n]+=dEdUi*weight
                model.gW[n]+=dEdWi*weight
                model.gs[n]+=dEdsi*weight
                if optimizer.name!='const':
                    sqrtDUi=np.sqrt(optimizer.DU[n])
                    try:
                        model.dU[n]+=np.div(sharp.sharp(np.div(dEdUi,sqrtDUi)),sqrtDUi)*weight
                    except:
                        exception_num+=1
                        model.dU[n]+=np.div(dEdUi,optimizer.DU[n])*weight
                    sqrtDWi=np.sqrt(optimizer.DW[n])
                    try:
                        model.dW[n]+=np.div(sharp.sharp(np.div(dEdWi,sqrtDWi)),sqrtDWi)*weight
                    except:
                        exception_num+=1
                        model.dW[n]+=np.div(dEdWi,optimizer.DW[n])*weight
                    model.ds[n]+=np.div(dEdsi,optimizer.Ds[n])*weight
                else:
                    model.dU[n]+=sharp.sharp(dEdUi)*weight
                    model.dW[n]+=sharp.sharp(dEdWi)*weight
                    model.ds[n]+=dEdsi*weight

    model.buffer+=1
    return exception_num, err