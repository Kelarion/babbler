CODE_DIR = r'/home/matteo/Documents/github/rememberforget/'
svdir = r'/home/matteo/Documents/uni/columbia/babbler/'

import sys, os, re
sys.path.append(CODE_DIR)

import torch
import torch.optim as optim

from sklearn import svm, calibration, linear_model, discriminant_analysis, manifold
import scipy.stats as sts
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.linalg as la
import scipy.stats as sts
import umap
from cycler import cycler

from students import RNNModel, LinearDecoder, Dichotomies
from needful_functions import *

#%%
def AnBn(nseq, nT, L, eps=0.5, cue=True, align=False, atfront=True):
    """
    Generate nseq sequences according to the A^n B^n grammar
    Sequences are padded with -1, with tokens occuring at random times
    eps sets the proportion of sequences which are ungrammatical
    
    the ungrammatical ('noise') sequences are of random length and A/B proportion
    """
    
    p_gram = (1-eps)
    p_nois = eps
    # here's one way to generate the sequences, 
    # going to create an empty array, fill it with the valid sequences first
    seqs = -1*np.ones((nseq, nT))
    
    n = int(p_gram*nseq/len(L))
    N = 0
    for l in L:
        
        valid_seqs = np.apply_along_axis(np.repeat, 1, np.repeat([[0,1]],n,0), [l, l])
        
        if align:
            idx = np.arange(0,nT-np.mod(nT,2*l),np.floor(nT/(2*l)))
            idx = np.ones(n,nT)*idx[None,:]
        else:
            idx = np.random.rand(n,nT).argsort(1)[:,:(2*l)]
            idx = np.sort(idx,1)
        np.put_along_axis(seqs[N:N+n,:], idx, valid_seqs, axis=1)
        N+=n
    
    # now I want to add noise sequences, i.e. random number of A and B tokens
    # but I want to make sure that the sparseness of the sequences isn't
    # too different from the grammatical ones -- so I set that manually
    
    thr = sts.norm.ppf(2*np.mean(L_train)/nT)
    noise_seqs = ((np.ones(nseq-N)[:,None]*np.arange(nT) - np.random.choice(nT-5,(nseq-N,1)))>0).astype(int)
    noise_seqs[np.random.randn(nseq-N,nT)>thr] = -1
    
    seqs[N:,:] = noise_seqs
    labels = (seqs == 0).sum(1) == (seqs==1).sum(1)
    
    if cue:
        seqs = np.append(seqs, np.ones(nseq)[:,None]*2, axis=1)
    if atfront:
        # push to the front
        seqs = np.where(seqs==-1, np.nan, seqs)
        seqs = np.sort(seqs,1)
        seqs = np.where(np.isnan(seqs),-1,seqs)
    
    shf = np.random.choice(nseq,nseq,replace=False)
    seqs = seqs[shf,:]
    labels = labels[shf]
    
    return seqs, labels

# def disperse_in_array(vals, ):
    

#%% generate sequences
nseq = 3000
nT = 24
dense = True # is the sequence dense? (no time without input)
L_train = [3,5,7,12]
L_test = [2,4,6,8]
cued = False

# train set -- manually append the query token
seqs, labels = AnBn(nseq, nT, L_train, atfront=dense, cue=cued)

# test set
tseqs, tlabels = AnBn(nseq, nT, L_train, atfront=dense, cue=cued)

#%%
# TODO: see if the performance depends on the amount of dead time
# this would tell us something about the role of dynamics?

N = 25
rnn_type = 'tanh'

nepoch = 2000               # max how many times to go ￼￼over data
alg = optim.Adam
dlargs = {'num_workers': 2, 
          'batch_size': 64, 
          'shuffle': True}  # dataloader arguments
optargs = {'lr': 1e-3}
criterion = torch.nn.CrossEntropyLoss()

ninp = 2+1*cued

train_seqs = torch.tensor(seqs, requires_grad=False).type(torch.LongTensor)
train_labels = torch.tensor(labels, requires_grad=False).type(torch.LongTensor)

test_seqs = torch.tensor(tseqs, requires_grad=False).type(torch.LongTensor)
test_labels = torch.tensor(tlabels, requires_grad=False).type(torch.LongTensor)


rnn = RNNModel(rnn_type, ntoken=ninp, ninp=ninp, nhid=N, nlayers=1)

rnn.train(train_seqs, train_labels, optargs, dlargs, criterion=criterion,
          test_data=(test_seqs,test_labels), nepoch=nepoch)

#%%
ntest = 1000
L_testing = list(range(1,max([max(L_train),max(L_test)])+1))

hid = rnn.init_hidden(ntest)

perf = np.zeros(len(L_testing))
for i,l in enumerate(L_testing):
    new_seq, new_labs = AnBn(ntest, nT, [l], cue=cued)
    
    test_seqs = torch.tensor(new_seq, requires_grad=False).type(torch.LongTensor)
    test_labels = torch.tensor(new_labs, requires_grad=False).type(torch.LongTensor)

    outp, _ = rnn(test_seqs.transpose(1,0), hid)
    test_tfinal = -(np.fliplr(new_seq>-1).argmax(1)+1)
    outp = outp[test_tfinal, np.arange(new_seq.shape[0]), :]
    
    perf[i] = torch.sum(outp.argmax(1)==test_labels).detach().numpy()/ntest
    
plt.plot(L_testing,perf)
plt.ylim([0,1.1])
plt.plot([min(L_testing),max(L_testing)],[0.5,0.5],'k--')
plt.plot([L_train,L_train],plt.ylim(),'-.', c=(0.5,0.5,0.5))

plt.legend(['accuracy','chance','trained n'])

plt.ylabel('test accuracy')
plt.xlabel('n')

#%% Compute the PCs on a lot of sequences

tseqs, tlabels = AnBn(nseq, nT, L_testing, cue=cued)
# tseqs = np.append(tseqs, np.ones(nseq)[:,None]*2, axis=1

test_seqs = torch.tensor(tseqs, requires_grad=False).type(torch.LongTensor)
test_labels = torch.tensor(tlabels, requires_grad=False).type(torch.LongTensor)

hid = rnn.init_hidden(nseq)
out, H = rnn.transparent_forward(test_seqs.transpose(1,0),hid)

X = H.detach().numpy().transpose((1,2,0)).reshape((25,-1))
perf = (out[-1,:,:].argmax(1) == test_labels).detach().numpy().sum()/nseq

_, S, V = la.svd(X.T-X.mean(1), full_matrices=False)
pcs = X.T@V[:3,:].T

#%% plot PCs
nplot = 5000
cmap_name = 'nipy_spectral'
# cmap_name = 'hot'

colorby = np.repeat(tlabels,nT+1)
# alphaby = np.tile(np.arange(nT+1),nseq)

cols = getattr(cm, cmap_name)(colorby.astype(float))
# cols[:,3] = 1/(1+np.exp(-3*alphaby/nT))

plot_these = np.random.choice(X.shape[1],nplot, replace=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter(pcs[plot_these,0],pcs[plot_these,1],pcs[plot_these,2], 
                  c=colorby[plot_these], alpha=1, cmap=cmap_name)
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')

cb = plt.colorbar(scat, 
                  ticks=np.unique(colorby),
                  drawedges=True,
                  values=np.unique(colorby))
cb.set_ticklabels(np.unique(colorby))
cb.set_alpha(1)
cb.ax.set_yticklabels(['Ungrammatical','Grammatical']) 
cb.draw_all()


#%%
viz_L = 5
viz_n = 10
# nT = 24

new_seq, new_labs = AnBn(viz_n, nT, [viz_L])
# new_seq = np.append(new_seq, np.ones(viz_n)[:,None]*2, axis=1)

# idx = np.arange(0,nT+1-np.mod(nT+1,2*viz_L),np.floor((nT+1)/(2*viz_L)),dtype=int)
# idx = np.ones((viz_n,2*viz_L), dtype=int)*idx[None,:]
# # idx = np.ones((viz_n,2*viz_L), dtype=int)*np.random.rand(1,nT).argsort(1)[:,:(2*viz_L)]
# # idx = np.sort(idx,1)

# viz_seqs = -1*np.ones((viz_n, nT))
# np.put_along_axis(viz_seqs, idx, new_seq, axis=1)
# new_seq = np.append(viz_seqs, np.ones(viz_n)[:,None]*2, axis=1)

test_seqs = torch.tensor(new_seq, requires_grad=False).type(torch.LongTensor)
test_labels = torch.tensor(new_labs, requires_grad=False).type(torch.LongTensor)

hid = rnn.init_hidden(viz_n)
out, H = rnn.transparent_forward(test_seqs.transpose(1,0),hid)

X = H.detach().numpy().transpose((1,2,0)).reshape((25,-1))
perf = (out[-1,:,:].argmax(1) == test_labels).detach().numpy().sum()/viz_n


pcs = X.T@V[:3,:].T
D = pcs.reshape((viz_n,nT+1,3))

#%%
plot_win = 6

# how to show the inputs at each time
inp_id = smear_tokens(new_seq.T).T
signals = np.array([[1.,1.,1.,1.], # wait
                    [1.,0.,0.,1.], # token A
                    [0.,0.,1.,1.], # token B
                    [0.,1.,0.,1.]]) # cue

# colorby = np.repeat(new_labs[:,None],nT+1,axis=1)
colorby = new_labs.astype(float)
cols = getattr(cm, cmap_name)(np.unique(colorby))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xdata, ydata, zdata = D[:,0,0], D[:,0,1], D[:,0,2]
ln = [[] for _ in range(viz_n+1)]

ln[0] = ax.scatter(xdata, ydata, zdata, c=colorby, marker='o', s=100, cmap=cmap_name,
                   linewidth=2.0, edgecolors='k')
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')

for s in range(viz_n):
    ln[s+1], = ax.plot(D[s,:1,0],D[s,:1,1],D[s,:1,2], c=cols[colorby[s].astype(int),:], alpha=0.5)

cb = plt.colorbar(scat, 
                  ticks=np.unique(colorby),
                  drawedges=True,
                  values=np.unique(colorby))
cb.set_ticklabels(np.unique(colorby))
cb.set_alpha(1)
cb.ax.set_yticklabels(['Ungrammatical','Grammatical']) 
cb.draw_all()

# ax.set_title('Token: 0')

def init():
    ax.set_xlim(D[...,0].min()-1, D[...,0].max()+1)
    ax.set_ylim(D[...,1].min()-1, D[...,1].max()+1)
    ax.set_zlim(D[...,2].min()-1, D[...,2].max()+1)
    return ln

def update(i):
    xdata = D[:,i,0]
    ydata = D[:,i,1]
    zdata = D[:,i,2]
    # ln[0]._offsets3d = (xdata, ydata, zdata)
    
    ec = signals[inp_id[:,i].astype(int)+1,:]
    ln[0].set_edgecolor(ec)
    ln[0].set_facecolor(getattr(cm, cmap_name)(colorby))
    ln[0].set_3d_properties(zdata,'z')
    ln[0]._offsets3d = (xdata, ydata, zdata)
    
    t0 = np.max([0, i-plot_win])
    for s in range(viz_n):
        ln[s+1].set_data((D[s,t0:i+1,0],D[s,t0:i+1,1]))
        ln[s+1].set_3d_properties(D[s,t0:i+1,2])
        
    # if new_seq[s,i]>-1:
        # ax.set_title('Token: %d'%int(2*i/viz_L))
        
    return ln

anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=nT+1, interval=100, blit=False)

# anim.save(svdir+'AnBn_parser.avi', fps=20, extra_args=['-vcodec', 'libx264'])

plt.show()


