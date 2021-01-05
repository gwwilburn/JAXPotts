import jax.numpy as np

from jxp_alphabet import *
from jxp_msa import *


def Get_Henikoff_Counts_Residue(i, a, c):
   return c[i,a]

Get_Henikoff_Counts_MSA = vmap(Get_Henikoff_Counts_Residue, in_axes = ((True,True),(True,True), False), out_axes = ((True,True)))

def MSAWeight_PB(msa):
   gap_idx = msa.abc.charmap['-']
   q = msa.abc.q
   ax = msa.ax
   (N,L) = ax.shape

   ## step 1: get counts:

   c = np.sum(msa.ax_1hot, axis=0)

   # set gap counts to 0
   c = index_update(c, index[:,gap_idx], 0)

   # get N x L array with count value for corresponding residue in alignment
   # first, get  N x L "column id" array (convenient for vmap)
   # col_id[n,i] = i
   col_id = np.int16(np.tensordot(np.ones(N), np.arange(L), axes=0))
   # ax_c[n, i] = c[i, ax[n,i]]
   ax_c = Get_Henikoff_Counts_Residue(col_id , ax, c)

   ## step 2: get number of unique characters in each column
   r = np.float32(np.sum( np.array(c > 0), axis = 1))

   # transform r from Lx1 array to NxL array, where r2[n,i] = r[i])
   # will allow for easy elementwise operations with ax_c
   r2 = np.tensordot(np.ones(N), r, axes=0)

   ## step 3: get ungapped seq lengths
   nongap = np.array(ax != gap_idx)
   l = np.float32(np.sum( nongap, axis = 1))

   ## step 4: calculate unnormalized weights
   # get array of main terms in Henikoff sum
   #wgt_un[n,i] = 1 / (r_[i] * c[i, ax[n,i] ])
   wgt_un = np.reciprocal(np.multiply(ax_c, r2))

   # set all terms involving  gap to zero
   wgt_un = np.nan_to_num(np.multiply(wgt_un, nongap))

   # sum accoss all positions to get prelim unnormalized weight for each sequence
   wgt_un = np.sum(wgt_un, axis=1)

   # divide by gapless sequence length
   wgt_un = np.divide(wgt_un, l)

   # step 4: Normalize sequence wieghts
   wgt = (wgt_un * np.float32(N)) / np.sum(wgt_un)
   msa.wgt = wgt

   return

def MSAWeight_PBLoops(msa):

   gap_idx = msa.abc.charmap['-']
   q = msa.abc.q
   ax = msa.ax
   (N,L) = ax.shape

   c = np.zeros((L,q),dtype=np.float32)

   # step 1: get frequency counts
   for n  in range(0, N):
      for i in range(0,L):
         if ax[n,i] != gap_idx:
            c = index_update(c, index[i, ax[n,i]], c[i, ax[n,i]] +1)

   # step 2: find number of unique counts in each columnn
   r = np.zeros(L, dtype=np.float32)
   for i in range(0,L):
      for a in range(0, q-1):
         if c[i,a] > 0:
            r = index_update(r, index[i], r[i]+1)

   # Step 3: get unnormalized weights
   l = np.zeros(N, dtype=np.float32)
   wgt_un = np.zeros(N, dtype=np.float32)
   for n in range(0,N):
      for i in range(0, L):
         if ax[n,i] != gap_idx:
            l = index_update(l, index[n], l[n]+1)
            wgt_un = index_update(wgt_un, index[n],  wgt_un[n] + 1. / (r[i] * c[i,ax[n,i]]))
      # divide by gapless sequence length
      if l[n] > 0:
         wgt_un = index_update(wgt_un, index[n], wgt_un[n] / l[n])

   # step 4: normalize weights
   msa.wgt = index_update(wgt_un, index[:], wgt_un * N / np.sum(wgt_un))

   return
