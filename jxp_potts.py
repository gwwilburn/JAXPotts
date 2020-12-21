import jax.numpy as np
from jax import vmap, jit
import jax.nn as nn
from jax.ops import index, index_add, index_update
import numpy as onp

from jxp_alphabet import *
from jxp_msa import *

class PottsModel(object):

   def __init__(self, L = 2, name = "", h = None,
                e = None, abc = ABC_RNA):

      self.L    = L       # model length (number of sites)
      self.abc  = abc     # biological alphabet
      self.q    = abc.q   # alphabet size
      self.h    = h       # local field terms (L x q)
      self.e    = e       # coupling terms (L x q x L x q)
      self.name = name    # model name (highly optional)

      # initalize h array
      if h is None:
         self.h = np.zeros((L,q), dtype=np.float32)

      # initialze e array
      if e is None:
         self.e = np.zeros((L,q,L,q), dytpe=np.float32)

def Potts_ScoreMSA(potts, msa):
   if msa.ax_1hot is None:
      msa.ax_1hot = MSA_Onehot(msa.ax,msa.abc.q)

   return Potts_ScoreMSACore(potts.h, potts.e, msa.ax_1hot)

# Potts Model scoring function (Hamiltonian calculation) for 1 aligned sequence
@jit
def Potts_ScoreSeqCore(h, e, seq_1hot):

   seq_useme = SeqUseMe(seq_1hot)

   return np.tensordot(h, seq_1hot, axes=([0,1],[0,1])) + (np.tensordot(e,seq_useme,axes=([0,1,2,3],[0,1,2,3])) / 2.0)

# Potts model scoring function (Hamiltonian calculation) for an MSA
Potts_ScoreMSACore = vmap(Potts_ScoreSeqCore, in_axes = (None, None, 0), out_axes = 0)

def Potts_ShiftGaugeZeroSum(potts):

   # make L x q array of mean value of h_i for all L sites
   h_mean = np.tensordot(np.mean(potts.h,axis = 1), np.ones(potts.q), axes=0)

   # update h to zero-sum gauge
   potts.h = index_update(potts.h, index[:,:], potts.h-h_mean)

   # make L x q x L x q of mean value of e_ij for all L choose 2 site pairs
   # should be symmetric and 0 on diagonal
   e_mean = np.tensordot(np.mean(potts.e,axis=(1,3)), np.ones((potts.q,potts.q)), axes=0)

   # transpose e so that it is L x L x q x q
   e_transpose = np.transpose(potts.e, axes=[0,2,1,3])

   # shift to zero-sum gauge
   e_transpose = index_update(e_transpose, index[:,:,:,:], e_transpose - e_mean)

   # undo the transpotition to get an L x q x L x q array
   potts.e = index_update(potts.e, index[:,:,:,:], np.transpose(e_transpose, axes=(0,2,1,3)))

   return

def Potts_Symmetrize(potts):

   # remove e_ii's
   potts.e = index_update(potts.e, index[:,:,:,:], potts.e*np.reshape(1-np.eye(potts.L), (potts.L,1,potts.L,1)))

   # set e_ij(a,b) = (e_ij(a,b) + e_ji(b,a)) / 2
   potts.e = index_update(potts.e, index[:,:,:,:], (potts.e + np.transpose(potts.e, [2,3,0,1])) / 2. )


   return


def Potts_Write(potts, outpath):

   f = open(outpath, 'w')
   h = potts.h
   e = potts.e
   L = potts.L
   q = potts.q

   # part 1: write h terms
   for i in range(0,L):
      print(i)
      h_str = "%d " % (i)
      for a in range (0,q):
         h_str += "%.4f " % h[i][a]

      h_str += '\n'
      f.write(h_str)

   # part 2: write e terms
   for i in range(0, L):
      for j in range(i + 1, L):
         print(i,j)
         f.write("%d %d\n" % (i,j) )

         for a in range(0,q):
            e_str = ""
            for b in range(0, q):
               e_str += "%.4f " % e[i][a][j][b]

            e_str += '\n'
            f.write(e_str)

         f.write('\n')


   f.close()
   return

def Potts_Read(inpath, abc = ABC_RNA):
   f = open(inpath, 'r')

   first_line = True
   L = 0

   # loop through once to determine model dimensions
   for line in f:
      line = line.rstrip()
      linelist = line.split(' ')
      if "#" in linelist[0]:
         continue

      if first_line:
         q = len(linelist) -1
         first_line = False

      if len(linelist) == q+1:
         L += 1
   f.close()

   h = np.zeros((L,q), dtype = np.float32)
   e = np.zeros((L,q,L,q), dtype = np.float32)

   f = open(inpath, 'r')

   for line in f:
      line = line.rstrip()
      linelist = line.split(' ')
      if "#" in linelist[0]:
         continue

      # get h_i's
      if len(linelist) == q+1:
         i = int(linelist[0])
         h = index_update(h, index[i,:], np.array(linelist[1:],dtype=np.float32))


      # get e_{ij}'s
      elif linelist[0].isdigit() and linelist[1].isdigit():
         i = int(linelist[0])
         j = int(linelist[1])
         print(i,j)
         a = 0

      elif len(linelist) == q:
         e = index_update(e, index[i,a,j,:], np.array(linelist, dtype=np.float32))
         e = index_update(e, index[j,:,i,a], np.array(linelist, dtype=np.float32))
         a += 1

   f.close()

   return PottsModel(abc=abc, L=L, e=e, h=h)

