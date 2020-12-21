import jax.numpy as np
from jax import vmap, jit
from jax.ops import index, index_add, index_update
import numpy as onp
import jax.nn as nn

from jxp_alphabet import *
from jxp_msa import *
from functools import partial

import sys

class MSA(object):

   def __init__(self, abc = ABC_RNA, L=2, N=2, ax = None, ax_1hot = None, ax_useme = None,
               name = "", sqname = None, wgt = None, ss_cons = None, rf = None,
               bp_map = None):

      self.abc   = abc         # alphabet
      self.L     = L           # length (columns)
      self.N     = N           # number of seqs (rows)
      self.q     = abc.q       # alphabet size
      self.ax    = ax          # 2D array of ints representing MSA (N x L)
      self.ax_1hot = ax_1hot   # 1-hot encondign of MSA (N x L x q)
      self.ax_useme = ax_useme #
      self.sqname = sqname

      # per-column annotation from Stockholm files
      self.ss_cons = ss_cons   # secondary structure annotation
      self.rf = rf             # RF annotation (consensus cols)
      self.bp_map = bp_map     # base-pairing map


      if ax is not None:
         (self.N, self.L) = ax.shape

      if sqname is None:
         self.sqname = []

      if wgt is None:
         self.wgt = np.ones(N, dtype=np.float32)

      # if we have standard ax but not 1-hot version, create it
      if ax is not None and ax_1hot is None:
         self.ax_1hot = MSA_Onehot(self.ax,self.abc.q)

   # TODO: add function that returns h3-style traces
   def GetTraces(self):
      pass

def MSA_Read(path, abc, fileformat = "afa", name = ""):
   ret_msa = None
   if fileformat.lower() == "stockholm":
      ret_msa = MSA_ReadStockholm(path, abc, name)
   if fileformat.lower() in ["fasta", "afa"]:
      ret_msa = MSA_ReadAFA(path, abc, name)
   else:
      print("Invalid file format! Returning None")

   return ret_msa

# TODO: right stockholm-reading function
def MSA_ReadStockholm( path, abc, name):
   return None

def MSA_ReadAFA(path,abc, name):
   sqname = []   # sequence names
   msa_array = [] # nested list, will convert to np array

   af = open(path, 'r')

   for line in af:
      line = line.rstrip()

      # skip blank lines
      if len(line) == 0:
         continue

      if '>' in line:
         linelist = line.split(' ')
         sqname.append(linelist[0][1:])
         msa_array.append([])

      else:
         msa_array[-1].append(line)

   af.close()

   msa_array = [list(''.join(seq)) for seq in msa_array]
   L = len(msa_array[0])
   N = len(msa_array)

   # convert characters to integers
   for n in range(0, N):
      msa_array[n] =  Seq2Idx(msa_array[n], abc)


   ax = np.array(msa_array,dtype=np.int8)
   ret_msa = MSA(abc = abc, L = L, N = N, sqname = sqname, ax = ax)

   return ret_msa

def MSA_Write(msa, outpath, fileformat = "afa"):

   if msa.sqname is None or (len(msa.sqname) != msa.N):
      sys.exit("\nError: number of seq names in msa.sqname does not match number of seqs in msa\n\n")

   if fileformat.lower() == "afa":
      MSA_WriteAFA(msa, outpath)
   elif fileformat.lower() == "pfam":
      MSA_WritePfam(msa, outpath)


def MSA_WriteAFA(msa, outpath):
   ax = msa.ax
   sqname = msa.sqname
   (N,L) = ax.shape
   abc = msa.abc

   NLine = int(np.ceil(float(L) / 60.0))
   print(NLine)

   outfile = open(outpath, 'w')
   for n in range(0,N):
      outfile.write(">%s\n" % sqname[n])

      for m in range(0, NLine-1):
         seq_str = ''.join([abc.idxmap[i]  for i in ax[n,m*60:m+1*60]])
         outfile.write(seq_str + "\n")

      seq_str = ''.join([abc.idxmap[i]  for i in ax[n,(NLine-1)*60:]])
      outfile.write(seq_str + "\n")


   outfile.close()
   return

def MSA_WritePfam(msa, outpath):
   pass

# Function: MSA_Frequencies
# Comments: Slimmed-down frequency function
#           Does not include option for weighted frequencies
#           To be called MANY times during BMLearn
# args:    -MSA object
# returns: -f1: L x q array of single site frequencies
#          -f2: L x q x L x q symmetric array of pairwise frequencies
#           (with no terms on i=j diagonal)
#
@jit
def MSA_Frequencies(ax_1hot):

   ax_useme = MSA_UseMe(ax_1hot)

   Nf = np.float32(ax_1hot.shape[0])

   f1 = np.sum(ax_1hot, axis=0)  / Nf
   f2 = np.sum(ax_useme, axis=0) / Nf

   return f1, f2

def MSA_WeightedFrequencies(msa, weights = None):

   # if no weights given, use uniform weights
   if weights is None:
      weights = 37.5* np.ones(msa.N, dtype=np.int32)

   # sum weights (denominator for frequency calculation)
   Nw = np.sum(weights)

   # make sure 1-hot tensor exists
   if msa.ax_1hot is None:
      msa.ax_1hot = MSA_Onehot(msa.ax,msa.abc.q)

   wgt_1hot = MSA_WeightedOneHot(msa.ax_1hot, weights)
   wgt_useme = MSA_WeightedUseMe(msa.ax_1hot, weights)

   f1 = np.sum(wgt_1hot, axis=0)  / Nw
   f2 = np.sum(wgt_useme, axis=0) / Nw

   return f1, f2

# Function: MSA_2PtCorrelations
# Comments: Given single-site and pairwise MSA frequences,
#           returns 2-point pairwise correlation function
#           defined as
#           c_{ij}(a,b) = f_ij(a,b) - f_i(a) f_j(b)
#
#           Note that this is NOT mutual information
#
# Args:    f1: L x q array of single-site frequencies
#          f2: L x q x L x q symmetric array of pairwise frequencies
#
# Returns: c2: L x q x L x q symmetric array of 2-point correlations

@jit
def MSA_2PtCorrelations(f1, f2):

   L,q = f1.shape

   # multiplty single-site frequencies, but not on the i=j diagonal
   #fifj = np.tensordot(f1, f1, axes=0)  * np.reshape(1-np.eye(L), (L,1,L,1))
   #c2 = f2 - fifj

   return f2 -  (np.tensordot(f1, f1, axes=0)  * np.reshape(1-np.eye(L), (L,1,L,1)))

# function for converting and aligned sequence to a list of indices
def Seq2Idx(seq, abc):
   return [abc.charmap[char] for char in seq]

# function to convert MSA into one-hot array
# input array: n x L
# output array: n x L x q

MSA_Onehot  = vmap(nn.one_hot,in_axes = (0,None), out_axes = 0 )



# function to convert 1-hot array into "useme" tensor
# Input dimension: L x q
# Output Dimension L x q x L x q
# U_{i,a,j,b} = \delta_{x_i, a} \delta{x_j, b} for i \neq j
# U_{i,a,i,b} =  0 for all i, a, b
@jit
def SeqUseMe(seq_1hot):
   L = seq_1hot.shape[0]
   return np.tensordot(seq_1hot, seq_1hot, axes=0)  * np.reshape(1-np.eye(L), (L,1,L,1))

# function to convert 1-hot MSA array into "useme" tensor
# Input dimension: N x L x q
# output dimension: N x L x q x L x q
MSA_UseMe = vmap(SeqUseMe, in_axes = 0, out_axes = 0)


# function to return a 1-hot array multiplied by a scalar weight
# useful for calculating weighted MSA frequencies
@jit
def Seq_WeightedOneHot(seq_1hot, wgt):
   return wgt * seq_1hot

# Function to create a weighted 1-Hot
# each sequence's useme entry is multiplied by a scalar weight
#
MSA_WeightedOneHot = vmap(Seq_WeightedOneHot, in_axes = (0, 0), out_axes = 0)

# function to return a sequences binary usme array by a scalar weight
# useful for calculating weighted pairwise MSA frequencies
@jit
def Seq_WeightedUseMe(seq_1hot, wgt):
   L = seq_1hot.shape[0]
   return wgt * (np.tensordot(seq_1hot, seq_1hot, axes=0)  * np.reshape(1-np.eye(L), (L,1,L,1)))

# function to multiply a MSA's useme arrray by a set of sequence weights
MSA_WeightedUseMe = vmap(Seq_WeightedUseMe , in_axes = (0, 0), out_axes = 0)


# Function: Seq_OnehotInverse()
# Comments: Given a sequence in 1-hot representation,
#           return the same sequence in categorical representation.
#           Each position in the categorical array is defined as
#
#           seq[i] = argmax seq_1hot[i,:]
#
#           This function is the inverse of nn.one_hot()
#
#
# Args:    seq: L x q binary, 1-hot representation of a sequence
#
# Returns:  L -element categorical representation of the sequence
@jit
def Seq_OnehotInverse(seq_1hot):
   return np.argmax(seq_1hot, axis=1)


# Function: MSA_OnehotInverse
# Comments: Given an alignment in 1-hot representation,
#           return the same sequence in categorical representation.
#           Each position in the categorical array is defined as
#
#           msa[n,i] = argmax msa_1hot[n,i,:]
#
#           This function is the Inverse of MSA_Onehot()
#
# Args:    msa_1hot: n x L x q binary, 1-hot representation of an MSA
#
# Returns:  n x L  categorical representation of the MSA
MSA_OnehotInverse = vmap(Seq_OnehotInverse, in_axes = 0, out_axes = 0)


