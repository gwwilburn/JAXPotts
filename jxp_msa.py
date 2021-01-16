import jax.numpy as np
from jax import vmap, jit
from jax.ops import index, index_add, index_update
import numpy as onp
import jax.nn as nn

from jxp_alphabet import *
from jxp_msa import *
from functools import partial

import sys
import string

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
   elif fileformat.lower() == "pfam":
      ret_msa = MSA_ReadPfam(path, abc, name)
   elif fileformat.lower() in ["fasta", "afa"]:
      ret_msa = MSA_ReadAFA(path, abc, name)
   else:
      print("Invalid file format! Returning None")

   return ret_msa

def MSA_ReadStockholm( path, abc, name):
   return None

def MSA_ReadPfam( path, abc, name):
   sqname = []
   msa_array = []
   ss_cons = ""
   rf = ""


   af = open(path, 'r')

   in_aligment = False

   for line in af:
      line = line.rstrip()

      if "# STOCKHOLM" in line:
         in_alignment = True

      if not in_alignment:
         continue

      # remove spacing
      linelist = line.split(' ')
      linelist= [token for token in linelist if token != '']

      if len(linelist) == 0:
         continue
      if linelist[0] == "//":
         break


      # handle markup lines
      if list(linelist[0])[0] == '#':

         # per-file markup lines
         # NEEDS TO BE FINISHED
         if linelist[0] == "#=GF":
            continue

         # per-sequence markup lines
         # NEEDS TO BE FINISHED
         # ESPECIALLY SEQUENCE WEIGHTS
         elif linelist[0] == "#=GS":
            #if linelist[2] == "WT":
            #   wgt.append(float(linelist[3]))
            continue

         # per-column markup lines
         elif linelist[0] == "#=GC":

            if linelist[1] == "SS_cons":
               ss_cons = linelist[2]

            elif linelist[1] == 'RF':
               rf = linelist[2]

            #elif linelist[1] == "PP":
            #   pp_cons += linelist[2]

         # per-residue markup
         # NEEDS TO BE FINISHED
         elif linelist[0] == "#=GR":
            continue

      # extract sequences
      else:

         if not len(linelist) == 2:
            continue

         sqname.append(linelist[0])
         msa_array.append(list(linelist[1]))


   # get alignment dimensions
   L = len(msa_array[0])
   N = len(msa_array)


   # convert characters to integers
   for n in range(0, N):
      msa_array[n] =  Seq2Idx(msa_array[n], abc)
   ax = np.array(msa_array,dtype=np.int8)

   ret_msa = MSA(abc = abc, ax = ax, L = L, N = N, sqname = sqname)

   if len(ss_cons) > 0:
      ret_msa.ss_cons = onp.array(list(ss_cons))
      if abc == ABC_RNA:
         ret_msa.bp_map = MSA_BasePairMap(ret_msa.ss_cons)

   if len(rf) > 0:
      ret_msa.rf = onp.array(list(rf))

   return ret_msa

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

def MSA_BasePairMap(ss_cons):
   L = ss_cons.size
   bp_map = (-1)*np.ones(L, dtype=np.int32)

   bra_list_nest = ['(', '<', '[', '{']
   ket_list_nest = [')', '>', ']', '}']

   bra_list_pk = list(string.ascii_uppercase)
   ket_list_pk = list(string.ascii_lowercase)

   bra_list = bra_list_nest + bra_list_pk
   ket_list = ket_list_nest + ket_list_pk

   elev       = np.zeros(L, dtype = np.int32)
   elev_count = 0
   last_bp    = '' # keep track of last bp

   for i in range(0,L):
      if ss_cons[i] in bra_list:

         bra = ss_cons[i]
         ket = ket_list[bra_list.index(bra)]

         # find position of matching ket
         count = 1
         j = i

         # march over downstream positions until we find the matching ket
         while count > 0:
            j += 1
            c = ss_cons[j]

            #if we see a bra of this type, increase count
            if ss_cons[j] == bra:
               count += 1

            #if we see a ket of this type, decrease count
            elif ss_cons[j] == ket:
               count -= 1

         bp_map = index_update(bp_map, index[i], j)
         bp_map = index_update(bp_map, index[j], i)

   return bp_map

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
#
# args:    -ax_1hot: A digital MSA in 1 hot format (N x L x q)
#
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

# Function: MSA_WeightedFrequencies
# Comments: Function to calculate single-site and pairwise
#           frequencies with provided relative sequence weights.
#           If no weights are provided, then uniform weights
#           are used and the result is the same as that from
#           MSA_Frequencies().
#
# args:    -msa: MSA object
#          -weights:Relative sequence weights (N-dimensional array)
#
# returns: -f1: L x q array of (possibly weighted) single site frequencies
#          -f2: L x q x L x q symmetric array of (possibly weighted) pairwise
#           frequencies (with no terms on i=j diagonal)
#

def MSA_WeightedFrequencies(msa, weights = None):

   # if no weights given, use uniform weights
   if weights is None:
      weights = np.ones(msa.N, dtype=np.int32)

   # sum weights (denominator for frequency calculation)
   Nw = np.sum(weights)

   # make sure 1-hot tensor exists
   if msa.ax_1hot is None:
      msa.ax_1hot = MSA_Onehot(msa.ax,msa.abc.q)

   wgt_1hot = MSA_WeightedOneHot(msa.ax_1hot, weights)
   wgt_useme = MSA_WeightedUseMe(msa.ax_1hot, weights)

   #mask_useme = MSA_MaskUseme(ss_cons, pknot)

   f1 = np.sum(wgt_1hot, axis=0)  / Nw
   f2 = np.sum(wgt_useme, axis=0) / Nw

   return f1, f2

# Function: MSA_MaskedFrequencies
# Comments: Given an MSA and consensus secondary structure annotation,
#           Calculate single-site frequencies as normal, but observed
#           pairwise frequiencies only for column pairs corresponding
#           to annotated base pairs. All other pairwise frequency values
#           are set to the product of corresponding single-site frequencies.
#           This function can handle relative sequence weights if provided.
#
# Args:    msa:     An MSA object (ss_cons required)
#          weights: Optional relative sequence weights (N-dimensional array)
#          pknot:   Boolean switch to handle non-nested annotated bp
#
# Returns: f1: L x q array of (possibly weighted) single site frequencies
#          f2: L x q x L x q symmetric array of (possibly wighted)
#              pairwise frequencies, calculated as described above
#              (with no terms on i=j diagonal)
#
def MSA_MaskedFrequencies(msa, weights=None, pknot=False):

   if msa.ss_cons is None:
      sys.exit("\n\nError: MSA_MaskedFrequencies() requires secondary structure annotation (ss_cons)\n\n")

   L = msa.L
   q = msa.abc.q

   # list of SS_cons characters corresponding to 5' base pairs
   bra_list = ['(', '<', '[', '{']
   if pknot:
      bra_list += list(string.ascii_uppercase)

   # if no weights given, use uniform weights
   if weights is None:
      weights = np.ones(msa.N, dtype=np.int32)

   # sum weights (denominator for frequency calculation)
   Nw = np.sum(weights)

   # make sure 1-hot tensor exists
   if msa.ax_1hot is None:
      msa.ax_1hot = MSA_Onehot(msa.ax,msa.abc.q)

   wgt_1hot = MSA_WeightedOneHot(msa.ax_1hot, weights)
   wgt_useme = MSA_WeightedUseMe(msa.ax_1hot, weights)

   # calculate single site frequencies
   f1 = np.sum(wgt_1hot, axis=0)  / Nw

   # initially set pairwise frequencies to be product of single-site freqs
   # (with no i = j diagonal term)
   f2 =  np.tensordot(f1, f1, axes=0)  * np.reshape(1-np.eye(L), (L,1,L,1))

   # Loop over MSA positions
   # if position is annotated as 5' position of basepair (bra),
   # reset pairwise frequencies for this position and its pair to
   # weighted, observed pairwise frequencies in MSA
   for i in range(0, msa.L):
      ssi = msa.ss_cons[i]

      if ssi in bra_list:
         j  = msa.bp_map[i]
         f2 = index_update(f2, index[i,:,j,:], np.sum(wgt_useme[:,i,:,j,:], axis=0) / Nw)
         f2 = index_update(f2, index[j,:,i,:], np.sum(wgt_useme[:,j,:,i,:], axis=0) / Nw)

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


