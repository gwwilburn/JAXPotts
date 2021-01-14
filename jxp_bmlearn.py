import jax.numpy as np
from jax.ops import index, index_update
from jax import random, lax
import argparse
import sys
from resource import getrusage, RUSAGE_SELF

from jxp_mcmc import *
from jxp_alphabet import *
from jxp_msa import *
from jxp_prior import *
from jxp_potts import *
from jxp_msaweight import *

def BMLearn_Naive(f1, f2, epsilon, niter, nseq, nflip, key):

   (L, q) = f1.shape

   #initialize potts model params
   h = np.log(f1)
   e = np.zeros((L,q,L,q), dtype = np.float32)

   """
   for n in range(0,niter):
      print(n)

      key, subkey = random.split(key)
      keys = random.split(subkey, nseq)

      ax_mcmc_onehot, H_mcmc  = MCMC_MSAEmit(h,e,keys,nflip)

      # estimate single-site, pairwise probabilities
      f1_mcmc, f2_mcmc = MSA_Frequencies(ax_mcmc_onehot)

      h = index_update(h, index[:,:], h +  epsilon * (f1 - f1_mcmc))
      e = index_update(e, index[:,:,:,:], e + epsilon *(f2 - f2_mcmc))

      key, subkey = random.split(key)
   """

   @jit
   def loop_fun_scan(loop_carry, n):

      h, e, key, = loop_carry

      key, subkey = random.split(key)
      keys = random.split(subkey, nseq)

      ax_mcmc_onehot, H_mcmc = MCMC_MSAEmit(h,e,keys,nflip)

      f1_mcmc, f2_mcmc = MSA_Frequencies(ax_mcmc_onehot)

      h = index_update(h, index[:,:], h +  epsilon * (f1 - f1_mcmc))
      e = index_update(e, index[:,:,:,:], e + epsilon *(f2 - f2_mcmc))

      return (h, e, key), n

   (h, e, key), n = lax.scan(loop_fun_scan, (h, e, key), None, length=niter)

   return h, e


def BMLearn_Naive2(f1, f2, epsilon, niter, nseq, nflip, key):

   (L, q) = f1.shape

   #initialize potts model params
   h = np.log(f1)
   e = np.zeros((L,q,L,q), dtype = np.float32)
   keys = random.split(key, nseq)

   for n in range(0,niter):
      ###print(n)

      ax_mcmc_onehot, H_mcmc, keys  = MCMC_MSAEmit(h,e,keys,nflip)

      # estimate single-site, pairwise probabilities
      f1_mcmc, f2_mcmc = MSA_Frequencies(ax_mcmc_onehot)

      h = index_update(h, index[:,:], h +  epsilon * (f1 - f1_mcmc))
      e = index_update(e, index[:,:,:,:], e + epsilon *(f2 - f2_mcmc))

   return h, e

def BMLearn_Convergence(f1, f2, c2, epsilon, niter, nseq, nflip, key):

   (L, q) = f1.shape

   #initialize potts model params
   h = np.log(f1)
   e = np.zeros((L,q,L,q), dtype = np.float32)
   keys = random.split(key, nseq)

   corr_err = np.zeros(niter)

   for n in range(0,niter):
      ###print(n)

      ax_mcmc_onehot, H_mcmc, keys  = MCMC_MSAEmit(h,e,keys,nflip)

      # estimate single-site, pairwise probabilities
      f1_mcmc, f2_mcmc = MSA_Frequencies(ax_mcmc_onehot)
      c2_mcmc = MSA_2PtCorrelations(f1_mcmc, f2_mcmc)

      err = np.max(np.abs(c2_mcmc - c2))

      h = index_update(h, index[:,:], h +  epsilon * (f1 - f1_mcmc))
      e = index_update(e, index[:,:,:,:], e + epsilon *(f2 - f2_mcmc))

      corr_err = index_update(corr_err, index[n], err)

   return h, e,  corr_err

def BMLearn_Persistent(f1, f2, c2, epsilon, niter, nseq, nflip, errthresh, key):

   (L, q) = f1.shape

   #initialize potts model params
   h = np.log(f1)
   e = np.zeros((L,q,L,q), dtype = np.float32)

   # generate initial sequences
   ax_1hot_mcmc = nn.one_hot(random.choice(key, np.arange(0,q), shape=(nseq,L)), q)

   # make nseq pRNG keys
   keys = random.split(key, nseq)

   # intiate correlation error array
   corr_err = np.zeros(niter+1)

   for n in range(0,niter):
      ax_1hot_mcmc, H_mcmc, keys  = MCMC_MSAEmitPersistent(h,e,ax_1hot_mcmc,keys,nflip)

      # estimate single-site, pairwise probabilities
      f1_mcmc, f2_mcmc = MSA_Frequencies(ax_1hot_mcmc)
      c2_mcmc = MSA_2PtCorrelations(f1_mcmc, f2_mcmc)

      err = np.max(np.abs(c2_mcmc - c2))
      corr_err = index_update(corr_err, index[n], err)

      # if we've reached the minimum error threshold, stop training!
      if err < errthresh:
         print("\nError threshold reached!\nGrad descent iteration: %d\n"  % n)
         break


      h = index_update(h, index[:,:], h +  epsilon * (f1 - f1_mcmc))
      e = index_update(e, index[:,:,:,:], e + epsilon *(f2 - f2_mcmc))

   # estimte frequencies via MCMC to estimate final model marginal  probabilities
   ax_1hot_mcmc, H_mcmc, keys  = MCMC_MSAEmitPersistent(h,e,ax_1hot_mcmc,keys,nflip)
   f1_mcmc, f2_mcmc = MSA_Frequencies(ax_1hot_mcmc)

   # estimate final correlation error
   c2_mcmc = MSA_2PtCorrelations(f1_mcmc, f2_mcmc)
   err = np.max(np.abs(c2_mcmc - c2))
   corr_err = index_update(corr_err, index[niter], err)


   return h, e, corr_err, f1_mcmc, f2_mcmc





### setup arguments
parser = argparse.ArgumentParser(description=
         'Train a Potts model using Boltzmann machine learning',
         formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(dest = "msa_inpath",
                    help="Input msa path")

parser.add_argument(dest = "potts_outpath", help=
                    "Output potts model path")

# alphabet arguments
parser.add_argument('--rna', action='store_true',help=
                    'Use switch to indicate we are dealing with RNA')

# learning algorithm arguments
parser.add_argument('--Niter', action='store', nargs=1, type=int, default = [5000], help=
                    "Number of iterations to run BML gradient descent algorithm.")

parser.add_argument('--Nseq', action='store', nargs=1, type=int, default = [1000], help=
                    "Number of MCMC sequences generated at each step of MCMC algorithm")

parser.add_argument('--Nflip', action='store', nargs=1, type=int, default = [2500], help=
                    "Number of MCMC spin flips performed before returning sequence.")

parser.add_argument('--epsilon', action='store', nargs=1, type=float, default = [0.02], help=
                    "Learning rate.")

parser.add_argument('--errthresh', action='store', nargs=1, type=float, default = [0.00], help=
                    "Threshold for correlation error (gradient descent stops if err <= <x>)")

# sequence weighting argument
parser.add_argument('--wpb', action='store_true',help=
                    "Use Henikoff position-based weights when calculating training freqs.")
# RNG arguments
parser.add_argument('--seed', action='store', nargs=1, type=int, default = [0], help=
                    "Seed for RNG.")

# optional output arguments
parser.add_argument('--errfile', action='store', nargs=1, type=str, default=None, help=
                    "Output array of correlation error at each step of gradient descent to <f> (.npy format)")

parser.add_argument('--f1file', action='store', nargs=1, type=str, default=None, help=
                    "Output L x q array of single-site frequencies estimated via MCMC at the end of gradieant descent to <f>")

parser.add_argument('--f2file', action='store', nargs=1, type=str, default=None, help=
                    "Output L x q x L x q array of pairwise frequencies estimated via MCMC at the end of gradient descent to <f>")



if __name__ == "__main__":

   args = parser.parse_args()

   msa_inpath   = args.msa_inpath
   potts_outpath = args.potts_outpath

   Niter   = np.int32(args.Niter[0])
   Nseq    = np.int32(args.Nseq[0])
   Nflip   = np.int32(args.Nflip[0])
   epsilon = np.float32(args.epsilon[0])
   errthresh = np.float32(args.errthresh[0])

   wpb =  args.wpb
   seed = args.seed[0]

   abc = ABC_AMINO
   if args.rna:
      abc = ABC_RNA

   msa_in = MSA_Read(path = msa_inpath, abc = abc)


   # calculate henikoff weights, if requested:
   if wpb:
      MSAWeight_PB(msa_in)

   #f1, f2 = MSA_Frequencies(msa_in.ax_1hot)
   f1, f2 = MSA_WeightedFrequencies(msa_in, msa_in.wgt)

   f1_plp, f2_plp = Prior_Laplace(f1, f2, msa_in.N, 1.0)

   c2_plp = MSA_2PtCorrelations(f1_plp, f2_plp)

   key = random.PRNGKey(seed)

   #h_bml, e_bml, corr_err = BMLearn_Convergence(f1_plp, f2_plp, c2_plp, epsilon, niter=Niter, nflip=Nflip, nseq=Nseq, key=key)
   h_bml, e_bml, corr_err, f1_mcmc, f2_mcmc = BMLearn_Persistent(f1_plp, f2_plp, c2_plp,
                                                                epsilon, niter=Niter, nflip=Nflip,
                                                                nseq=Nseq, errthresh=errthresh, key=key)


   pm_bml = PottsModel(h = h_bml, e = e_bml, abc = abc, L = msa_in.L)

   Potts_ShiftGaugeZeroSum(pm_bml)

   print("Peak memory usage (MB): ", np.int32(getrusage(RUSAGE_SELF).ru_maxrss / 1024))

   if args.errfile is not None:
      np.save(args.errfile[0], corr_err)

   if args.f1file:
      np.save(args.f1file[0], f1_mcmc)

   if args.f2file:
      np.save(args.f2file[0], f2_mcmc)


   Potts_Write(pm_bml, potts_outpath)
   #print("Peak memory usage (MB): ", np.int32(getrusage(RUSAGE_SELF).ru_maxrss / 1024))
