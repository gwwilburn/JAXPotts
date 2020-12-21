import argparse

import jax.numpy as np
from jax import random

from jxp_alphabet import *
from jxp_mcmc import *
from jxp_msa import *
from jxp_potts import *


### setup arguments
parser = argparse.ArgumentParser(description=
         'Emit aligned sequences from a Potts model using MCMC',
         formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(dest = "potts_inpath",
                    help="Input potts model path")

parser.add_argument(dest = "msa_outpath", help=
                    "Output MSA path")
# alphabet arguments
parser.add_argument("--rna", action='store_true',help=
                    'Use switch to indicate we are dealing with RNA')

# MCMC arguments
parser.add_argument("--Nseq", action='store', nargs=1, type=int, default = [1], help=
                    "Number of MCMC sequences generated at each step of MCMC algorithm")

parser.add_argument("--Nflip", action='store', nargs=1, type=int, default = [2500], help=
                    "Number of MCMC spin flips performed before returning sequence.")

# pRNG arguments
parser.add_argument("--seed", action='store', nargs=1, type=int, default = [0], help=
                    "Seed for pRNG.")


if __name__ == "__main__":
   print("hello world")

   args = parser.parse_args()

   potts_inpath = args.potts_inpath
   msa_outpath  = args.msa_outpath

   Nseq  = np.int32(args.Nseq[0])
   Nflip = np.int32(args.Nflip[0])
   seed  = args.seed[0]

   abc = ABC_AMINO
   if args.rna:
      abc = ABC_RNA

   # create pRNG Keys
   key = random.PRNGKey(seed)
   keys = random.split(key, Nseq)

   # read potts model
   potts = Potts_Read(potts_inpath, abc)

   # generate aligned sequences
   ax_1hot_mcmc, H_mcmc, keys = MCMC_MSAEmit(potts.h, potts.e, keys, Nflip)

   # create MSA object
   ax_mcmc = MSA_OnehotInverse(ax_1hot_mcmc)
   msa_mcmc = MSA(abc =abc, ax = ax_mcmc, ax_1hot = ax_1hot_mcmc)

   # add names for synthetic sequences
   for n in range(0, Nseq):
      seqname = "seq_{}".format(n)
      msa_mcmc.sqname.append(seqname)

   # write msa
   MSA_Write(msa_mcmc, msa_outpath, fileformat="afa")

