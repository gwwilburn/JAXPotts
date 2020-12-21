import jax.numpy as np
from jax import vmap, jit
from jax.ops import index, index_add, index_update

@jit
def Prior_Laplace(f1, f2, N, C):

   (L,q) = f1.shape
   qf = np.float32(q)
   Nf = np.float32(N)

   # new normalization: 1 / (eff. seq. number)
   nrm = 1. / (Nf + C)

   # binary L x q x L x q term: keeps us from adding pseudocounts for f_ii
   no_diag = np.reshape(1-np.eye(L), (L,1,L,1))

   f1_prior = nrm *  ( (C/qf)+ Nf*f1 )
   f2_prior = nrm *  ( ((C/(qf*qf))* no_diag )  + Nf*f2)

   return f1_prior, f2_prior

