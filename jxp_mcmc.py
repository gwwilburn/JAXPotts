import jax.numpy as np
from jax import vmap, pmap, jit, random, lax
from jax.ops import index, index_add, index_update
import jax.nn as nn
from jxp_potts import *

@jit
def MCMC_SpinFlip(h,e, seq_1hot, H, key):

   (L,q) = h.shape

   # advance RNG
   key, subkey_site, subkey_char, subkey_accept = random.split(key, 4)

   # choose site to flip
   i = random.choice(subkey_site, np.arange(L))

   # pick random character for chosen site
   a = random.choice(subkey_char, np.arange(q))

   # make proposal sequence
   seq_1hot_tmp = index_update(seq_1hot, index[i,:], nn.one_hot(a, q))

   H_tmp = Potts_ScoreSeqCore(h,e,seq_1hot_tmp)

   accept_prob = np.exp(H_tmp - H)


   flip = np.zeros((L,q),dtype=np.bool_)
   accept_draw = random.uniform(subkey_accept)
   flip = index_update(flip, index[i,:],  accept_draw < accept_prob)

   return np.where(flip, seq_1hot_tmp, seq_1hot), np.where( accept_draw < accept_prob, H_tmp, H), key

def MCMC_SeqEmit(h, e, key, nflip):
   (L,q) = h.shape

   seq_1hot = nn.one_hot(random.choice(key, np.arange(0,q), shape = (1,L))[0],q)
   H = Potts_ScoreSeqCore(h,e,seq_1hot)

   @jit
   def loop_fun_scan(loop_carry, i):
      h, e, seq_1hot, H, key = loop_carry
      seq_1hot, H, key = MCMC_SpinFlip(h, e, seq_1hot, H, key)
      return (h, e, seq_1hot, H, key), i

   (h, e, seq_1hot, H, key), i = lax.scan(loop_fun_scan, (h,e,seq_1hot, H, key), None, length = nflip)

   return seq_1hot, H, key

MCMC_MSAEmit = vmap(MCMC_SeqEmit, in_axes = (None, None, 0, None), out_axes = (0,0,0))


