import jax.numpy as np
import numpy as onp

class alphabet(object):

   def __init__ (self, name, charmap, idxmap, q):

      self.name = name
      self.charmap = charmap
      self.idxmap = idxmap
      self.q = q


# define RNA alphabet
ABC_RNA = alphabet(name    = "RNA",
                   q       = 5,
                   charmap = {'A': 0,  # Adenine
                              'C': 1,  # Cytosine
                              'G': 2,  # Guanine
                              'U': 3,  # Uracil
                              'T': 3,  # Thymine
                              '-': 4,  # Deletion (gap)

                              # Degenerate characters
                              '.': 4,  # Insertion (gap)
                              'X': 4,  # Unknown nucleotide
                              'B': 4,  # Not A
                              'D': 4,  # Not C
                              'H': 4,  # Not G
                              'K': 4,  # Keto (G or U)
                              'M': 4,  # Amino (A or C)
                              'N': 4,  # Any (A, C, G, or U)
                              'R': 4,  # Purine (A or G)
                              'S': 4,  # Strong (C, G)
                              'V': 4,  # Not T or U
                              'W': 4,  # Weak (A or U)
                              'Y': 4}, # Pyrimidine (C or U)
                   idxmap  = {0: 'A',
                              1: 'C',
                              2: 'G',
                              3: 'U',
                              4: '-'})


# define DNA alphabet
ABC_DNA = alphabet(name    = "RNA",
                   q       = 5,
                   charmap = {'A': 0,
                              'C': 1,
                              'G': 2,
                              'U': 3,
                              'T': 3,
                              '-': 4,

                              # Degenerate/nonstandard characters (read as gap)
                              '.': 4,  # Insertion (gap)
                              'X': 4,  # Unknown nucleotide
                              'B': 4,  # Not A
                              'D': 4,  # Not C
                              'H': 4,  # Not G
                              'K': 4,  # Keto (G or T)
                              'M': 4,  # Amino (A or C)
                              'N': 4,  # Any (A, C, G, or T)
                              'R': 4,  # Purine (A or G)
                              'S': 4,  # Strong (C, G)
                              'V': 4,  # Not T or U
                              'W': 4,  # Weak (A or T)
                              'Y': 4}, # Pyrimidine (C or T)
                   idxmap  = {0: 'A',
                              1: 'C',
                              2: 'G',
                              3: 'T',
                              4: '-'})



# Define protein alphabet
ABC_AMINO = alphabet(name    = "Amino",
                     q       = 21,
                      charmap = {'A': 0,  # Alanine (Ala)
                                 'C': 1,  # Cysteine (Cys)
                                 'D': 2,  # Aspartic Acid/Aspartate (Asp)
                                 'E': 3,  # Glutamic Acid/Glutamate (Glu)
                                 'F': 4,  # Phenylalanine (Phe)
                                 'G': 5,  # Glycine (Gly)
                                 'H': 6,  # Histidine (His)
                                 'I': 7,  # Isoleucine (Ile)
                                 'K': 8,  # Lysine (Lys)
                                 'L': 9,  # Leucine (Leu)
                                 'M': 10, # Methionine (Met)
                                 'N': 11, # Asparagine (Asn)
                                 'P': 12, # Proline (Pro)
                                 'Q': 13, # Glutamine (Glu) aka Cuteamine or Q-tamine
                                 'R': 14, # Arginine (Arg)
                                 'S': 15, # Serine (Ser)
                                 'T': 16, # Threonine (Thr)
                                 'V': 17, # Valine (Val)
                                 'W': 18, # Tryptophan (Trp)
                                 'Y': 19, # Tyrosine (Tyr)
                                 '-': 20, # Deletion (gap)

                                 # Degenerate/nonstandard characters (read as gap)
                                 '.': 20, # Insertion (gap)
                                 'X': 20, # Any amino acid
                                 'O': 20, # Pyrrolysine (pyl)
                                 'Z': 20, # Glutamine or Glutamic Acid (Q or E)
                                 'U': 20, # Selenocysteine (Sec)
                                 'B': 20, # Asparagine or Aspartic Acid (N or D)
                                 'J': 20, # Leucine or Isoleucine  (L or I)
                                 '~': 20},
                      idxmap  = {0:  'A',
                                 1:  'C',
                                 2:  'D',
                                 3:  'E',
                                 4:  'F',
                                 5:  'G',
                                 6:  'H',
                                 7:  'I',
                                 8:  'K',
                                 9:  'L',
                                 10: 'M',
                                 11: 'N',
                                 12: 'P',
                                 13: 'Q',
                                 14: 'R',
                                 15: 'S',
                                 16: 'T',
                                 17: 'V',
                                 18: 'W',
                                 19: 'Y',
                                 20: '-'})



