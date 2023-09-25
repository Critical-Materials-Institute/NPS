
"""
Simple example script demonstrating how to use the PeptideBuilder library.

The script generates a peptide consisting of six arginines in alpha-helix
conformation, and it stores the peptide under the name "example.pdb".
"""

import sys
sys.path.append('/g/g90/zhou6/lassen-space/PeptideBuilder')

from PeptideBuilder import Geometry
import PeptideBuilder


# create a peptide consisting of 6 glycines
# geo = Geometry.geometry("G")
# geo.phi = -60
# geo.psi_im1 = -40
# structure = PeptideBuilder.initialize_res(geo)

def seq_dihedral2peptide(seq, angle):
    for i, res_name in enumerate(seq):
        res = Geometry.geometry(res_name)
        res.phi = angle[2*i]
        res.psi_im1 = angle[2*i+1]
        if i == 0:
            structure = PeptideBuilder.initialize_res(res)
        else:
            PeptideBuilder.add_residue(structure, res)
    # add terminal oxygen (OXT) to the final glycine
    PeptideBuilder.add_terminal_OXT(structure)
    return structure
