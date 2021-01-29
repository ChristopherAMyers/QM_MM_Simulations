from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm import * #pylint: disable=unused-wildcard-import
from simtk.openmm.openmm import * #pylint: disable=unused-wildcard-import
from simtk.unit import * #pylint: disable=unused-wildcard-import

# pylint: disable=no-member
import simtk.unit as unit
picosecond = picoseconds = unit.picosecond
nanometer = nanometers = unit.nanometer
femtoseconds = unit.femtoseconds
# pylint: enable=no-member

from mdtraj.reporters import HDF5Reporter
import numpy as np

def 


def determine_mult_from_coords(coords, topology, current_mult, qm_atoms, bond_dist=1.8*angstrom):
    
    oxygen_atoms = []
    for atom in topology.atoms():
        if atom.element.symbol == 'O' and atom.index in qm_atoms:
            oxygen_atoms.append(atom)
    
    bonds = set()
    new_mult = 1
    for i, atom_i in enumerate(oxygen_atoms):
        bonded_to_i = []
        for j, atom_j in enumerate(oxygen_atoms):
            if i != j:
                dist = np.linalg.norm((coords[atom_i.index] - coords[atom_j.index])/nanometers)*nanometers
                if dist <= bond_dist:
                    bonded_to_i.append(j)
        
        if len(bonded_to_i) == 1:
            bonded_to_i.append(i)
            bonds.add(tuple(sorted(bonded_to_i)))
        elif len(bonded_to_i) > 1:
            #   can't determine molecule, current frame might be
            #   in the middle of a transition. Return the old mult.
            return current_mult


    new_mult = int(len(bonds) * 2 + 1)
    return new_mult