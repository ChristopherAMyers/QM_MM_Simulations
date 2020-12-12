from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm.app import element
from simtk.openmm import *
from simtk.unit import *
import argparse
import numpy as np
import sys
from scipy.spatial.transform import Rotation as rot


# pylint: disable=no-member
import simtk.unit as unit
picosecond = picoseconds = unit.picosecond
nanometer = nanometers = unit.nanometer
femtoseconds = unit.femtoseconds
# pylint: enable=no-member

sys.path.insert(1, "/network/rit/lab/ChenRNALab/bin/Pymol2.3.2/pymol/lib/python3.7/site-packages")
#from pymol import cmd, stored # pylint: disable=import-error
import pdb_to_qc

class SolventAdder(object):
    def __init__(self):
        self._water_coords = np.array([[  0.0000000000,    0.0000000000,    0.1184242041],  # O
                                       [ -0.8012202255,    0.0000000000,   -0.4278321020],  # H
                                       [  0.8012202255,    0.0000000000,   -0.4278321020]]) # H
        self._o2_coords = np.array([[  0.0000000000,    0.0000000000,    0.0000000000],  # O
                                       [  1.2590000000,    0.0000000000,    0.0000000000]]) # O
        self._water_vdw_radii = np.array([3.1507, 0.4000, 0.4000])
        self._o2_vdw_radii = np.array([3.1507, 3.1507])*0.5

    def copy_topology(self, topology):
        newTopology = Topology()
        newAtoms = {}
        newPositions = []*nanometer
        for chain in topology.chains():
            newChain = newTopology.addChain(chain.id)
            for residue in chain.residues():
                newResidue = newTopology.addResidue(residue.name, newChain, residue.id, residue.insertionCode)
                for atom in residue.atoms():
                    newAtom = newTopology.addAtom(atom.name, atom.element, newResidue, atom.id)
                    newAtoms[atom] = newAtom
                    newPositions.append(copy.deepcopy(positions[atom.index]))
        for bond in topology.bonds():
            newTopology.addBond(newAtoms[bond[0]], newAtoms[bond[1]])
        
        return newTopology

    def add_solvent(self, solute_coords, topology, forcefield, radius, center_coord, n_waters=0, n_o2=0):
        #   get vdw radii from forcefield
        system = forcefield.createSystem(topology)
        nonbonded = None
        for i in range(system.getNumForces()):
            if isinstance(system.getForce(i), NonbondedForce):
                nonbonded = system.getForce(i)
        if nonbonded is None:
            raise ValueError('The ForceField does not specify a NonbondedForce')
        vdw_radii = []
        for i in range(system.getNumParticles()):
            params = nonbonded.getParticleParameters(i)
            vdw_radii.append(params[1]/angstrom)

        solute_coords = np.array(solute_coords/angstrom)
        all_coords = np.copy(solute_coords)
        all_vdw_radii = np.copy(vdw_radii)
        new_waters = []
        new_o2s = []

        
        for mol_type in ['o2', 'water']:
            if mol_type == 'water':
                mol_coords = self._water_coords
                mol_vdw_radii = self._water_vdw_radii
                max_tries = n_waters*1000
                max_mols = n_waters
            else:
                mol_coords = self._o2_coords
                mol_vdw_radii = self._o2_vdw_radii
                max_tries = n_o2*1000
                max_mols = n_o2

            n_accepted = 0
            for n in range(max_tries):
                #   random orientation of solvent molecule to place
                mat = rot.random().as_matrix()
                rot_mol = np.array([mat @ x for x in mol_coords])

                #   random location of solvent molecule
                rand_dist = np.random.random()*radius #pylint: disable=no-member
                p, t, g = rot.random().as_euler('zyz')
                x = rand_dist * np.sin(t)*np.cos(p)
                y = rand_dist * np.sin(t)*np.sin(p)
                z = rand_dist * np.cos(t)
                rand_loc = np.array([x, y, z])

                #   place solvent
                location = rand_loc + center_coord/angstrom
                mol = rot_mol + location

                #   determine collisions
                accept = True
                for i, atom in enumerate(mol):
                    sigma = mol_vdw_radii[i]
                    dists = np.linalg.norm(all_coords - atom, axis=1)
                    n_intersects = np.sum(dists < (all_vdw_radii + sigma)/2)
                    if n_intersects != 0:
                        #where = np.where(dists < all_vdw_radii)[0]
                        #print(where, dists[where], all_vdw_radii[where], rand_dist)
                        #input()
                        accept = False
                        break
                if accept:
                    all_coords = np.append(all_coords, mol, axis=0)
                    all_vdw_radii = np.append(all_vdw_radii, mol_vdw_radii*1.4)
                    if mol_type == 'water':
                        new_waters.append(mol)
                    else:
                        new_o2s.append(mol)
                    n_accepted += 1

                if n_accepted == max_mols:
                    break

            print("Placed {:d} {:s}s".format(n_accepted, mol_type))

        #   add waters to topology
        new_positions = list(positions)
        new_topology = self.copy_topology(topology)
        newChain = list(new_topology.chains())[0]
        for o2 in new_o2s:
            new_residue = new_topology.addResidue('EXT', newChain)
            new_topology.addResidue(new_residue, newChain)
            #   add atoms
            mol_atoms = []
            atom1 = new_topology.addAtom('O1', element.oxygen, new_residue)
            atom2 = new_topology.addAtom('O2', element.oxygen, new_residue)
            #   add bonds
            new_topology.addBond(atom1, atom2)


        for water in new_waters:
            new_residue = new_topology.addResidue('HOH', newChain)
            new_topology.addResidue(new_residue, newChain)
            #   add atoms
            mol_atoms = []
            mol_atoms.append(new_topology.addAtom('O', element.oxygen, new_residue))
            mol_atoms.append(new_topology.addAtom('H1', element.hydrogen, new_residue))
            mol_atoms.append(new_topology.addAtom('H2', element.hydrogen, new_residue))
            #   add bonds
            for atom1 in mol_atoms:
                if atom1.element == element.oxygen:
                    for atom2 in mol_atoms:
                        if atom2.element == element.hydrogen:
                            new_topology.addBond(atom1, atom2)
            for coord in water:
                new_positions.append(coord)




        return new_topology, all_coords*angstrom

if __name__ == "__main__":

    pdb = PDBFile(sys.argv[1])
    pdb_to_qc.add_bonds(pdb)
    data, bondedToAtom = pdb_to_qc.determine_connectivity(pdb.topology)
    ff_loc = '/network/rit/lab/ChenRNALab/awesomeSauce/2d_materials/ZnSe/quant_espres/znse_2x2/qm_mm/forcefield'
    forcefield = ForceField(os.path.join(ff_loc, 'forcefields/forcefield2.xml'))
    unmatched_residues = forcefield.getUnmatchedResidues(pdb.topology)
    [templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(pdb.topology)
    for n, template in enumerate(templates):
        residue = residues[n]
        atom_names = []
        for atom in template.atoms:
            if residue.name in ['EXT', 'OTH']:
                atom.type = 'OTHER-' + atom.element.symbol
            else:
                atom.type = residue.name + "-" + atom.name.upper()
            atom_names.append(atom.name)

        # Register the template with the forcefield.
        template.name += str(n)
        forcefield.registerResidueTemplate(template)
    system = forcefield.createSystem(pdb.topology)

    adder = SolventAdder()
    positions = pdb.getPositions(True)
    center = np.mean(positions[[94, 95, 98, 99]], axis=0)
    print(" Center : ", center)
    for n in range(3):
        new_top, new_pos = adder.add_solvent(positions, pdb.topology, forcefield, 4.0, center, n_waters=0, n_o2=8)
        with open('init.{:d}.pdb'.format(n + 1), 'w') as file:
            pdb.writeFile(new_top, new_pos, file)

