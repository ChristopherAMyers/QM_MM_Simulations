from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm import *
from openmmtools.integrators import VelocityVerletIntegrator
from simtk.openmm.openmm import *
from simtk.unit import *
import argparse
import numpy as np
from numpy.random import random
import sys

from scipy.spatial.transform import Rotation as rot

# pylint: disable=no-member
import simtk
picoseconds = simtk.unit.picoseconds
picosecond = picoseconds
nanometer = nanometers = simtk.unit.nanometer
femtoseconds = simtk.unit.femtoseconds
# pylint: enable=no-member

from sim_extras import *
import pdb_to_qc



def print_pressure(simulation, masses_in_kg):
    state = simulation.context.getState(getPositions=True, getForces=True, getVelocities=True, getEnergy=True)

        #   calculate kinetic energies
    system = simulation.context.getSystem()
    vel = state.getVelocities(True).in_units_of(meters/second)
    v2 = np.linalg.norm(vel, axis=1)**2
    k = 1.380649E-23

    #   calculate temperatures from kinetic energies
    temp_all = np.sum(v2*masses_in_kg)/(len(v2) * k * 3)

    #   calculate the center of mass
    pos = state.getPositions(True)
    weighted_pos = masses_in_kg[:, None] * pos
    com = np.sum(weighted_pos, axis=0)/np.sum(masses_in_kg)

    pos -= com
    forces = state.getForces(True)

    pressure_int = -0.5*np.sum(pos * forces)
    print(pressure_int)
    print(state.getKineticEnergy())
    exit()

def add_sphere_water(solute_coords, topology, forcefield, radius=1.5*nanometers, origin=np.array([0, 0, 0])*nanometers):

    #   get vdw radii of solute from forcefield
    vdw_padding= 1.1
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
        vdw_radii.append(params[1]/nanometers)
    vdw_radii = np.array(vdw_radii)

    solvent = Modeller(Topology(), [])
    solvent.addSolvent(ForceField('amber14/tip3pfb.xml'), neutralize=False, 
    boxSize=Vec3(radius*2, radius*2, radius*2)/nanometers )

    #   import pre-computed water box
    pos = numpy.array(solvent.getPositions()/nanometers)*nanometers

        #   center the box
    center = np.mean(pos, axis=0)
    pos -= center

    #   translate box by random vector
    box_len = 2*radius
    trans_vec = random(3)*box_len - box_len/2
    for n, val in enumerate(trans_vec):
        for res in solvent.topology.residues():
            shift = 0.0*nanometers
            for atom in res.atoms():
                if atom.element.symbol == 'O':
                    if val/nanometers > 0:
                        if pos[atom.index][n] < val - box_len/2:
                            shift = box_len
                    else:
                        if pos[atom.index][n] > val + box_len/2:
                            shift + -box_len
                    break
            for atom in res.atoms():
                pos[atom.index][n] += shift

    #   re-center positions
    center = np.mean(pos, axis=0)
    pos -= center

    #   rotate to new random orientation
    mat = rot.random().as_matrix()
    rot_pos = (np.array(pos/pos.unit) @ mat)*nanometers

    #   move to designated origin
    rot_pos = rot_pos + origin

    #   delete solvent not in sphere or that intersects solute vdw radius
    modeller = Modeller(solvent.topology, rot_pos)
    modeller.add(topology, solute_coords)
    to_delete = []
    for res in modeller.topology.residues():
        for atom in res.atoms():
            if atom.element.symbol == 'O' and res.name == 'HOH':
                oxy_pos = rot_pos[atom.index]
                oxy_sigma = 0.312
                rad_norm = np.linalg.norm(oxy_pos - origin)*nanometers

                dists = np.linalg.norm(solute_coords - oxy_pos, axis=1)
                n_intersects = np.sum(dists < (vdw_radii + oxy_sigma)*vdw_padding/2)

                if rad_norm> radius or n_intersects != 0:
                    to_delete.append(res)
    modeller.delete(to_delete)

    solvent.topology = modeller.topology
    solvent.positions = modeller.positions
    return solvent


if __name__ == "__main__":

    print(" Importing Solute")
    pdb = PDBFile(sys.argv[1])
    pdb_to_qc.add_bonds(pdb, remove_orig=True)
    forcefield = ForceField('/network/rit/lab/ChenRNALab/awesomeSauce/2d_materials/ZnSe/quant_espres/znse_2x2/qm_mm/forcefield/forcefields/forcefield2.xml', 'amber14/tip3pfb.xml')
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

    origin = np.mean(pdb.getPositions(True)[[94]], axis=0)
    print(" Center : ", origin)

    n_atoms_init = pdb.topology.getNumAtoms()
    print(" Initial number of atoms: {:d}".format(n_atoms_init))
    print(" Adding Solvent")
    mols = add_sphere_water(pdb.positions, pdb.topology, forcefield, origin=origin)
    system = forcefield.createSystem(mols.topology, nonbondedMethod=NoCutoff,
            nonbondedCutoff=1*nanometer, constraints=HBonds)
    n_atoms_final = mols.topology.getNumAtoms()
    print(" Final number of atoms:   {:d}".format(n_atoms_final))
    print(" Number of solvent atoms: {:d}".format(n_atoms_final - n_atoms_init))

    out_file_loc = 'solvated.pdb'
    print(" Writing %s" % out_file_loc)
    PDBFile.writeModel(mols.topology, mols.positions, open(out_file_loc, 'w'))
   
    #   change to True for test simulation stuff
    if False:        
        integrator = LangevinIntegrator(300*kelvin, 1/(100*femtoseconds), 0.001*picoseconds)
        simulation = Simulation(mols.topology, system, integrator)
        simulation.context.setPositions(mols.positions)
        simulation.context.setVelocitiesToTemperature(300*kelvin)
        print(" Minimizing")
        simulation.minimizeEnergy()
        simulation.reporters.append(PDBReporter('output.pdb', 5))
        simulation.reporters.append(StateDataReporter('stats.txt', 1, step=True,
            potentialEnergy=True, temperature=True, separator=' ', ))

        masses_in_kg = np.array([system.getParticleMass(n)/dalton for n in range(simulation.topology.getNumAtoms())])*(1.67377E-27)
        
        print(" Running")
        for n in range(10):
            simulation.step(1)
            simulation.topology.getNumAtoms
            #print_pressure(simulation, masses_in_kg)
    
    print(" Done")