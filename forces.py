from simtk.openmm.openmm import *
import numpy as np
import os

def add_ext_qm_force(qm_atoms, system):
    ext_force = CustomExternalForce('a*x + b*y + c*z - k + qm_energy')
    ext_force.addGlobalParameter('k', 0.0)
    ext_force.addGlobalParameter('qm_energy', 0.0)
    ext_force.addPerParticleParameter('a')
    ext_force.addPerParticleParameter('b')
    ext_force.addPerParticleParameter('c')
    for n in qm_atoms:
        ext_force.addParticle(n, [0, 0, 0])

    system.addForce(ext_force)

    return ext_force

def add_ext_mm_force(qm_atoms, system, charges):
    ext_force = CustomExternalForce('-q*(Ex*x + Ey*y + Ez*z - sum)')
    ext_force.addPerParticleParameter('Ex')
    ext_force.addPerParticleParameter('Ey')
    ext_force.addPerParticleParameter('Ez')
    ext_force.addPerParticleParameter('q')
    ext_force.addPerParticleParameter('sum')
    
    n_atoms = system.getNumParticles()
    for n in range(n_atoms):
        if n not in qm_atoms:
            ext_force.addParticle(n, [0, 0, 0, charges[n], 0])

    system.addForce(ext_force)

    return ext_force

def add_ext_force_all(system, charges):
    #   adds external force to all atoms
    #   Gx is the gradient of the energy in the x-direction, etc.
    #   sum is used as a constant to set the energy to zero at each step
    ext_force = CustomExternalForce('q*(Gx*x + Gy*y + Gz*z - sum) + qm_energy')
    ext_force.addPerParticleParameter('Gx')
    ext_force.addPerParticleParameter('Gy')
    ext_force.addPerParticleParameter('Gz')
    ext_force.addPerParticleParameter('q')
    ext_force.addPerParticleParameter('sum')
    ext_force.addGlobalParameter('qm_energy', 0.0)
    n_atoms = system.getNumParticles()
    for n in range(n_atoms):
        ext_force.addParticle(n, [0, 0, 0, 0, 0])
    
    system.addForce(ext_force)
    return ext_force

def add_pull_force(coords, system):
    pull_force = CustomExternalForce('px*x + py*y + pz*z')
    pull_force.addPerParticleParameter('px')
    pull_force.addPerParticleParameter('py')
    pull_force.addPerParticleParameter('pz')
    
    direction = coords[98] - coords[95]
    norm = direction / np.linalg.norm(direction)
    force_mag = 2000
    pull_force.addParticle(94, -norm*force_mag)
    pull_force.addParticle(98, -norm*force_mag)
    system.addForce(pull_force)
    return pull_force

def add_rachet_pawl_force(system, pair_file_loc, coords, strength, topology):
    ''' Applies a ratched-and-pawl force that favors restricts
        the approachment of two atoms and is zero if they are
        moving away.
        pair_file_loc is afile with an Nx2 list of atom id's to apply this force to
    '''
    #   import atom pairs from file
    atom_pairs = []
    index_dict = {}
    for atom in topology.atoms():
        index_dict[atom.id] = atom.index
    with open(pair_file_loc, 'r') as file:
        for line in file.readlines():
            sp = line.split()
            p1 = index_dict[int(sp[0])]
            p2 = index_dict[int(sp[1])]
            atom_pairs.append([p1, p2])

    #   create force object
    force_string = 'on_off*0.5*k*(r - r_max)^2; '
    force_string += 'on_off = step(r_max - r); ' #    equals 1 if r < r_max, 0 otherwise
    force_string += 'r = distance(p1, p2); '
    custom_force = CustomCompoundBondForce(2, force_string)
    custom_force.addPerBondParameter('k')
    custom_force.addPerBondParameter('r_max')

    for pair in atom_pairs:

        dist = np.linalg.norm(coords[pair[0]] - coords[pair[1]])
        custom_force.addBond(pair, [strength, dist])

    system.addForce(custom_force)
    return custom_force

def update_rachet_pawl_force(force, context, coords):
    n_bonds = force.getNumBonds()
    for n in range(n_bonds):

        pair, params = force.getBondParameters(n)
        params = list(params)
        dist = np.linalg.norm(coords[pair[0]] - coords[pair[1]])
        params[1] = np.max([params[1], dist])
        force.setBondParameters(n, pair, params)
    force.updateParametersInContext(context)

