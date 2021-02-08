from simtk.openmm.openmm import *  #pylint: disable=unused-wildcard-import
from simtk.unit import * #pylint: disable=unused-wildcard-import
import numpy as np
import os

# pylint: disable=no-member
import simtk.unit as unit
picosecond = picoseconds = unit.picosecond
nanometer = nanometers = unit.nanometer
femtoseconds = unit.femtoseconds
# pylint: enable=no-member

def add_oxygen_repulsion(system, topology, boundary=0.25):
    force_string = '4*epsilon*(sigma/r)^12; '
    force_string += 'sigma = (sigma1 + sigma2)*0.5; '
    force_string += 'epsilon = sqrt(epsilon1*epsilon2)'
    customForce = CustomNonbondedForce(force_string)
    customForce.addPerParticleParameter("sigma")
    customForce.addPerParticleParameter("epsilon")

    #   add only oxygen atoms that are not waters
    group = []
    res_ids = []    
    for atoms in topology.atoms():
        if atoms.element.symbol == 'O' and atoms.residue.name != 'HOH':
            group.append(atoms.index)
            res_ids.append(int(atoms.residue.id))
            customForce.addParticle([boundary, 1.0])
        else:
            customForce.addParticle([boundary, 0.0])

    #   exclude oxygens that are of the same residue
    exclusions = []        
    for i, atom1 in enumerate(group):
        for j, atom2 in enumerate(group):
            if res_ids[i] == res_ids[j] and i < j:
                exclusions.append((atom1, atom2))

    customForce.addInteractionGroup(group, group)
    for excl in exclusions:
        print("EXCL: ", excl)
        customForce.addExclusion(*excl)

    system.addForce(customForce)

class RandomKicksForce():
    def __init__(self, simulation, topology, temperature, scale=0.2):
        kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA/1000 #Boltzmann constant in kJ/mole/K
        self.kB = kB/kB.unit
        self._kB_m = self.kB/np.array([x.element.mass/dalton for x in topology.atoms()])
        self._simulation = simulation
        self.temperature = temperature/kelvin
        self.scale = scale

    def update(self, temperature=None, outfile=None):
        temp = self.temperature
        if temperature:
            temp = temperature/kelvin
        
        state = self._simulation.context.getState(getVelocities=True)
        velocities = state.getVelocities(True)

        rands = np.random.normal(size=(len(velocities), 3)) 
        kicks = np.sqrt(self._kB_m * temp)[:, None] * rands* self.scale
        new_vel = velocities + kicks*(nanometers/picosecond)

        self._simulation.context.setVelocities(new_vel)

        if outfile:
            max_vel = np.max(np.linalg.norm(kicks, axis=1))
            max_idx = np.argmax(np.linalg.norm(kicks, axis=1))
            kT_m = self._kB_m[max_idx] * temp
            rand_norm = np.linalg.norm(rands[max_idx])
            print(max_idx, np.argmax(np.linalg.norm(rands, axis=1)))
            print(self.kB/self._kB_m[max_idx])
            print(kicks[max_idx])
            print(rands[max_idx])
            print(self.scale)
            print(" Applying random thermal kicks", file=outfile)
            print(" Largest velocity kick: {:10.5f} with kT/m {:10.5f} and rand_f {:10.5f}".format(max_vel, kT_m, rand_norm), file=outfile)


    def set_temperature(self, temperature):
        self.temperature = temperature

class HugsForce():
    def __init__(self, system, topology, hugs_file_loc, time_step, switch_time):
        if os.path.isfile(hugs_file_loc):
            force_string = 'scale*0.5*k*(r - r0)^2'
            custom_force = CustomBondForce(force_string)
            custom_force.addPerBondParameter('k')
            custom_force.addPerBondParameter('r0')
            custom_force.addPerBondParameter('scale')

            self._p1 = []
            self._p2 = []
            self._k = []
            self._r0 = []
            self.switch_time = switch_time


            idx = {}
            for atom in topology.atoms():
                idx[int(atom.id)] = atom.index

            with open(hugs_file_loc, 'r') as file:
                for n, line in enumerate(file.readlines()):
                    sp = line.split()
                    if len(sp) == 0: continue
                    if len(sp) >= 4:
                        self._p1.append(idx[int(sp[0])])
                        self._p2.append(idx[int(sp[1])])
                        self._k.append(float(sp[2]))
                        self._r0.append(float(sp[3]))


                    else:
                        raise ValueError("Invalid number of elements in hugs file line %d" % n)

            for n in range(len(self._p1)):
                custom_force.addBond(self._p1[n], self._p2[n], [self._k[n], self._r0[n], 1])

            system.addForce(custom_force)

            self.force_obj = custom_force
            self.time_step = time_step
            self._total_time = 0 * femtoseconds

        else:
            raise FileNotFoundError("Hugs force file not found")

    def update(self, context, outfile=sys.stdout):
        self._total_time += self.time_step
        if self._total_time < self.switch_time:

            switch = np.sin(self._total_time/self.switch_time * np.pi/2)**2

            print("Updating hugs force: surrent switch vale: {:8.5f}".format(switch), file=outfile)

            for n in range(len(self._p1)):
                params = [self._k[n], self._r0[n], switch]
                self.force_obj.setBondParameters(n, self._p1[n], self._p2[n], params)
            self.force_obj.updateParametersInContext(context)

class BoundryForce():
    def __init__(self, system, topology, positions, qm_atoms, max_dist=0.4*nanometers, scale=10000.0):
        '''
            Adds a force that keeps oxygen molecules within a spicific distance
            from at least one material atom. Applies only to QM atoms

            Parameters:
                system: openmm.system
                    openmm system object for the simulation
                topology: openmm.topology
                    openmm topology object for all atoms in system
                positions : list, np.array
                    array of all atomic positions (unit must be included)
                qm_atoms:   list
                    list of QM atom indicies
                max_dist: Quantity
                    maximum distance oxygen atoms can be away from the material
                scale: float
                    strength of the force to be applied (kJ/mol/nanometers^2)
        '''
        
        force_string = 'on_off*scale*r^2; '
        force_string += 'on_off = step(r - max_dist); '
        force_string += 'r = sqrt((x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2)'
        force_obj = CustomExternalForce(force_string)
        force_obj.addPerParticleParameter('scale')
        force_obj.addPerParticleParameter('max_dist')
        force_obj.addPerParticleParameter('x_0')
        force_obj.addPerParticleParameter('y_0')
        force_obj.addPerParticleParameter('z_0')
        self.oxygen_idx = []
        self.oxygen_atoms = []
        self.material_idx = []
        self.material_atoms = []
        self.scale = scale
        self.max_dist = max_dist
        self.topology = topology

        #   extract atoms used in this force
        for atom in topology.atoms():
            if atom.index in qm_atoms:
                if atom.element.symbol == 'O' and atom.residue.name != 'HOH':
                    self.oxygen_idx.append(atom.index)
                    self.oxygen_atoms.append(atom)
                elif atom.element.symbol in ['Se', 'Zn']:
                    self.material_idx.append(atom.index)
                    self.material_atoms.append(atom)

        material_pos = np.array([positions[x]/nanometers for x in self.material_idx])*nanometers
        for oxy in self.oxygen_idx:
            distances = np.linalg.norm(positions[oxy] - material_pos, axis=1)
            min_idx = np.argmin(distances)
            x, y, z = material_pos[min_idx]
            force_obj.addParticle(oxy, [scale, max_dist/nanometers, x, y, z])

        system.addForce(force_obj)
        self.force_obj = force_obj

    def update(self, context, positions, outfile=None):
        '''
        Update the force field parameters for the force object.
        Parameters
        ----------
        context : openmm.context
            OpenMM context object that contains the forces to update
        positions : list, np.array
            array of all atomic positions (unit must be included)
        outfile: output stream
            if supplied, a summary of the forces will be printed to this stream
            for example, file, std.out, etc

        Returns
        -------
        Nothing
        '''

        if outfile:
            print(' ------------------------------------------------------------ ', file=outfile)
            print('                     Updating Boundry Force', file=outfile)
            print('      Oxygen Atom            Nearest Material Atom   Dist (Ang) ', file=outfile)

        material_pos = np.array([positions[x]/nanometers for x in self.material_idx])*nanometers
        for n, oxy in enumerate(self.oxygen_idx):
            idx = self.force_obj.getParticleParameters(n)[0]
            distances = np.linalg.norm(positions[oxy] - material_pos, axis=1)
            min_idx = np.argmin(distances)
            x, y, z = material_pos[min_idx]

            if outfile:
                oxy = self.oxygen_atoms[n]
                mat = self.material_atoms[min_idx]
                min_dist = distances.min() *10
                print(' {:4s} id {:5s} res {:5s} | {:4s} id {:5s} res {:5s} | {:8.3f} '. \
                    format(oxy.name, oxy.id, oxy.residue.id, mat.name, mat.id, mat.residue.id, min_dist), file=outfile)
            
            self.force_obj.setParticleParameters(n, idx, [self.scale, self.max_dist, x, y, z])

        if outfile:
            print(' ------------------------------------------------------------ \n', file=outfile)

        self.force_obj.updateParametersInContext(context)
        

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
    ext_force = CustomExternalForce('q*(Gx*x + Gy*y + Gz*z - sum) + 0*qm_energy')
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

def add_nonbonded_force(qm_atoms, system, bonds, outfile=sys.stdout):
    forces = system.getForces()
    forceString = "lj_on*4*epsilon*((sigma/r)^12 - (sigma/r)^6) + coul_on*138.935458 * q/r; "
    forceString += "sigma=0.5*(sigma1+sigma2); "
    forceString += "epsilon=sqrt(epsilon1*epsilon2); "
    forceString += "q=q1*q2; "
    forceString += "lj_on=1 - min(is_qm1, is_qm2); "
    forceString += "coul_on=1 - max(is_qm1, is_qm2); "
    customForce = CustomNonbondedForce(forceString)
    customForce.addPerParticleParameter("q")
    customForce.addPerParticleParameter("sigma")
    customForce.addPerParticleParameter("epsilon")
    customForce.addPerParticleParameter("is_qm")

    #   get list of bonds for exclusions
    bond_idx_list = []
    for bond in bonds:
        bond_idx_list.append([bond.atom1.index, bond.atom2.index])
    
    #   add the same parameters as in the original force
    #   but separate based on qm - mm systems
    charges = []
    qm_charges = []
    mm_atoms = []
    for i, force in enumerate(forces):
        if isinstance(force, NonbondedForce):
            print(" Adding custom non-bonded force")
            for n in range(force.getNumParticles()):
                chg, sig, eps = force.getParticleParameters(n)
                charges.append(chg / elementary_charge)

                #if n not in [952, 1329]:
                #    chg *= 0
                #    eps *= 0

                if n in qm_atoms:
                    if n == 414:
                        print("414: ", chg, sig, eps)
                    qm_charges.append(chg / elementary_charge)
                    customForce.addParticle([chg, sig, eps, 1])
                else:
                    mm_atoms.append(n)
                    customForce.addParticle([chg, sig, eps, 0])
                

            system.removeForce(i)
            #customForce.addInteractionGroup(qm_atoms, mm_atoms)
            #customForce.addInteractionGroup(mm_atoms, mm_atoms)
            customForce.createExclusionsFromBonds(bond_idx_list, 2)
            system.addForce(customForce)

    total_chg = np.sum(charges)
    total_qm_chg = np.sum(qm_charges)
    total_mm_chg = total_chg - total_qm_chg
    print("", file=outfile)
    print(" Force field charge distributions:", file=outfile)
    print(" Total charge:    %.4f e" % round(total_chg, 4), file=outfile)
    print(" Total MM charge: %.4f e" % round(total_mm_chg, 4), file=outfile)
    print(" Total QM charge: %.4f e" % round(total_qm_chg, 4), file=outfile)
    print("", file=outfile)
    print(" Number of atoms: {:d}".format(len(charges)), file=outfile)
    print("", file=outfile)
    return charges

def update_mm_forces(qm_atoms, system, context, coords, topology, outfile=sys.stdout):
    """
    Adjust forces for MM atoms.
    If bonds have stretched too far, it leaves them as
    QM atoms and returns a new list of qm_atoms
    """

    if not is_quantity(coords):
        raise ValueError('coords must have units')
    coords = coords/nanometers

    new_atoms = set()
    bond_force = None
    for force in system.getForces():
        #   remove bond from dynamic qm_atoms pairs
        if isinstance(force, HarmonicBondForce):
            bond_force = force
            for n in range(force.getNumBonds()):
                a, b, r, k = force.getBondParameters(n)
                if a in qm_atoms and b in qm_atoms:
                    force.setBondParameters(n, a, b, r, k*0.000)
                else:
                    dist = np.linalg.norm(coords[a] - coords[b])
                    #   k=0 identifies that the bond was a QM bond.
                    #   if while QM, the two atoms have stretched
                    #   too far, leave as QM atoms
                    if dist > r/nanometer*1.3:
                        new_atoms.add(a)
                        new_atoms.add(b)
                    else:
                        force.setBondParameters(n, a, b, r, 462750.4*kilojoules_per_mole/nanometer**2)
            force.updateParametersInContext(context)

    #   also check that any QM atom is not to close to an MM atom
    #   if so, add the MM atoms back to the list
    for atom in topology.atoms():
        if atom.index in qm_atoms or atom.index in new_atoms:
            this_coord = coords[atom.index]
            distances = np.linalg.norm(this_coord- coords, axis=1)
            for n in range(len(coords)):
                if n == atom.index: continue
                if n in new_atoms: continue
                if n in qm_atoms: continue
                if distances[n] < 0.13:
                    print("FOUND: ", atom.id, list(topology.atoms())[n].id)
                    print()
                    new_atoms.add(n)

    #   for all atoms in the new QM list, add all other atoms in their residues
    #   to the QM list.
    for res in topology.residues():
        add_to_list = False
        for atom in res.atoms():
            if atom.index in new_atoms:
                add_to_list = True
                break
        if add_to_list:
            for atom in res.atoms():
                new_atoms.add(atom.index)

    #   now that the new atoms are added, continue with nonbonded force
    new_qm_atoms = sorted(list(set(list(new_atoms) + qm_atoms)))
    for force in system.getForces():
        #   non-bonded force has built in parameters to turn on/off terms
        if isinstance(force, CustomNonbondedForce):
            for n in range(force.getNumParticles()):
                params = list(force.getParticleParameters(n))
                if n in new_qm_atoms:
                    params[-1] = 1
                else:
                    params[-1] = 0
                force.setParticleParameters(n, params)
            force.updateParametersInContext(context)


    #   re-update bonds and angles with new QM list
    for n in range(bond_force.getNumBonds()):
        a, b, r, k = bond_force.getBondParameters(n)
        if a in new_qm_atoms and b in new_qm_atoms:
            bond_force.setBondParameters(n, a, b, r, k*0.000)
        else:
            bond_force.setBondParameters(n, a, b, r, 462750.4*kilojoules_per_mole/nanometer**2)
    bond_force.updateParametersInContext(context)

    for force in system.getForces():
        if isinstance(force, HarmonicAngleForce):
            for n in range(force.getNumAngles()):
                a, b, c, t, k = force.getAngleParameters(n)
                in_qm_atoms = [x in new_qm_atoms for x in [a, b, c]]
                num_qm_atoms = np.sum(in_qm_atoms)
                if num_qm_atoms > 0:
                    force.setAngleParameters(n, a, b, c, t, k*0.000)
                else:
                    force.setAngleParameters(n, a, b, c, t, 836.8*kilojoules_per_mole)
            force.updateParametersInContext(context)


    return new_qm_atoms

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

def add_rachet_pawl_force(system, pair_file_loc, coords, strength, topology, half_dist=1000.0, switch_type='exp'):
    ''' Applies a ratched-and-pawl force that favors restricts
        the approachment of two atoms and is zero if they are
        moving away.

        pair_file_loc:  file with an Nx2 list of atom id's to apply this force to
        coords:         coordinates of all atoms in nanometers
        strength:       scales the ratchet-pawl energy
        topology:       Topology object of molecule
        half_dist:      distance in nanometers from the initial bond distance
                        at which the RP energy is halved. The energy continues 
                        to be decease exponentially from the initial distance.
    '''
    #   import atom pairs from file
    atom_pairs = []
    half_dists =[]
    strengths = []
    index_dict = {}
    for atom in topology.atoms():
        index_dict[int(atom.id)] = atom.index
    with open(pair_file_loc, 'r') as file:
        for line in file.readlines():
            sp = line.split()
            if len(sp) == 0:
                continue
            p1 = index_dict[int(sp[0])]
            p2 = index_dict[int(sp[1])]
            atom_pairs.append([p1, p2])
            if len(sp) == 4:
                strengths.append(float(sp[2]))
                half_dists.append(float(sp[3]))
            else:
                strengths.append(strength)
                half_dists.append(half_dist)

    #   create force object
    force_string = 'on_off*0.5*k*switch*(r - r_max)^2; '
    if switch_type == 'exp':
        force_string += 'switch=exp(-a*(r - r_0)); '
    elif switch_type == 'tanh':
        force_string += 'switch=(1 - tanh((r - r_0)*a))*0.5; '
    else:
        raise ValueError("Invalid switching type; must be 'exp' or 'tanh'")
    force_string += 'on_off = step(r_max - r); ' #    equals 1 if r < r_max, 0 otherwise
    force_string += 'r = distance(p1, p2); '
    custom_force = CustomCompoundBondForce(2, force_string)
    custom_force.addPerBondParameter('k')
    custom_force.addPerBondParameter('r_max')
    custom_force.addPerBondParameter('r_0')
    custom_force.addPerBondParameter('a')
    custom_force.addPerBondParameter('s_type')

    
    for n, pair in enumerate(atom_pairs):
        dist = np.linalg.norm(coords[pair[0]] - coords[pair[1]])
        if switch_type == 'exp':
            a = -np.log(0.5)/half_dists[n]
            r_0 = dist
            s_type = 0
        elif switch_type == 'tanh':
            a = 50.0
            r_0 = dist + half_dist
            s_type = 1
        custom_force.addBond(pair, [strengths[n], dist, r_0, a, s_type])

    system.addForce(custom_force)
    return custom_force

def update_rachet_pawl_force(force, context, coords, outfile=None):

    if outfile:
        print(" Updating Ratchet-Pawl Force", file=outfile)
        print(" \t idx1  idx2  r_max      curr_k         k", file=outfile)

    n_bonds = force.getNumBonds()
    for n in range(n_bonds):

        pair, params = force.getBondParameters(n)
        params = list(params)
        dist = np.linalg.norm(coords[pair[0]] - coords[pair[1]])
        params[1] = np.max([params[1], dist])
        force.setBondParameters(n, pair, params)

        if outfile:
            k = params[0]
            r = dist
            r_0 = params[2]
            a = params[3]
            s_type = params[4]
            if s_type == 0:
                curr_strength = k*np.exp(-a*(r - r_0))
            elif s_type == 1:
                curr_strength = k*(1 - tanh((r - r_0)*a))*0.5
            print( "\t{:4d}  {:4d}  {:8.5f}  {:10.1f}  {:10.1f}"\
                .format(pair[0], pair[1], params[1], curr_strength, k), file=outfile)


    force.updateParametersInContext(context)



