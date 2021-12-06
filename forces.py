from numpy.lib.arraysetops import isin
from simtk.openmm.app.topology import Topology
from simtk.openmm.openmm import *
from simtk.unit import * #pylint: disable=unused-wildcard-import
import numpy as np
import os
from openmm_import import *

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

class PointRestraintForce():
    def __init__(self, system, topology, param_file) -> None:
        #raise NotImplementedError("PointRestraintForce is not finished")
        force_string = "0.5*k*(x_on*(x - x0)^2 + y_on*(y - y0)^2 + z_on*(z - z0)^2);"
        force = CustomExternalForce(force_string)
        force.addPerParticleParameter('k')
        force.addPerParticleParameter('x0')
        force.addPerParticleParameter('y0')
        force.addPerParticleParameter('z0')
        force.addPerParticleParameter('x_on')
        force.addPerParticleParameter('y_on')
        force.addPerParticleParameter('z_on')

        used_ids = set()
        with open(param_file, 'r') as file:
            for line in file:
                entry = line.split('!')
                data = entry[0]
                sp = data.split()
                if len(sp) == 0: continue
                if len(sp) != 6:
                    raise SyntaxError('Invalid format for PointRestraintForce')
                id_str = sp[0]
                if ":" in id_str:
                    start, stop = id_str.split(':')
                    id_list = range(int(start), int(stop)+1)
                elif id_str.isdigit():
                    id_list = [int(id_str)]
                else:
                    raise ValueError("Only integers or integer ranges using the ':' character are allowed")
                #id = int(sp[0])
                k = float(sp[1])

                bdry_type = sp[2].lower()
                x_on = y_on = z_on = 0
                if 'x' in bdry_type: x_on = 1
                if 'y' in bdry_type: y_on = 1
                if 'z' in bdry_type: z_on = 1
                x, y, z = [float(x) for x in sp[3:6]]

                index_list = [x.index for x in topology.atoms() if int(x.id) in id_list]
                for index in index_list:
                    force.addParticle(index, (k, x, y, z, x_on, y_on, z_on))

        system.addForce(force)
        self._force = force

    def getForce(self):
        return self._force
                
class PointRestraintForceReporter():
    def __init__(self, force, out_file, report_interval, system):
        #raise NotImplementedError("PointRestraintForceReporter is not finished")
        self._force = force
        self._report_interval = report_interval
        self._system = system
        if isinstance(out_file, str):
            self._out = open(out_file, 'w')
        else:
            self._out = out_file

        self._index = []
        self._points = []
        self._springs = []
        self._bound_types = []
        self._coords_on = []
        for n in range(self._force.getNumParticles()):
            idx, params = self._force.getParticleParameters(n)
            k, x, y, z, x_on, y_on, z_on = params
            bound_type = int(x_on)*'x' + int(y_on)*'y' + int(z_on)*'z'
            self._index.append(idx)
            self._springs.append(k)
            self._points.append(np.array([x, y, z]))
            self._coords_on.append([x_on, y_on, z_on])
            self._bound_types.append(bound_type)

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep%self._report_interval
        return (steps, True, False, False, False)

    def report(self, simulation, state):
        self._out.write('\n')
        self._out.write(' -------------------------------------------------------\n')
        self._out.write('               Point Restraint Forces \n')
        self._out.write(' -------------------------------------------------------\n')
        self._out.write(' {:5s}  {:4s}  {:>14s}  {:>14s}\n'.format('Index', 'type', 'energy(kJ/mol)', 'dist(Ang.)'))
        pos = state.getPositions(True)/nanometers

        for n, point in enumerate(self._points):
            x_on, y_on, z_on = self._coords_on[n]
            x, y, z = pos[n] - point
            dist = sqrt(x_on*(x*x) + y_on*(y*y) + z_on*(z*z))
            eng = 0.5*self._springs[n]*dist**2
            self._out.write(' {:5d}  {:4s}  {:>14.5f}  {:>14.5f}\n'.format(self._index[n], self._bound_types[n], eng, dist*10))
        self._out.write(' -------------------------------------------------------\n')

class RingAngleForce():

    def __init__(self, system, topology, ring_file_loc):

        #   extra particles will be added to the sytem, so we need to add
        #   nonbonded parameters for these particles
        nb_force = None
        for force in system.getForces():
            if isinstance(force, CustomNonbondedForce):
                nb_force = force
                break
        num_nb_params = force.getNumPerParticleParameters()

        with open(ring_file_loc) as file:
            for line in file:
                #   import data from parameter file
                data = line.split("!")[0]
                elms = data.split()
                if len(data) == 0: continue
                ids = [int(x) for x in elms[0:3]]
                spring_k = float(elms[3])
                theta_0 = float(elms[4])
                atom1, atom2, atom3 = [atom.index for atom in topology.atoms() if int(atom.id) in ids]

                #   add another particle to the system for the virtual site
                new_idx = system.addParticle(0)
                nb_force.addParticle(tuple([0]*num_nb_params))

                #   create virtual site above the ring
                origin_wt = Vec3(1/3, 1/3, 1/3)
                x_wt = Vec3(2/3, -1/3, -1/3)
                y_wt = Vec3(-1/3, 2/3, -1/3)
                local_pos = Vec3(0, 0, 1)
                virtual = LocalCoordinatesSite(atom1, atom2, atom3, origin_wt, x_wt, y_wt, local_pos)
                system.setVirtualSite(new_idx, virtual)


                

class CentroidRestraintForce():
    '''
        Adds a restraining force between the centroid of two groups of atoms.

        The input is specified by the $centroids section.
    '''
    def __init__(self, system, topology, restraints_file_loc, pdb=None):
        ''' Add a restraining force between the centroid of two groups of atoms.

            Parameters
            ----------
            system: openmm.app.System
                OpenMM system to add force to
            topology: openmm.app.Topology
                A Topology describing the the system to simulate
            restraints_file_loc: str
                Restraint file with input options 

            Notes
            -----
            Example of $centroids input section:\n
            $centroids
                GROUP
                    15
                ENDGROUP \n
                GROUP
                    16:26 25 27 28 29 30 30
                ENDGROUP \n
                FORCES
                    1 2 9000 0.55
                ENDFORCES
            $end
        '''

        #   extra particles may be added to the system, so we need to add
        #   nonbonded parameters for these particles
        nb_force = None
        for force in system.getForces():
            if isinstance(force, CustomNonbondedForce):
                nb_force = force
                break
        num_nb_params = force.getNumPerParticleParameters()

        from simtk.openmm.openmm import CustomCentroidBondForce
        #force_string = "0.5*k*(distance(g1,g2) - r0)^2"
        force_string = "0.5*k*(r - r0)^2; "
        force_string += "r = sqrt(x_on*((x1 - x2)^2) + y_on*((y1 - y2)^2) + z_on*((z1 - z2)^2));"
        custom_force = CustomCentroidBondForce(2, force_string)
        custom_force.addPerBondParameter('k')
        custom_force.addPerBondParameter('r0')
        custom_force.addPerBondParameter('x_on')
        custom_force.addPerBondParameter('y_on')
        custom_force.addPerBondParameter('z_on')

        #   for bonds with the origin
        force_string2 = "0.5*k*on_off*(r - r0)^2;"
        force_string2 += "on_off = step(r - r0);"
        force_string2 += "r = sqrt(x_on*(x1^2) + y_on*(y1^2) + z_on*(z1^2));"
        boundary_force = CustomCentroidBondForce(1, force_string2)
        boundary_force.addPerBondParameter('k')
        boundary_force.addPerBondParameter('r0')
        boundary_force.addPerBondParameter('x_on')
        boundary_force.addPerBondParameter('y_on')
        boundary_force.addPerBondParameter('z_on')

        #   ring forces
        force_string3 = "0.5*k*(theta - theta0)^2; "
        force_string3 += "theta = acos((z2-z1)/distance(g1, g2));"
        ring_force = CustomCentroidBondForce(2, force_string3)
        ring_force.addPerBondParameter('k')
        ring_force.addPerBondParameter('theta0')

        groups = []
        self.virtual_pos = []
        if pdb is not None:
            v_chain = pdb.topology.addChain()
            v_res = pdb.topology.addResidue('RingV', v_chain)
        top_atoms = list(topology.atoms())
        with open(restraints_file_loc, 'r') as file:
            for line in file:
                #   create a group of atoms to use
                if line.split()[0].lower() == 'group':
                    line = next(file)
                    atom_ids = list()
                    while line.split()[0].lower() != 'endgroup':
                        weights = None
                        for i, elm in enumerate(line.split()):
                            if i == 0 and elm.lower() == 'weights':
                                weights = []
                            elif weights is not None and i > 0:
                                weights.append(int(elm))
                            elif ":" in elm:
                                start, stop = elm.split(':')
                                for i in range(int(start), int(stop) + 2):
                                    atom_ids.append(i)
                            elif elm.isdigit():
                                atom_ids.append(int(elm))
                            else:
                                raise ValueError("Only integers or integer ranges using the ':' character are allowed")

                        line = next(file)
                    indicies = []
                    for id in atom_ids:
                        for atom in top_atoms:
                            if int(atom.id) == id:
                                indicies.append(atom.index)
                                break
                    #indicies = [atom.index for atom in topology.atoms() if int(atom.id) in atom_ids]
                    groups.append(indicies)
                    if weights is not None:
                        custom_force.addGroup(list(set(indicies)), weights)
                        boundary_force.addGroup(list(set(indicies)), weights)
                        ring_force.addGroup(list(indicies), weights)
                    else:
                        custom_force.addGroup(list(set(indicies)))
                        boundary_force.addGroup(list(set(indicies)))
                        ring_force.addGroup(list(indicies))
                #   add the forces between groups
                elif line.split()[0].lower() == 'forces':
                    line = next(file)
                    while line.split()[0].lower() != 'endforces':
                        sp = line.split()
                        g1 = int(sp[0]) - 1
                        g2 = int(sp[1]) - 1
                        k = float(sp[2])
                        r0 = float(sp[3])

                        x_on = y_on = z_on = 0
                        if len(sp) == 4:
                            x_on = y_on = z_on = 1
                        else:
                            bdry_type = sp[4].lower()
                            if 'x' in bdry_type: x_on = 1
                            if 'y' in bdry_type: y_on = 1
                            if 'z' in bdry_type: z_on = 1

                        custom_force.addBond([g1, g2], [k, r0, x_on, y_on, z_on])
                        line = next(file)
                #   add forces from the origin a.k.a. "boundary" forces
                elif line.split()[0].lower() == 'boundary':
                    line = next(file)
                    while line.split()[0].lower() != 'endboundary':
                        sp = line.split()
                        g1 = int(sp[0]) - 1
                        bdry_type = sp[1].lower()
                        k = float(sp[2])
                        r0 = float(sp[3])

                        x_on = y_on = z_on = 0
                        if 'x' in bdry_type: x_on = 1
                        if 'y' in bdry_type: y_on = 1
                        if 'z' in bdry_type: z_on = 1

                        boundary_force.addBond([g1], [k, r0, x_on, y_on, z_on])
                        line = next(file)
                #   add forces that keep rings planar
                elif line.split()[0].lower() == 'ring':
                    line = next(file)
                    while line.split()[0].lower() != 'endring':
                        sp = line.split()
                        g1 = int(sp[0]) - 1
                        spring_k = float(sp[1])
                        theta_0 = float(sp[2])*np.pi/180.0

                        #   check group properties
                        orig_atoms, orig_weights = ring_force.getGroupParameters(g1)
                        if len(orig_atoms) != 3:
                            raise ValueError("Ring groups must only use 3 atoms")
                        if len(orig_weights) == 0:
                            ring_force.setGroupParameters(g1, orig_atoms, (1/3, 1/3, 1/3))
                        elif orig_weights != (1/3, 1/3, 1/3):
                            raise ValueError("Ring groups must use 1/3 for each weight")
                        

                        #   add another particle to the system for the virtual site above the ring
                        new_idx = system.addParticle(0)
                        nb_force.addParticle(tuple([0]*num_nb_params))

                        #   create virtual site above the ring
                        origin_wt = Vec3(1/3, 1/3, 1/3)
                        x_wt = Vec3(2/3, -1/3, -1/3)
                        y_wt = Vec3(-1/3, 2/3, -1/3)
                        local_pos = Vec3(0, 0, 0.2)
                        print("RING GROUP: ", groups[g1])
                        #   the vector normal to the ring follows a clock-wise ordering
                        atom2, atom1, atom3 = groups[g1]
                        virtual = LocalCoordinatesSite(atom1, atom2, atom3, origin_wt, x_wt, y_wt, local_pos)
                        system.setVirtualSite(new_idx, virtual)

                        #   add virtual site to provided PDB
                        if pdb is not None:
                            p1, p2, p3 = Vec3(*tuple(pdb.getPositions()[n]/nanometers for n in [atom1, atom2, atom3]))
                            orig = (p1 + p2 + p3)/3
                            dx = x_wt[0]*p1 + x_wt[1]*p2 + x_wt[2]*p3
                            dy = y_wt[0]*p1 + y_wt[1]*p2 + y_wt[2]*p3
                            dz = np.cross(dx, dy)
                            z_vec = dz/sqrt(np.dot(dz, dz))
                            new_pos = orig + 0.2*z_vec
                            pdb.positions.append(new_pos*nanometers)
                            pdb.topology.addAtom('RV{:d}'.format(ring_force.getNumBonds()), None, v_res)

                        #   create another group that is only the virtual site
                        groups.append([new_idx])
                        ring_force.addGroup([new_idx], [1])

                        ring_force.addBond([g1, len(groups)-1], [spring_k, theta_0])
                        line = next(file)
                else:
                    raise ValueError('Invalid section identifier for CentroidRestraintForce')


        #   only add force to system if bonds were created
        self._force = custom_force
        self._bound_force = boundary_force
        self._ring_force = ring_force
        if custom_force.getNumBonds() > 0:
            system.addForce(custom_force)
            
        if boundary_force.getNumBonds() > 0:
            system.addForce(boundary_force)

        if ring_force.getNumBonds() > 0:
            system.addForce(ring_force)

    def getForces(self):
        return (self._force, self._bound_force, self._ring_force)

class CentroidRestraintForceReporter():
    def __init__(self, forces, file, report_interval, system) -> None:
        self._force1, self._bound_force, self._ring_force = forces
        self._report_interval = report_interval
        self._system = system

        if isinstance(file, str):
            self._out = open(file, 'w')
        else:
            self._out = file

        self._groups = []
        self._weights = []
        self._bonds = []
        self._params = []
        self._sum_weights = []
        self._force_types = []

        for g in range(self._ring_force.getNumGroups()):
            idx, weights = self._ring_force.getGroupParameters(g)
            if len(weights) == 0:
                weights = [system.getParticleMass(n)/dalton for n in idx]
            self._groups.append(np.array(idx))
            self._weights.append(np.array(weights))
            self._sum_weights.append(np.sum(weights))

        for g in range(self._force1.getNumBonds()):
            groups, params = self._force1.getBondParameters(g)
            self._bonds.append(groups)
            self._params.append(params)
            self._force_types.append('normal')

        for g in range(self._bound_force.getNumBonds()):
            groups, params = self._bound_force.getBondParameters(g)
            self._bonds.append(groups)
            self._params.append(params)
            self._force_types.append('boundary')

        for g in range(self._ring_force.getNumBonds()):
            groups, params = self._ring_force.getBondParameters(g)
            self._bonds.append(groups)
            self._params.append(params)
            self._force_types.append('ring')

        self._n_bonds = len(self._bonds)

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep%self._report_interval
        return (steps, True, False, False, False)

    def report(self, simulation, state):
        self._out.write('\n')
        self._out.write(' -------------------------------------------------------\n')
        self._out.write('             Centroid Restraint Forces \n')
        self._out.write(' -------------------------------------------------------\n')
        self._out.write(' {:8s}  {:6s}  {:6s}  {:>14s}  {:>10s}   {:>10s}  \n'.format('type', 'group1', 'group2', 'energy(kJ/mol)', 'X', 'X0'))
        pos = state.getPositions(True)/nanometers
        for n in range(self._n_bonds):
            if self._force_types[n] == 'normal':
                g1, g2 = self._bonds[n]

                pos1 = pos[self._groups[g1]]
                pos2 = pos[self._groups[g2]]
                x1, y1, z1 = np.sum(pos1*self._weights[g1][:, None], axis=0)/self._sum_weights[g1]
                x2, y2, z2 = np.sum(pos2*self._weights[g2][:, None], axis=0)/self._sum_weights[g2]

                k, r0, x_on, y_on, z_on = self._params[n]
                dist = sqrt(x_on*((x1 - x2)**2) + y_on*((y1 - y2)**2) + z_on*((z1 - z2)**2))

                energy = 0.5*k*(dist - r0)**2
                self._out.write(' {:8s}  {:6d} {:6d} {:14.3f}  {:10.3f} Ang. {:10.3f} Ang. \n'.format('normal', g1+1, g2+1, energy, dist*10, r0*10))
            elif self._force_types[n] == 'boundary':
                g1 = self._bonds[n][0]

                pos1 = pos[self._groups[g1]]
                x, y, z = np.sum(pos1*self._weights[g1][:, None], axis=0)/self._sum_weights[g1]                
                k, r0, x_on, y_on, z_on = self._params[n]
                dist = sqrt(x_on*(x*x) + y_on*(y*y) + z_on*(z*z))
                bound_type = int(x_on)*'x' + int(y_on)*'y' + int(z_on)*'z'

                if (dist - r0) < 0:
                    energy = 0.0
                else:
                    energy = 0.5*k*(dist - r0)**2
                self._out.write(' {:8s}  {:6d} {:>6s} {:14.3f}  {:10.3f} Ang. {:10.3f} Ang. \n'.format('boundary', g1+1, bound_type, energy, dist*10, r0*10))
            else:
                g1, g2 = self._bonds[n]

                pos1 = pos[self._groups[g1]]
                pos2 = pos[self._groups[g2]]
                x1, y1, z1 = np.sum(pos1*self._weights[g1][:, None], axis=0)/self._sum_weights[g1]
                x2, y2, z2 = np.sum(pos2*self._weights[g2][:, None], axis=0)/self._sum_weights[g2]

                k, theta_0 = self._params[n]
                dist = np.linalg.norm(np.array([x2, y2, z2]) - np.array([x1, y1, z1]))
                theta = np.arccos((z2 - z1)/dist)

                energy = 0.5*k*(theta - theta_0)**2
                self._out.write(' {:8s}  {:6d} {:6d} {:14.3f}  {:10.3f} Deg. {:10.3f} Deg. \n'.format('ring', g1+1, g2+1, energy, theta*180/np.pi, theta_0*180/np.pi))

        self._out.write(' -------------------------------------------------------\n')
        self._out.flush()


class RestraintsForce():
    '''
        Adds a restraing quadratic cost potential between two atoms.
        Switch_time is the time in femtoseconds for the potential to reach
        full magnitude and increases from zero over a sine function.
    '''

    def __init__(self, system, topology, restraints_file_loc, switch_time):
        if os.path.isfile(restraints_file_loc):
            force_string = 'scale*step(equality*(r - r0))*0.5*k*(r - r0)^2'
            custom_force = CustomBondForce(force_string)
            custom_force.addPerBondParameter('k')
            custom_force.addPerBondParameter('r0')
            custom_force.addPerBondParameter('scale')
            custom_force.addPerBondParameter('equality')

            self._p1 = []
            self._p2 = []
            self._k = []
            self._r0 = []
            self._equality = []
            self.switch_time = switch_time


            idx = {}
            for atom in topology.atoms():
                idx[int(atom.id)] = atom.index

            with open(restraints_file_loc, 'r') as file:
                for n, line in enumerate(file.readlines()):
                    sp = line.split()
                    if len(sp) == 0: continue
                    if sp[0] == '!' or sp[0] == '#': continue # comment lines
                    if len(sp) >= 5:
                        self._p1.append(idx[int(sp[0])])
                        self._p2.append(idx[int(sp[1])])
                        self._k.append(float(sp[2]))
                        self._r0.append(float(sp[3]))
                        self._equality.append(-float(sp[4]))

                        #   removes -0 as an option, not sure how if it would work with OpenMM
                        if self._equality[n] == 0.0:
                            self._equality[n] = 0.0

                    else:
                        raise ValueError("Invalid number of elements in restraints file line %d" % n)

            for n in range(len(self._p1)):
                custom_force.addBond(self._p1[n], self._p2[n], [self._k[n], self._r0[n], 1, self._equality[n]])

            system.addForce(custom_force)

            self.force_obj = custom_force
            self._total_time = 0 * femtoseconds
            self.system = system

        else:
            raise FileNotFoundError("Restraints file not found")
        
    def _step(self, x):
        if x < 0:
            return 0.0
        else:
            return 1.0

    def update(self, simulation, pos, outfile=sys.stdout):
        self._total_time += simulation.integrator.getStepSize()
        if self._total_time < self.switch_time:

            switch = np.sin(self._total_time/self.switch_time * np.pi/2)**2

            print(" Updating restraints force with switch value {:8.5f} at time {:6.2f} fs".format(switch, self._total_time/femtoseconds), file=outfile)

            for n in range(len(self._p1)):
                params = [self._k[n], self._r0[n], switch, self._equality[n]]
                self.force_obj.setBondParameters(n, self._p1[n], self._p2[n], params)
            self.force_obj.updateParametersInContext(simulation.context)
        else:
            switch = 1.0

        total_energy = 0.0
        for n in range(len(self._p1)):
            p1 = pos[self._p1[n]]
            p2 = pos[self._p2[n]]
            r = np.linalg.norm(p1 - p2)
            k, r0, eq = [self._k[n], self._r0[n], self._equality[n]]
            total_energy += switch*self._step(eq*(r - r0))*0.5*k*(r - r0)**2

        print(" Total restraint energy: {:15.5f} kJ/mol".format(total_energy), file=outfile)

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
        
        force_string = 'on_off*scale*(r - max_dist)^2; '
        force_string += 'on_off = step(r - max_dist); '
        force_string += 'r = sqrt((x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2)'
        force_obj = CustomExternalForce(force_string)
        force_obj.addGlobalParameter('scale', scale)
        force_obj.addGlobalParameter('max_dist', max_dist/nanometers)
        force_obj.addPerParticleParameter('x_0')
        force_obj.addPerParticleParameter('y_0')
        force_obj.addPerParticleParameter('z_0')
        self.oxygen_idx = []
        self.oxygen_atoms = []
        self.material_idx = []
        self.material_atoms = []
        self.scale = scale
        self.max_dist = max_dist/nanometers
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
            force_obj.addParticle(oxy, [x, y, z])

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

            
            self.force_obj.setParticleParameters(n, idx, [x, y, z])

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


                if n in qm_atoms:
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
    atoms = list(topology.atoms())
    res_names = [x.residue.name for x in atoms]
    

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
                    if dist > r/nanometer*1.3 and "HOH" in [res_names[a], res_names[b]]:
                        new_atoms.add(a)
                        new_atoms.add(b)
                    else:
                        force.setBondParameters(n, a, b, r, 462750.4*kilojoules_per_mole/nanometer**2)
            force.updateParametersInContext(context)

    #   also check that any QM atom is not to close to an MM atom
    #   if so, add the MM atoms back to the list
    for atom in atoms:
        if atom.index in qm_atoms or atom.index in new_atoms:
            this_coord = coords[atom.index]
            distances = np.linalg.norm(this_coord- coords, axis=1)
            for n in range(len(coords)):
                if n == atom.index: continue
                if n in new_atoms: continue
                if n in qm_atoms: continue
                if distances[n] < 0.13:
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
                #   exception for hydrogens from waters, these are known to get
                #   too close to link atoms, so we use a small vdW radii
                if atoms[n].residue.name == 'HOH' and atoms[n].element.symbol == 'H':
                    if n in new_qm_atoms:
                        params[1] = 0.10    # sigma
                        params[2] = 0.20    # epsilon
                    else:
                        #   original tip3p parameters
                        params[1] = 1.0
                        params[2] = 0.0
                force.setParticleParameters(n, params)
            force.updateParametersInContext(context)


    #   re-update bonds and angles with new QM list
    if bond_force:
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

    #   print out the newly added QM atoms:
    for new_atom in new_qm_atoms:
        if new_atom not in qm_atoms:
            print(" Added atom id {:d} to qm_atoms list".format(int(atoms[new_atom].id)), file=outfile)


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



