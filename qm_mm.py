from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm import * #pylint: disable=unused-wildcard-import
from simtk.openmm.openmm import * 
from simtk.unit import * #pylint: disable=unused-wildcard-import
import argparse
import numpy as np
from numpy.linalg import norm
import sys
from multiprocessing import cpu_count
from subprocess import run
import time
import shutil
from scipy import optimize
from optimize import GradientMethod, BFGS, MMOnlyBFGS
from mdtraj.reporters import HDF5Reporter
from distutils.util import strtobool
from random import shuffle
from cmd_line_args import parse_cmd_line_args
from qm_fragments import QM_Fragments
from qchem import QChemRunner

# pylint: disable=no-member
import simtk.unit as unit
picosecond = picoseconds = unit.picosecond
nanometer = nanometers = unit.nanometer
femtoseconds = unit.femtoseconds
# pylint: enable=no-member

sys.path.insert(1, "/network/rit/lab/ChenRNALab/bin/Pymol2.3.2/pymol/lib/python3.7/site-packages")
#from pymol import cmd, stored # pylint: disable=import-error
import pdb_to_qc
from sim_extras import *
from forces import *
from spin_mult import *
from add_solvent_sphere import WaterFiller


scratch = os.path.join(os.path.curdir, 'qm_mm_scratch/')
qchem_path = ''
qc_scratch = '/tmp'
qm_fragments = None

#   set available CPU resources
if 'SLURM_NTASKS' in os.environ.keys():
    n_procs = int(os.environ['SLURM_NTASKS'])
else:
    #   if not running a slurm job, use number of cores
    n_procs = cpu_count()
print("SLURM: ", 'SLURM_NTASKS' in os.environ.keys(), n_procs)


def parse_args(args_in):
    parser = argparse.ArgumentParser('')
    parser.add_argument('-pdb',   required=True, help='pdb molecule file to use')
    parser.add_argument('-rem',   required=False, help='rem arguments to use with Q-Chem')
    parser.add_argument('-idx',   required=False, help='list of atoms to treat as QM')
    parser.add_argument('-out',   help='output file', default='output.txt')
    parser.add_argument('-pos',   help='set positions using this pdb file')
    parser.add_argument('-state', help='start simulation from simulation state xml file')
    parser.add_argument('-repf',  help='file to print forces to')
    parser.add_argument('-repv',  help='file to print velocities to')
    parser.add_argument('-pawl',  help='list of atom ID pairs to apply ratchet-pawl force between')
    parser.add_argument('-rest',  help='list of atom ID pairs to apply restraint force between')
    parser.add_argument('-nt',    help='number of threads to use in Q-Chem calculations', type=int)
    parser.add_argument('-freeze', help='file of atom indicies to freeze coordinates')
    parser.add_argument('-ipt',   help='condensed input file')
    parser.add_argument('-frags',   help='QM fragments file')
    parser.add_argument('-link',   help='QM/MM link atom ids file')
    parser.add_argument('-centroid', help='centroid restraint force file')
    return parser.parse_args(args_in)


def parse_idx(idx_file_loc, topology):
    qm_fixed_atoms = []
    qm_origin_atoms = []
    with open(idx_file_loc, 'r') as file:
        for line in file.readlines():
            sp = line.split()
            #   assume that just a column of numbers is used
            if len(sp) == 1:
                if '*' in line: 
                    num = list(filter(str.isdigit, sp[0]))
                    idx = ''.join(num)
                    qm_origin_atoms.append(int(idx))
                else: 
                    qm_fixed_atoms.append(int(sp[0]))
            #   assume that the output from pymol is used
            elif len(sp) == 3 and "cmd.identify" in line:
                if '*' in line:
                    num = list(filter(str.isdigit, line))
                    idx = ''.join(num)
                    qm_origin_atoms.append(int(idx))
                else:
                    idx = sp[-1].split('`')[-1].split(')')[0]
                    qm_fixed_atoms.append(int(idx))
            else:
                print("ERROR: Can't determin index format")
    qm_fixed_atoms = sorted(qm_fixed_atoms + qm_origin_atoms)
    qm_origin_atoms = sorted(qm_origin_atoms)

    qm_fixed_atoms_indices = []
    qm_origin_atoms_indices = []
    for atom in topology.atoms():
        if int(atom.id) in qm_fixed_atoms:
            qm_fixed_atoms_indices.append(atom.index)
    for atom in topology.atoms():
        if int(atom.id) in qm_origin_atoms:
            qm_origin_atoms_indices.append(atom.index)
    
    qm_fixed_atoms_indices = sorted(qm_fixed_atoms_indices)
    qm_origin_atoms_indices = sorted(qm_origin_atoms_indices)

    return (qm_fixed_atoms_indices, qm_origin_atoms_indices)

def get_qm_spheres(originAtoms, qm_atoms, radius_in_ang, xyz_in_ang, topology):          

    '''Finds all atoms within a given radius of each atom in 
       originAtoms to treat as QM and returns a list of atom indices.'''

    qmSpheres = []
    resList = []
    for i in originAtoms:
        for residue in list(topology.residues()):
                if residue.name != 'HOH': continue 
                if residue.id in resList: continue
                isQuantum = None
                for atom in list(residue.atoms()):
                    if atom.index in qm_atoms: continue
                    if atom.index in originAtoms: continue
                    dist = np.linalg.norm(xyz_in_ang[int(i)] - xyz_in_ang[atom.index])
                    if dist < radius_in_ang:
                        isQuantum = True
   
                        break
                if isQuantum:
                    for atom in list(residue.atoms()):
                    	qmSpheres.append(atom.index)
                    resList.append(residue.id)
    return list(sorted(set(qmSpheres)))
    
def find_all_qm_atoms(mat_idx_list, bondedToAtom, topology):
    qm_idx_list = []
    atoms = list(topology.atoms())
    for idx in mat_idx_list:
        qm_idx_list.append(idx)
        for bonded_to in bondedToAtom[idx]:
            if atoms[bonded_to].element.symbol not in ['Se', 'Zn']:
                pass

def fix_qm_mm_bonds(system, qm_atoms, pos, outfile=sys.stdout):

    print(" Adding QM/MM Bond constraints ", file=outfile)
    print(" {:>10s}  {:>10s}  {:10s} ".format('Index 1', 'Index 2', 'R (Ang.)'), file=outfile)
    num_constr = 0
    for force in system.getForces():
        if  isinstance(force, HarmonicBondForce):
            for n in range(force.getNumBonds()):
                a, b, r, k = force.getBondParameters(n)
                if (a in qm_atoms and b not in qm_atoms) or \
                   (b in qm_atoms and a not in qm_atoms):
                   dist = np.linalg.norm(pos[a]/nanometers - pos[b]/nanometers)*nanometers
                   system.addConstraint(a, b, dist)
                   print(" {:10d}  {:10d}  {:>6.3f} ".format(a, b, dist/angstroms), file=outfile)
                   num_constr += 1
    if num_constr == 0:
        print(" None", file=outfile)
    print("\n", file=outfile)

def adjust_forces(system, context, topology, qm_atoms, outfile=sys.stdout):
    #   set the force constants for atoms included
    #   in QM portion to zero
    num_bonds_removed = 0
    num_angles_removed = 0
    num_tors_removed = 0
    charges = []
    forces = system.getForces()
    qm_bonds = []
    qm_angles = []
    qm_tors = []
    atoms = list(topology.atoms())
    for force in forces:
        if  isinstance(force, HarmonicBondForce):
            for n in range(force.getNumBonds()):
                a, b, r, k = force.getBondParameters(n)
                if a in qm_atoms and b in qm_atoms:
                    force.setBondParameters(n, a, b, r, k*0.000)
                    num_bonds_removed += 1
                if (a in qm_atoms and b not in qm_atoms) or \
                   (b in qm_atoms and a not in qm_atoms):
                   qm_bonds.append([a, b, r, k])
            force.updateParametersInContext(context)

        elif isinstance(force, HarmonicAngleForce):
            for n in range(force.getNumAngles()):
                a, b, c, t, k = force.getAngleParameters(n)
                in_qm_atoms = [x in qm_atoms for x in [a, b, c]]
                num_qm_atoms = np.sum(in_qm_atoms)
                if num_qm_atoms > 2:
                    force.setAngleParameters(n, a, b, c, t, k*0.000)
                    num_angles_removed += 1
                if num_qm_atoms > 0 and num_qm_atoms < 3  :
                    qm_angles.append([a, b, c, t, k])
            force.updateParametersInContext(context)

        elif isinstance(force, PeriodicTorsionForce):
            for n in range(force.getNumTorsions()):
                a, b, c, d, mult, phi, k = force.getTorsionParameters(n)
                num_qm_atoms = np.sum(in_qm_atoms)
                in_qm_atoms = [x in qm_atoms for x in [a, b, c, d]]
                num_qm_atoms = np.sum(in_qm_atoms)
                if num_qm_atoms > 0:
                    force.setTorsionParameters(n, a, b, c, d, mult, phi, k*0.000)
                    num_tors_removed  += 1
                if num_qm_atoms >= 0 and num_qm_atoms < 3:
                    qm_tors.append([a, b, c, d, mult, phi, k])
            force.updateParametersInContext(context)

        elif isinstance(force, NonbondedForce):

            for n in range(force.getNumParticles()):
                chg, sig, eps = force.getParticleParameters(n)
                charges.append(chg / elementary_charge)
                force.setParticleParameters(n, chg*0, sig, eps*0)
                #if n in qm_atoms or True:
                #    force.setParticleParameters(n, chg*0.0, sig, eps*0)
            force.updateParametersInContext(context)
                

    print(" Number of bonds removed:    ", num_bonds_removed, file=outfile)
    print(" Number of angles removed:   ", num_angles_removed, file=outfile)
    print(" Number of torsions removed: ", num_tors_removed, file=outfile)
    print(" QM-MM Bonds: ", file=outfile)
    for bond in qm_bonds:
        a, b, r, k = bond
        print(" {:3d}  {:3d}".format(a+1, b+1), file=outfile)
    print(" QM-MM Angles: ", file=outfile)
    for angle in qm_angles:
        a, b, c, t, k = angle
        print(" {:3d}  {:3d}  {:3d}".format(a+1, b+1, c+1), file=outfile)
    #print(" QM-MM Torsions: ")
    #for tors in qm_tors:
    #    a, b, c, d, mult, phi, k = tors
    #    print(" {:3d}  {:3d}  {:3d}  {:3d}".format(a+1, b+1, c+1, d+1), file=outfile)
    
    return qm_bonds, qm_angles

def update_qm_force(context, gradient, ext_force, qm_coords_in_nm, qm_atoms, qm_energy=0.0):
    
    dim = len(qm_coords_in_nm)
    total = np.sum(qm_coords_in_nm * gradient)
    context.setParameter('k', total / dim)
    context.setParameter('qm_energy', qm_energy/dim)
    for n in range(ext_force.getNumParticles()):
        idx = ext_force.getParticleParameters(n)[0]
        atom_idx = qm_atoms
        ext_force.setParticleParameters(n, idx, gradient[n])
    ext_force.updateParametersInContext(context)

def update_mm_force(context, ext_force, coords_in_nm, outfile=sys.stdout):
    if os.path.isfile('efield.dat'):
        print(' efield.dat found', file=outfile)
        efield = np.loadtxt('efield.dat') * 2625.5009 / bohrs.conversion_factor_to(nanometer)
        os.remove('efield.dat')
        for n in range(ext_force.getNumParticles()):
            idx, params = ext_force.getParticleParameters(n)
            params = list(params)
            for i in range(3):
                params[i] = efield[idx][i]
            params[4] = np.dot(efield[idx], coords_in_nm[idx])
            ext_force.setParticleParameters(n, idx, params)
        ext_force.updateParametersInContext(context)
    else:
        print(' efield.dat NOT found', file=outfile)

def update_ext_force(simulation, qm_atoms, qm_gradient, ext_force, coords_in_nm, charges, qm_energy=0.0, outfile=sys.stdout):
    ''' Updates external force for ALL atoms, this includes
        QM and MM atoms. QM forces are updated via the qm_gradient,
        while the MM forces are updated from efield.dat file (this
        function searches for this file).
        See add_ext_force_all for parameter listings
    '''
    
    #   import electric field components if file is available
    efield = []
    n_atoms = ext_force.getNumParticles()
    n_qm_atoms = len(qm_atoms)
    e_field_file_loc = 'efield.dat'
    if os.path.isfile(e_field_file_loc):
        print(' efield.dat found', file=outfile)
        with open('efield.dat', 'r') as file:
            lines = file.readlines()
            efield = np.zeros((len(lines), 3))
            for n, line in enumerate(lines):
                sp = line.split()
                if len(sp) == 3:
                    efield[n] = np.array([float(x) for x in sp])
                    
        #efield = np.loadtxt(e_field_file_loc) * 2625.5009 / bohrs.conversion_factor_to(nanometer)
        os.remove('efield.dat')
    else:
        print(' efield.dat NOT found', file=outfile)
        efield = np.zeros((n_atoms - n_qm_atoms, 3))

    #   QM and MM atoms use separate counters as their order 
    #   is (most likely) not contiguous
    qm_idx = 0
    mm_idx = 0
    for n in range(n_atoms):
        idx, params = ext_force.getParticleParameters(n)
        params = list(params)

        if n in qm_atoms:
            gradient = qm_gradient[qm_idx]
            qm_idx += 1
            params[3] = 1.0
        else:
            #gradient = -efield[n]
            gradient = -efield[mm_idx]
            mm_idx += 1
            params[3] = charges[n]

        #   check to make sure that gradient contains valid numbers
        if np.sum(np.isnan(gradient)) != 0:
            print(" ERROR: Gradient index {:d} at step {:d} contains NaN. Setting to Zero.".format(n, simulation.currentStep), file=outfile)
            gradient = np.zeros(3)
        if np.sum(np.isinf(gradient)) != 0:
            print(" ERROR: Gradient index {:d} at step {:d} contains INF. Setting to Zero.".format(n, simulation.currentStep), file=outfile)
            gradient = np.zeros(3)
        
        for i in range(3):
                params[i] = gradient[i]
        params[4] = np.dot(gradient, coords_in_nm[n])
        ext_force.setParticleParameters(n, idx, params)

    simulation.context.setParameter('qm_energy', qm_energy)
    ext_force.updateParametersInContext(simulation.context)
    outfile.flush()

def get_rem_lines(rem_file_loc, outfile):
    rem_lines = []
    rem_lines_in = []

    with open(rem_file_loc, 'r') as file:
        for line in file.readlines():
            if "$" not in line:
                rem_lines_in.append(line.replace('=', ' '))

    opts = JobOptions()

    for line in rem_lines_in:
        orig_line = line
        line = line.lower()
        sp_comment = line.split('!')
        if sp_comment[0] != '!':
            sp = line.split()
            if len(sp) == 0:
                continue
            option = sp[0].lower()
            if option == 'time_step':
                opts.time_step = float(sp[1]) * 0.0242 * femtoseconds
            elif option == 'jobtype':
                opts.jobtype = sp[1]
            elif option == 'aimd_steps' or option == 'geom_opt_max_cycles':
                opts.aimd_steps = int(sp[1])
            elif option == 'aimd_temp':
                opts.aimd_temp = float(sp[1]) * kelvin
            elif option == 'aimd_thermostat':
                opts.aimd_thermostat = sp[1]
            elif option == 'aimd_langevin_timescale':
                opts.aimd_langevin_timescale = float(sp[1]) * femtoseconds
            elif option == 'constrain_qmmm_bonds':
                opts.constrain_qmmm_bonds = strtobool(sp[1])
            elif option == 'ff_file':
                opts.force_field_files.append(orig_line.split()[1])
            elif option == 'constrain_hbonds':
                opts.constrain_hbonds = strtobool(sp[1])


            #   adaptive QM atoms
            elif option == 'qm_mm_radius':
                opts.qm_mm_radius = float(sp[1]) * angstroms
            elif option == 'qm_mm_update':
                opts.qm_mm_update = strtobool(sp[1])
            elif option == 'qm_mm_update_freq':
                opts.qm_mm_update_freq = int(sp[1])

            #   temperature anealing
            elif option == 'annealing':
                opts.annealing = bool(strtobool(sp[1]))
            elif option == 'annealing_peak':
                opts.annealing_peak = float(sp[1]) * kelvin
            elif option == 'annealing_period':
                opts.annealing_period = float(sp[1]) * femtoseconds

            #   osygen boundry force
            elif option == 'oxy_bound':
                opts.oxy_bound = bool(strtobool(sp[1]))
            elif option == 'oxy_bound_force':
                opts.oxy_bound_force = float(sp[1])
            elif option == 'oxy_bound_dist':
                opts.oxy_bound_dist = float(sp[1]) * nanometers            

            #   ratchet and pawl force
            elif option == 'ratchet_pawl':
                opts.ratchet_pawl = bool(strtobool(sp[1]))
            elif option == 'ratchet_pawl_force':
                opts.ratchet_pawl_force = float(sp[1])
            elif option == 'ratchet_pawl_half_dist':
                opts.ratchet_pawl_half_dist = float(sp[1])
            elif option == 'ratchet_pawl_switch':
                opts.ratchet_pawl_switch = sp[1].lower()

            #   charge and multiplicity
            elif option == 'mult':
                opts.mult = int(sp[1])
            elif option == 'charge':
                opts.charge = int(sp[1])
            elif option == 'adapt_spin':
                opts.adapt_mult = bool(strtobool(sp[1]))
            elif option == 'mc_spin':
                opts.mc_spin = bool(strtobool(sp[1]))
            elif option == 'mc_spin_max_mult':
                opts.mc_spin_max_mult = int(sp[1])
            elif option == 'mc_spin_min_mult':
                opts.mc_spin_min_mult = int(sp[1])

            #   oxygen repulsion force
            elif option == 'oxy_repel':
                opts.oxy_repel = bool(strtobool(sp[1]))
            elif option == 'oxy_repel_dist':
                opts.oxy_repel_dist = float(sp[1])

            #   random kicks
            elif option == 'random_kicks':
                opts.random_kicks = bool(strtobool(sp[1]))
            elif option == 'random_kicks_scale':
                opts.random_kicks_scale = float(sp[1])

            #   initial ionization
            elif option == 'ionization':
                opts.ionization = bool(strtobool(sp[1]))
            elif option == 'ionization_num':
                opts.ionization_num = int(sp[1])

            #   restraint force
            elif option == 'restraints':
                opts.restraints = bool(strtobool(sp[1]))
            elif option == 'restraints_switch_time':
                opts.restraints_switch_time = float(sp[1]) * femtoseconds

            elif option == 'cent_restraints':
                opts.cent_restraints = strtobool(sp[1])

            #   QM fragments
            elif option == 'qm_fragments':
                opts.qm_fragments = strtobool(sp[1])

            #   QM / MM Janus Link atoms
            elif option == 'link_atoms':
                opts.link_atoms = strtobool(sp[1])

            #   random number seeds
            elif option == 'aimd_temp_seed':
                seed = int(sp[1])
                if seed > 2147483647 or seed < -2147483648:
                    raise ValueError('rem AIMD_TEMP_SEED must be between -2147483648 and 2147483647')
                opts.aimd_temp_seed = seed
            elif option == 'aimd_langevin_seed':
                seed = int(sp[1])
                if seed > 2147483647 or seed < -2147483648:
                    raise ValueError('rem AIMD_LANGEVIN_SEED must be between -2147483648 and 2147483647')
                opts.aimd_langevin_seed = seed
            else:
                rem_lines.append(line)
            #else:
            #    print("ERROR: rem option < %s > is not supported" % option)
            #    print("       Script will now terminate")
            #    exit()

    #   print rem file to output so user can make sure
    #   it was interpereted correctly
    outfile.write(' Imported rem file \n')
    for line in open(rem_file_loc, 'r').readlines():
        outfile.write(line)
    outfile.write('\n')


    if opts.jobtype == 'aimd':
        if opts.aimd_thermostat:
            opts.integrator = 'Langevin'
        else:
            opts.integrator = 'Verlet'
    elif opts.jobtype.lower() == 'grad':
        opts.integrator = 'Steepest-Descent'
    else:
        opts.integrator = 'Conjugate-Gradient'


    
    outfile.write('--------------------------------------------\n')
    outfile.write('              Script Options                \n')
    outfile.write('--------------------------------------------\n')
    outfile.write(' jobtype:                   {:>10s} \n'.format(opts.jobtype) )
    outfile.write(' integrator:                {:>10s} \n'.format(opts.integrator) )
    outfile.write(' time step:                 {:>10.2f} fs \n'.format(opts.time_step/femtoseconds) )
    outfile.write(' number of steps:           {:>10d} \n'.format(opts.aimd_steps) )
    outfile.write(' Total QM charge:           {:10d} \n'.format(opts.charge))
    outfile.write(' QM Multiplicity:           {:10d} \n'.format(opts.mult))
    outfile.write(' QM/MM radius:              {:>10.2f} Ang. \n'.format(opts.qm_mm_radius/angstroms) )
    outfile.write(' QM/MM update:              {:10d}. \n'.format(opts.qm_mm_update) )
    outfile.write(' QM/MM update frequency:    {:10d} steps. \n'.format(opts.qm_mm_update_freq) )
    outfile.write(' Constrain QM/MM bonds:     {:10d} \n'.format(opts.constrain_qmmm_bonds))

    if opts.adapt_mult:
        outfile.write(' Adaptive Spin:             {:10d} \n'.format(int(opts.adapt_mult)))
    if opts.mc_spin:
        outfile.write(' MCMC Spin:                 {:10d} \n'.format(int(opts.mc_spin)))
        outfile.write(' Max MC Spin Multiplicity:  {:10d} \n'.format(int(opts.mc_spin_max_mult)))
        outfile.write(' Min MC Spin Multiplicity:  {:10d} \n'.format(int(opts.mc_spin_min_mult)))

    if opts.ratchet_pawl:
        outfile.write(' Ratchet-Pawl:              {:10d} \n'.format(int(opts.ratchet_pawl)))
        outfile.write(' Ratchet-Pawl Force:        {:10.1f} \n'.format(float(opts.ratchet_pawl_force)))
        outfile.write(' Ratchet-Pawl Half-Dist:    {:10.4f} \n'.format(opts.ratchet_pawl_half_dist))
        outfile.write(' Ratchet-Pawl Switching:    {:>10s} \n'.format(opts.ratchet_pawl_switch))

    if opts.jobtype == 'aimd':
        outfile.write(' temperature:               {:>10.2f} K \n'.format(opts.aimd_temp/kelvin) )
        outfile.write(' temperature seed:          {:>10d} \n'.format(opts.aimd_temp_seed) )

    if opts.aimd_thermostat:
        outfile.write(' thermostat:                {:>10s} \n'.format(opts.aimd_thermostat) )
        outfile.write(' langevin frequency:      1/{:>10.2f} fs \n'.format(opts.aimd_langevin_timescale / femtoseconds) )
        outfile.write(' langevin seed:            {:11d} \n'.format(opts.aimd_langevin_seed))

    if opts.annealing:
        outfile.write(' Temperature Annealing:     {:10d} \n'.format(int(opts.annealing)))
        outfile.write(' Annealing Peak:            {:10.2f} K\n'.format(opts.annealing_peak / kelvin))
        outfile.write(' Annealing Period:          {:10.1f} fs\n'.format(opts.annealing_period/femtoseconds))

    if opts.oxy_bound:
        outfile.write(' Oxygen boundry:            {:10d} \n'.format(int(opts.oxy_bound)))
        outfile.write(' Oxygen boundry force:      {:10.4f} nm \n'.format(opts.oxy_bound_dist / nanometers))
        outfile.write(' Oxygen boundry distance:   {:10.1f} \n'.format(opts.oxy_bound_force))

    if opts.oxy_repel:
        outfile.write(' Oxy - Oxy Repulsion:       {:10d} \n'.format(int(opts.oxy_repel)))
        outfile.write(' Oxy - Oxy Distance:        {:10.4f} \n'.format(float(opts.oxy_repel_dist)))

    if opts.random_kicks:
        outfile.write(' Random Thermal Kicks:      {:10d} \n'.format(int(opts.random_kicks)))
        outfile.write(' Random Kicks Scale:        {:10.5f} \n'.format(opts.random_kicks_scale))

    if opts.ionization:
        outfile.write(' Initial Ionization:        {:10d} \n'.format(int(opts.ionization)))
        outfile.write(' No. of ionized H2O pairs:  {:10d} \n'.format(opts.ionization_num))

    if opts.restraints:
        outfile.write(' Restraints:                {:10d} \n'.format(int(opts.restraints)))
        outfile.write(' Restraints Switch Time:    {:>10f} fs \n'.format(opts.restraints_switch_time/femtoseconds))

    if opts.cent_restraints:
        outfile.write(' Centroid Restraints:       {:10d} \n'.format(int(opts.cent_restraints)))

    if opts.qm_fragments:
        outfile.write(' QM Fragments:              {:10d} \n'.format(int(opts.qm_fragments)))

    if opts.link_atoms:
        outfile.write(' QM/MM Link Atoms:          {:10d} \n'.format(int(opts.link_atoms)))

    if opts.constrain_hbonds:
        outfile.write(' H-Bond Constraints:        {:10d} \n'.format(int(opts.constrain_hbonds)))

    outfile.write('--------------------------------------------\n')

    if len(opts.force_field_files) > 0:
        outfile.write(' Additional force fields:   \n')
        for f in opts.force_field_files:
            outfile.write('     %s\n' % f)
        outfile.write('--------------------------------------------\n')

    outfile.flush()
    return rem_lines, opts

def get_integrator(opts):
    if opts.jobtype == 'aimd':
        if opts.integrator.lower() == 'langevin':
            integrator =  LangevinIntegrator(opts.aimd_temp, 1/opts.aimd_langevin_timescale, opts.time_step)
            #   32-bit random number seed shifted to c++ min/max integer limits
            integrator.setRandomNumberSeed(int(os.urandom(4).hex(), 16) - 2147483647)
            return integrator
        else:
            return VerletIntegrator(opts.time_step)
    elif opts.jobtype == 'friction':
        #   Langevin integrator without the random noise
        integrator = CustomIntegrator(opts.time_step)
        integrator.addGlobalVariable("v_scale", 0.9)
        integrator.addComputePerDof("v", "v + dt*f/m")
        integrator.addComputePerDof("x", "x + 0.5*dt*v")
        integrator.addComputePerDof("v", "v_scale*v")
        integrator.addComputePerDof("x", "x + 0.5*dt*v")

        #integrator.addComputePerDof("x", "x + dt*f")
        return integrator

    else:
        if True:
            integrator = CustomIntegrator(opts.time_step)

            #timestep = 1.0 * unit.femtoseconds
            integrator.addGlobalVariable("step_size", opts.time_step*0 / nanometers)
            integrator.addGlobalVariable("energy_old", 0)
            integrator.addGlobalVariable("energy_new", 0)
            integrator.addGlobalVariable("delta_energy", 0)
            integrator.addGlobalVariable("accept", 0)
            integrator.addGlobalVariable("fnorm2", 0)
            integrator.addPerDofVariable("x_old", 0)
            integrator.addPerDofVariable("dir_old", 0)
            integrator.addPerDofVariable("dir", 0)
            
            integrator.addComputePerDof("x", "x")
            return integrator


        return integrator
   

def print_initial_forces(simulation, qm_atoms, topology, outfile):
    #   test to make sure that all qm_forces are fine
    state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
    forces = state.getForces()
    atoms = list(topology.atoms())
    print(" Initial Potential energy: ", state.getPotentialEnergy(), file=outfile)
    print(" Initial forces exerted on QM atoms: ", file=outfile)
    for n in qm_atoms:
        f = forces[n]/forces[n].unit
        f = np.linalg.norm(f)
        atom = atoms[n]
        print(" {:4s}  {:4s} residue {:4s} |  {:10.2f}  kJ/mol/nm ".format(atom.id, atom.name, atom.residue.id, f), file=outfile)

        #print(" Force on atom {:3d}: {:10.2f}  {:10.2f}  {:10.2f} kJ/mol/nm"
        #.format((n+1), f[0], f[1], f[2]), file=outfile)
    print(" Check to make sure that all forces are ~10^3 or less.", file=outfile)
    print(" Larger forces may indicate poor initial coordinates", file=outfile)
    print(" or an inproper force field parameterization.", file=outfile)

def ionize(topology, coords, qm_atoms, num):

    #   extract which residues are QM waters
    qm_water_residues = set()
    for res in topology.residues():
        if res.name == 'HOH':
            for atom in res.atoms():
                if atom.index in qm_atoms:
                    qm_water_residues.add(res)
                    break

    for n in range(num):
        water_residues = list(qm_water_residues)
        idx_list = np.arange(len(qm_water_residues))
        shuffle(idx_list)
        res1 = water_residues[idx_list[0]]
        res2 = water_residues[idx_list[1]]
        qm_water_residues.remove(res1)
        qm_water_residues.remove(res2)        

        #   A single hydrogen is going to be moved from res1 to res2.
        #   The oxygen of res2 will act as the center of
        O = np.empty(3)
        H1 = np.empty(3)
        H2 = np.empty(3)
        for atom in res2.atoms():
            if atom.element.symbol == 'O':
                O = coords[atom.index]/nanometers
                print("O: ", atom.id, O)
            elif atom.name == 'H1':
                H1 = coords[atom.index]/nanometers
                print("H1: ", atom.id, H1)
            else:
                H2 = coords[atom.index]/nanometers
                print("H2: ", atom.id, H2)

        #   the new hydrogen is now positioned first in the plane of the
        #   water molecule, and then extruded slightly out of the plane to
        #   create the tetrahedral shape

        z_length = 0.05                              # length to extrude out of water plane
        xy_length = sqrt(0.1**2 - z_length**2)                # distance from oxygen atom to shift in plane
        center = (H1 + H2)/2                            # center of hydrogen atoms
        center_to_O = (O - center)/norm(O - center)     # unit vec from center to oxygen
        new_xy = O + center_to_O*xy_length              # position above oxygen in plane of water
        center_to_H1 = (H1 - center)/norm(H1 - center)  # unit vec form center to H1
        normal = np.cross(center_to_H1, center_to_O)       # unit normal to plane of water
        normal = normal/norm(normal)
        new_xyz = new_xy + normal*z_length              # now shift out of plane by small amount
                                                        # to make tetrahedral
        
        for atom in res1.atoms():
            if atom.name == 'H1':
                coords[atom.index] = new_xyz*nanometers


def main(args):
    global scratch, n_procs, qc_scratch, qchem_path, qm_fragments
    oxygen_force = None
    ratchet_pawl_force = None

    #   make sure Q-Chem is available, exit otherwise
    if 'QC' in os.environ:
        qchem_path = os.environ.get('QC')
        qc_scratch = os.environ.get('QCSCRATCH')
        print(" QC set as ", qchem_path)
        print(" QCSCRATCH set as ", qc_scratch)
    else:
        print(" Error: environment variable QC not defined. Cannot find Q-Chem directory")
        exit()

    if args.nt:
        n_procs = args.nt

    with open(args.out, 'w') as outfile:

        rem_lines, options = get_rem_lines(args.rem, outfile)
        pdb = PDBFile(args.pdb)
        pdb_to_qc.add_bonds(pdb, remove_orig=False)
        data, bondedToAtom = pdb_to_qc.determine_connectivity(pdb.topology)

        ff_loc = os.path.join(os.path.dirname(__file__), 'forcefields/forcefield2.xml')
        forcefield = ForceField(ff_loc, 'tip3p.xml', *tuple(options.force_field_files))
        #return forcefield, pdb.topology
        [templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(pdb.topology)
        #return (templates, residues)
        for n, template in enumerate(templates):
            residue = residues[n]
            atom_names = []            
            for atom in template.atoms:
                if residue.name in ['EXT', 'OTH']:
                    atom.type = 'OTHER-' + atom.element.symbol
                elif atom.element.symbol == 'N' and residue.name in ['ETH', 'MTH']:
                    atom.type = 'LIG-N'
                else:
                    atom.type = residue.name + "-" + atom.name.upper()
                atom_names.append(atom.name)

            # Register the template with the forcefield.
            template.name += str(n)
            forcefield.registerResidueTemplate(template)

        integrator = get_integrator(options)
        qm_fixed_atoms, qm_origin_atoms = parse_idx(args.idx, pdb.topology)

        #   set initial number of QM atoms
        qm_sphere_atoms = get_qm_spheres(qm_origin_atoms, qm_fixed_atoms, options.qm_mm_radius/angstroms, pdb.getPositions()/angstrom, pdb.topology)
        qm_atoms = qm_fixed_atoms + qm_sphere_atoms
        if options.constrain_hbonds:
            system = forcefield.createSystem(pdb.topology, rigidWater=False, constraints=HBonds)
        else:
            system = forcefield.createSystem(pdb.topology, rigidWater=False)

        #   QM fragment molecules
        if options.qm_fragments:
            qm_fragments = QM_Fragments(args.frags, pdb.topology)

        #   re-map nonbonded forces so QM (mostly) only interacts with MM through vdW
        charges = add_nonbonded_force(qm_atoms, system, pdb.topology.bonds(), outfile=outfile)

        #   setup Q-Chem Runner for submitting and running QM part of the simulation
        qchem = QChemRunner(rem_lines, pdb.topology, charges, options, outfile, scratch, args.frags, args.link)

        #   "external" force for updating QM forces and MM electrostatics
        ext_force = add_ext_force_all(system, charges)

        if options.ionization:
            ionize(pdb.topology, pdb.positions, qm_atoms, options.ionization_num)
            PDBFile.writeModel(pdb.topology, pdb.positions, file=open('ionized.pdb', 'w'), keepIds=True)

        #   add ratchet-pawl force
        if options.ratchet_pawl:
            ratchet_pawl_force = add_rachet_pawl_force(system, args.pawl, pdb.getPositions(True), \
                options.ratchet_pawl_force, pdb.topology, half_dist=options.ratchet_pawl_half_dist,
                switch_type=options.ratchet_pawl_switch)

        #   add oxygen boundry force
        if options.oxy_bound:
            oxygen_force = BoundryForce(system, pdb.topology, pdb.getPositions(True), qm_atoms, \
                options.oxy_bound_dist, options.oxy_bound_force)

        if options.oxy_repel:
            add_oxygen_repulsion(system, pdb.topology, options.oxy_repel_dist)

        if options.restraints:
            restraints = RestraintsForce(system, pdb.topology, args.rest, options.restraints_switch_time)
        
        if options.cent_restraints:
            cent_restraints = CentroidRestraintForce(system, pdb.topology, args.centroid)

        #   debug only: turns off forces except one
        if False:
            while system.getNumForces() > 1:
                for i, force in enumerate(system.getForces()):
                    if not isinstance(force, CustomNonbondedForce):
                        system.removeForce(i)
                        break

        #   turn on to freeze mm atoms in place
        if args.freeze:
            fix_idx = np.loadtxt(args.freeze, dtype=int)
            for atom in pdb.topology.atoms():
                if int(atom.id) in fix_idx:
                    print("FREEZE: ", atom.id, file=outfile)
                    system.setParticleMass(atom.index, 0*dalton)

        #   add constraints
        if options.constrain_qmmm_bonds:
            fix_qm_mm_bonds(system, qm_atoms, pdb.positions, outfile)

        #   initialize simulation and set positions
        simulation = Simulation(pdb.topology, system, integrator)

        if args.state:
            print(" Setting initial positions and velocities from state file: ", file=outfile)
            print(" {:s}".format(os.path.abspath(args.state)), file=outfile)
            simulation.loadState(args.state)
        elif args.pos:
            print(" Setting initial positions from PDB file: ", file=outfile)
            print(" {:s}".format(os.path.abspath(args.pos)), file=outfile)
            simulation.context.setPositions(PDBFile(args.pos).getPositions())
        else:
            print(" Setting initial positions from PDB file: ", file=outfile)
            print(" {:s}".format(os.path.abspath(args.pdb)), file=outfile)
            simulation.context.setPositions(pdb.positions)



        #   remove bonded forces between QM and MM system
        adjust_forces(system, simulation.context, pdb.topology, qm_atoms, outfile=outfile)

        #   random kicks force
        if options.random_kicks:
            kicks = RandomKicksForce(simulation, pdb.topology, options.aimd_temp, scale=options.random_kicks_scale)

        #   output files and reporters
        stats_reporter = StatsReporter('stats.txt', 1, options, qm_atoms=qm_atoms, vel_file_loc=args.repv, force_file_loc=args.repf)
        simulation.reporters.append(HDF5Reporter('output.h5', 1))
        qm_atoms_reporter = QMatomsReporter('qm_atoms.txt', pdb.topology)
        if options.cent_restraints:
            simulation.reporters.append(CentroidRestraintForceReporter(cent_restraints.getForce(), outfile, 1, system))
        

        #   set up files
        os.makedirs(scratch, exist_ok=True)
        os.makedirs(qc_scratch, exist_ok=True)
        atoms = list(pdb.topology.atoms())
        elements = [x.element.symbol for x in atoms]

        #   test to make sure that all qm_forces are fine
        print_initial_forces(simulation, qm_atoms, pdb.topology, outfile)

        if options.jobtype == 'opt':
            #opt = GradientMethod(options.time_step*0.001)
            opt = BFGS(options.time_step*0.001)

        if options.jobtype != 'opt' and not args.state and options.jobtype != 'friction':
            print(" Setting initial velocities to temperature of {:5f} K: ".format(options.aimd_temp/kelvin), file=outfile)
            simulation.context.setVelocitiesToTemperature(options.aimd_temp, options.aimd_temp_seed)
        else:
            print(" Setting initial velocities to Zero: ", file=outfile)
            simulation.context.setVelocities([Vec3(0, 0, 0)*nanometers/picosecond]*pdb.topology.getNumAtoms())

        #   for sanity checking
        print(' Integrator: ', type(integrator))
        sys.stdout.flush()

        simulation.saveState('initial_state.xml')
        water_filler = WaterFiller(pdb.topology, forcefield, simulation)

        #return system

        #   minimization for MM only system
        if len(qm_atoms) == 0 and options.jobtype == 'opt':
            reporter = (stats_reporter.report, (simulation, qm_atoms))
            optimize = MMOnlyBFGS(simulation, topology=pdb.topology, reporter=reporter, outfile=outfile)
            optimize.minimize()
            return

        #   run simulation
        for n in range(options.aimd_steps):
            if options.annealing:
                #   add increase in temperature
                #   new temperature is T_0 + A*sin(t*w)^2
                omega = np.pi / options.annealing_period
                temp_diff = options.annealing_peak - options.aimd_temp
                current_temp = options.aimd_temp  + temp_diff * np.sin(options.time_step * n * omega)**2
                integrator.setTemperature(current_temp)
                outfile.write("\n Current temperature: {:10.5f} K \n".format(current_temp / kelvin))
                if options.random_kicks:
                    kicks.set_temperature(current_temp)

            state = simulation.context.getState(getPositions=True)
            pos = state.getPositions(True)

            # update QM atom list and water positions
            if n % options.qm_mm_update_freq == 0 and options.qm_mm_update:
                #if len(qm_atoms) > 0:
                #    pos = water_filler.fill_void(pos, qm_atoms, outfile=outfile)

                qm_sphere_atoms = get_qm_spheres(qm_origin_atoms, qm_fixed_atoms, options.qm_mm_radius/angstroms, pos/angstrom, pdb.topology)
                qm_atoms = qm_fixed_atoms + qm_sphere_atoms
                qm_atoms = update_mm_forces(qm_atoms, system, simulation.context, pos, pdb.topology, outfile=outfile)


            qm_atoms_reporter.report(simulation, qm_atoms)
            if len(qm_atoms) > 0:
                qm_energy, qm_gradient = qchem.get_qm_force(pos/angstroms, qm_atoms, n)
                update_ext_force(simulation, qm_atoms, qm_gradient, ext_force, pos/nanometers, charges, qm_energy=qm_energy, outfile=outfile)

            #   additional force updates
            if options.ratchet_pawl:
                update_rachet_pawl_force(ratchet_pawl_force, simulation.context, pos/nanometers, outfile=outfile)
            if options.oxy_bound:
                oxygen_force.update(simulation.context, pos, outfile=outfile)
            if options.restraints:
                restraints.update(simulation, pos, outfile=outfile)            
            
            if n % 10  == 0:
                simulation.saveState('simulation.xml')

            # if options.jobtype == 'opt':
            #     opt.prepare_step(simulation, outfile=outfile)

            #   update current velocity with random kicks
            if options.random_kicks:
                kicks.update(outfile=outfile)

            #   call reporter before taking a step. OpenMM calls reporters after taking a step, but this
            #   will not report the correct force and energy as the current parameters are only valid 
            #   for the current positions, not after
            stats_reporter.report(simulation, qm_atoms)
            simulation.step(1)
            if options.jobtype == 'opt':
                opt.step(simulation)
            


    return simulation
              
if __name__ == "__main__":

    #tmp_args = parse_args(sys.argv[1:])
    arg_list = parse_cmd_line_args(scratch)
    prog_args = parse_args(arg_list)
    res = main(prog_args)

    ''' DEBUGGING ONLY  '''
    '''
    system = local['system']
    atoms = list(local['pdb'].topology.atoms())
    for force in system.getForces():
        if  isinstance(force, HarmonicBondForce):
            for n in range(force.getNumBonds()):
                a, b, r, k = force.getBondParameters(n)
                ids = [int(atoms[x].id) for x in [a, b]]
                for id in ids:
                    if id in [68]:
                        print(a, b, k, r, ids)
    '''
        #if  isinstance(force, CustomNonbondedForce):
        #    for n in range(force.getNumParticles()):
        #        params = list(force.getParticleParameters(n))
        #        print(atoms[n].id, params)
        


