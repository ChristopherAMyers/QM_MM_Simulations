from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm import * #pylint: disable=unused-wildcard-import
from openmmtools.integrators import VelocityVerletIntegrator
from simtk.openmm.openmm import * #pylint: disable=unused-wildcard-import
from simtk.unit import * #pylint: disable=unused-wildcard-import
import argparse
import numpy as np
from numpy.linalg import norm
import sys
import itertools
from multiprocessing import cpu_count
from subprocess import run
import time
import shutil
from scipy import optimize
from optimize import GradientMethod
from mdtraj.reporters import HDF5Reporter
from distutils.util import strtobool
from code import interact

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

qchem_path = ''
qc_scratch = '/tmp'
n_procs = cpu_count()


def parse_args(args_in):
    parser = argparse.ArgumentParser('')
    parser.add_argument('-pdb',   required=True, help='pdb molecule file to use')
    parser.add_argument('-rem',   required=True, help='rem arguments to use with Q-Chem')
    parser.add_argument('-idx',   required=True, help='list of atoms to treat as QM')
    parser.add_argument('-out',   help='output file', default='output.txt')
    parser.add_argument('-pos',   help='set positions using this pdb file')
    parser.add_argument('-state', help='start simulation from simulation state xml file')
    parser.add_argument('-repf',  help='file to print forces to')
    parser.add_argument('-repv',  help='file to print velocities to')
    parser.add_argument('-pawl',  help='list of atom ID pairs to apply ratchet-pawl force between')
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
                    if np.linalg.norm(xyz_in_ang[int(i)] - xyz_in_ang[atom.index]) < radius_in_ang:
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

def determine_mult(topology, coords, current_mult):
    '''
        determine system spin multiplicity based on number of O2 molecules
    '''

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
                #force.setBondParameters(n, a, b, r, k*0.000)
                #if 952 in [a, b]:
                #    print(a, b, r, k, atoms[a].name, atoms[b].name, atoms[a].residue.id, atoms[b].residue.id)
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
                if num_qm_atoms > 0:
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

def create_qc_input(coords, charges, elements, qm_atoms, total_chg=0, spin_mult=1, rem_lines=[], step_number=0, ghost_atoms=[], jobtype=None):
    global scratch
    input_file_loc = os.path.join(scratch, 'input')
    with open(input_file_loc, 'w') as file:
        #   copy over rem job lines
        file.write('$rem \n')
        for line in rem_lines:
            if line[-1] != '\n':
                line = line + '\n'
            file.write(line)

        #   use the previous jobs orbitals as an initial guess
        if step_number % 10 != 0:
            file.write('    scf_guess read \n')
        if not jobtype:
            file.write('    jobtype     force \n')
            file.write('    qm_mm       true \n')
        else:
            file.write('    jobtype     {:s} \n'.format(jobtype))
        file.write('    sym_ignore  true \n')
        file.write('$end \n\n')

        #   mm_atoms are represented as external charges
        mol_lines = []
        chg_lines = []
        for n, coord in enumerate(coords):
            if n in qm_atoms:
                mol_lines.append('    {:2s}  {:15.8f}  {:15.8f}  {:15.8f} \n'
                .format(elements[n], coord[0], coord[1], coord[2]))
            elif n in ghost_atoms:
                mol_lines.append('    {:2s}  {:15.8f}  {:15.8f}  {:15.8f} \n'
                .format('@H', coord[0], coord[1], coord[2]))
            else:
                chg_lines.append('    {:15.8f}  {:15.8f}  {:15.8f}  {:15.8f} \n'
                .format(coord[0], coord[1], coord[2], charges[n]))
        

        #   write molecule section
        file.write('$molecule \n')
        file.write('    {:d}  {:d} \n'.format(int(total_chg), spin_mult))
        for line in mol_lines:
            file.write(line)
        file.write('$end \n\n')

        #   write external charges
        file.write('$external_charges \n')
        for line in chg_lines:
            file.write(line)
        file.write('$end \n\n')

        return input_file_loc

def get_qm_force(coords, charges, elements, qm_atoms, output_file, topology, opts, total_chg=0, rem_lines=[], step_number=0, copy_input=False, outfile=sys.stdout, spin_mult=1):

    qm_energy, qm_gradient = calc_qm_force(coords, charges, elements, qm_atoms, output_file, total_chg, rem_lines, step_number, copy_input, outfile, spin_mult)


    #   adaptive spin multiplicity for O2
    if (opts.adapt_mult or opts.mc_spin) and (step_number % 10 == 0):
        #   spin mult based on number of O2 molecules
        if opts.adapt_mult:
            new_mult = determine_mult_from_coords(coords*10*nanometers, topology, spin_mult, qm_atoms)
            if new_mult != spin_mult:
                new_energy, new_gradient = calc_qm_force(coords, charges, elements, qm_atoms, output_file, total_chg, rem_lines, step_number, copy_input, outfile, new_mult)

            if new_energy < qm_energy:
                outfile.write("\n Step {:d}: Changing spin multiplicity from {:d} to {:d} \n".format(n, spin_mult, new_mult))
                outfile.write(" Energy difference: {:.5f} kJ/mol \n\n".format((new_energy - qm_energy)/kilojoules_per_mole))
                qm_energy = new_energy
                qm_gradient = new_gradient
                #   change spin mult of system to the new multiplicity
                opts.mult = new_mult

        #   marcov chain spin flip
        elif opts.mc_spin:
            up_or_down = np.random.randint(0, 2)
            if up_or_down == 0:
                new_mult = spin_mult - 2
            else:
                new_mult = spin_mult + 2

            new_energy, new_gradient = calc_qm_force(coords, charges, elements, qm_atoms, output_file, total_chg, rem_lines, step_number, copy_input, outfile, new_mult)
            energy_diff = (new_energy - qm_energy)/kilojoules_per_mole

            print(" Appempted new spin multiplicity at step {:d}".format(step_number), file=outfile)
            print(" Old spin multiplicity: {:d}".format(spin_mult), file=outfile)
            print(" New spin multiplicity: {:d}".format(new_mult), file=outfile)
            print(" Energy difference: {:15.5f} kJ/mol".format(energy_diff), file=outfile)


            accept = False
            if energy_diff < 0:
                print(" Accepted new multiplicity", file=outfile)
                accept = True
            else:
                y = np.random.rand()
                kT = opts.aimd_temp/kelvin * 0.00831446261815324
                if y < np.exp(-energy_diff/kT):
                    accept = True
                    print(" Accepted new multiplicity from Boltzmann probability", file=outfile)
                else:
                    print(" Denied new multiplicity from Boltzmann probability", file=outfile)
            
            if accept:
                qm_energy = new_energy
                qm_gradient = new_gradient
                opts.mult = new_mult

    return qm_energy, qm_gradient


def calc_qm_force(coords, charges, elements, qm_atoms, output_file, total_chg=0, rem_lines=[], step_number=0, copy_input=False, outfile=sys.stdout, spin_mult=1):
    global scratch, qc_scratch, n_procs
    redo = True
    failures = 0
    use_rem_lines = copy.copy(rem_lines)
    while redo:
        outfile.flush()
        input_file_loc = create_qc_input(coords, charges, elements, qm_atoms, total_chg=total_chg, spin_mult=spin_mult, rem_lines=use_rem_lines, step_number=step_number)
        output_file_loc = os.path.join(scratch, 'output')
        cmd = os.path.join(qchem_path, 'bin/qchem') + ' -save -nt {:d}  {:s}  {:s} save_files'.format(n_procs, input_file_loc, output_file_loc)

        outfile.write(' --------------------------------------------\n')
        outfile.write('           Starting Q-Chem\n')
        outfile.write(' --------------------------------------------\n')

        if copy_input:
            with open(input_file_loc, 'r') as file:
                for line in file.readlines():
                    outfile.write(line)

        outfile.flush()
        start = time.time()
        if step_number == 0 and False:
            shutil.copyfile(os.path.join(qc_scratch, 'save_files/output'), output_file_loc)
        else:
            run(cmd.split())
        end = time.time()
        #   forward Q-Chem output to output file and search for completion
        found_thank_you = False
        found_scf_failure = False
        with open(output_file_loc, 'r', errors='ignore') as file:
            for line in file.readlines():
                if "Thank you" in line:
                    found_thank_you = True
                if "SCF failed to converge" in line:
                    found_scf_failure = True
                outfile.write(line)

        if not found_thank_you:
            failures += 1

        #   if SCF failed, try once more with different algorithm 
        if found_scf_failure and failures == 1:
            redo = True
            use_rem_lines.append('scf_algorithm DIIS_GDM')
            if spin_mult == 1:
                use_rem_lines.append('unrestricted False')
        elif found_thank_you or failures > 1:
            redo = False

    gradient = []
    energy = 0.0
    grad_file_loc = os.path.join(qc_scratch, 'save_files/GRAD')
    
    if os.path.isfile(grad_file_loc):
        #shutil.copyfile(grad_file_loc, os.path.join(scratch, 'GRAD'))
        energy = np.loadtxt(grad_file_loc, skiprows=1, max_rows=1)
        gradient = np.loadtxt(grad_file_loc, skiprows=3, max_rows=len(qm_atoms))

    if found_thank_you:
        outfile.write(' -----------------------------------------------------------\n')
        outfile.write('              Q-Chem completed successfully\n')
        outfile.write(' -----------------------------------------------------------\n')
        outfile.write('  Total Q-Chem time: wall {:10.2f}s\n'.format(end - start))
    else:
        outfile.write(' -----------------------------------------------------------\n')
        outfile.write('       Q-Chem DID NOT complete successfully \n')
        outfile.write('       TERMINATING PROGRAM \n')
        outfile.write(' -----------------------------------------------------------\n')
        exit()

    if len(gradient) != 0:
        outfile.write(' Gradient fle found \n')
        outfile.write('                     {:13s}  {:13s}  {:13s} \n'.format('X', 'Y', 'Z'))
        outfile.write(' -----------------------------------------------------------\n')
        for n, grad in enumerate(gradient):
            outfile.write('    {:3d}  {:2s}  {:13.8f}  {:13.8f}  {:13.8f} \n'
            .format(n + 1, elements[qm_atoms[n]], grad[0], grad[1], grad[2]))
        outfile.write(' -----------------------------------------------------------\n')
    else:
        outfile.write(' Gradient fle NOT found \n')
        outfile.write('      TERMINATING PROGRAM \n')
        outfile.write(' -----------------------------------------------------------\n')
        exit()

    #   debug only: save all generated input files
    #shutil.copyfile(os.path.join(scratch, 'input'), os.path.join(scratch, 'input_{:d}'.format(step_number)))

    #   convert gradient to kJ/mol/nm and energy to kJ/mol
    gradient = (gradient * 2625.5009 * kilojoules_per_mole / nanometer / bohrs.conversion_factor_to(nanometer))
    energy = energy * 2625.5009 * kilojoules_per_mole
    return (energy, gradient)

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
    n_atoms = ext_force.getNumParticles()
    n_qm_atoms = len(qm_atoms)
    e_field_file_loc = 'efield.dat'
    if os.path.isfile(e_field_file_loc):
        print(' efield.dat found', file=outfile)
        efield = np.loadtxt(e_field_file_loc) * 2625.5009 / bohrs.conversion_factor_to(nanometer)
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
            elif option == 'qm_mm_radius':
                opts.qm_mm_radius = float(sp[1]) * angstroms

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

            #   oxygen repulsion force
            elif option == 'oxy_repel':
                opts.oxy_repel = bool(strtobool(sp[1]))
            elif option == 'oxy_repel_dist':
                opts.oxy_repel_dist = float(sp[1])

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
        opts.time_step = 1.0 * femtoseconds
    else:
        opts.integrator = 'Conjugate-Gradient'
        opts.time_step = 1.0 * femtoseconds
    
    outfile.write('--------------------------------------------\n')
    outfile.write('              Script Options                \n')
    outfile.write('--------------------------------------------\n')
    outfile.write(' jobtype:                   {:>10s} \n'.format(opts.jobtype) )
    outfile.write(' integrator:                {:>10s} \n'.format(opts.integrator) )
    outfile.write(' time step:                 {:>10.2f} fs \n'.format(opts.time_step/femtoseconds) )
    outfile.write(' QM/MM radius:              {:>10.2f} Ang. \n'.format(opts.qm_mm_radius/angstroms) )
    outfile.write(' number of steps:           {:>10d} \n'.format(opts.aimd_steps) )
    outfile.write(' Total QM charge:           {:10d} \n'.format(opts.charge))
    outfile.write(' QM Multiplicity:           {:10d} \n'.format(opts.mult))

    if opts.adapt_mult:
        outfile.write(' Adaptive Spin:             {:10d} \n'.format(int(opts.adapt_mult)))
    if opts.mc_spin:
        outfile.write(' MCMC Spin:                 {:10d} \n'.format(int(opts.mc_spin)))

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
        integrator.addGlobalVariable("step_size", opts.time_step)
        integrator.addComputePerDof("v", "v + dt*f/m")
        integrator.addComputePerDof("x", "x + 0.5*dt*v")
        integrator.addComputePerDof("v", "0.2*v")
        integrator.addComputePerDof("x", "x + 0.5*dt*v")
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
   
def gen_qchem_opt(options, simulation, charges, elements, qm_atoms, qm_bonds, qm_angles, rem_lines, outfile):
    global scratch, qc_scratch
    #   all MM atoms involved with bonds and angles
    #   with a QM atoms will be treated as a 'ghost' atom
    ghost_atoms = set()
    for bond in qm_bonds:
        a, b, r, k = bond
        if a in qm_atoms:
            ghost_atoms.add(b)
        else:
            ghost_atoms.add(a)

    keep_angles = []
    for angle in qm_angles:
        a, b, c, t, k = angle
        in_qm_atoms = [x in qm_atoms for x in [a, b, c]]
        num_qm_atoms = np.sum(in_qm_atoms)
        if num_qm_atoms == 1:
            keep_angles.append(angle)
            for x in [a, b, c]:
                if x not in qm_atoms:
                    ghost_atoms.add(x)

    #   create list of both qm_atoms and ghost_atoms
    qchem_atoms = copy.copy(ghost_atoms)
    for n in qm_atoms:
        qchem_atoms.add(n)
    print(qchem_atoms)
    qchem_atoms = sorted(qchem_atoms)
    ghost_atoms = sorted(ghost_atoms)

    #   create q-chem input file
    state = simulation.context.getState(getPositions=True)
    coords = state.getPositions(True) / angstrom
    input_file_loc = create_qc_input(coords, charges, elements, qm_atoms, rem_lines=rem_lines, ghost_atoms=ghost_atoms, jobtype='opt')

    #   append to q-chem input file the constraints
    with open(input_file_loc, 'a') as file:
        file.write('\n\n')
        file.write('$opt \n')
        file.write('CONSTRAINT \n')
        for bond in qm_bonds:
            a, b, r, k = bond
            idx1 = qchem_atoms.index(a) + 1
            idx2 = qchem_atoms.index(b) + 1
            file.write('    stre {:3d}  {:3d}  {:10.3f} \n'.format(idx1, idx2, r/angstrom))
        for angle in keep_angles:
            a, b, c, t, k = angle
            idx1 = qchem_atoms.index(a) + 1
            idx2 = qchem_atoms.index(b) + 1
            idx3 = qchem_atoms.index(c) + 1
            file.write('    bend {:3d}  {:3d}  {:3d}  {:10.3f} \n'.format(idx1, idx2, idx3, t / degrees))
        file.write('ENDCONSTRAINT \n')
        file.write('FIXED \n')
        print(ghost_atoms)
        for x in ghost_atoms:
            file.write('    {:3d}  XYZ \n'.format(qchem_atoms.index(x) + 1))
        file.write('ENDFIXED \n')
        file.write('$end \n')

    #   run Q-Chem
    output_file_loc = os.path.join(scratch, 'output')
    cmd = os.path.join(qchem_path, 'bin/qchem') + ' -save -nt {:d}  {:s}  {:s} save_files'.format(n_procs, input_file_loc, output_file_loc)

    outfile.write('--------------------------------------------\n')
    outfile.write('           Starting Q-Chem\n')
    outfile.write('--------------------------------------------\n')
    outfile.flush()
    run(cmd.split())

    with open(output_file_loc, 'r') as file:
        for line in file.readlines():
            outfile.write(line)

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


def main(args_in):
    global scratch, n_procs, qc_scratch, qchem_path
    oxygen_force = None
    ratchet_pawl_force = None

    args = parse_args(args_in)

    #   make sure Q-Chem is available, exit otherwise
    if 'QC' in os.environ:
        qchem_path = os.environ.get('QC')
        qc_scratch = os.environ.get('QCSCRATCH')
        print(" QC set as ", qchem_path)
        print(" QCSCRATCH set as ", qc_scratch)
    else:
        print(" Error: environment variable QC not defined. Cannot find Q-Chem directory")
        exit()

    with open(args.out, 'w') as outfile:
        rem_lines, options = get_rem_lines(args.rem, outfile)
        pdb = PDBFile(args.pdb)
        pdb_to_qc.add_bonds(pdb, remove_orig=True)
        data, bondedToAtom = pdb_to_qc.determine_connectivity(pdb.topology)

        ff_loc = os.path.join(os.path.dirname(__file__), 'forcefields/forcefield2.xml')
        forcefield = ForceField(ff_loc, 'tip3p.xml')
        [templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(pdb.topology)
        for n, template in enumerate(templates):
            residue = residues[n]
            atom_names = []
            for atom in template.atoms:
                if residue.name in ['EXT', 'OTH', 'MTH']:
                    atom.type = 'OTHER-' + atom.element.symbol
                else:
                    atom.type = residue.name + "-" + atom.name.upper()
                atom_names.append(atom.name)

            # Register the template with the forcefield.
            template.name += str(n)
            forcefield.registerResidueTemplate(template)


        integrator = get_integrator(options)
        qm_fixed_atoms, qm_origin_atoms = parse_idx(args.idx, pdb.topology)

        qm_sphere_atoms = get_qm_spheres(qm_origin_atoms, qm_fixed_atoms, options.qm_mm_radius/angstroms, pdb.getPositions()/angstrom, pdb.topology)
        qm_atoms = qm_fixed_atoms + qm_sphere_atoms
        system = forcefield.createSystem(pdb.topology, rigidWater=False)

        #   re-map nonbonded forces so QM only interacts with MM through vdW
        charges = add_nonbonded_force(qm_atoms, system, pdb.topology.bonds(), outfile=outfile)

        #   "external" force for updating QM forces and MM electrostatics
        ext_force = add_ext_force_all(system, charges)

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

        #   debug only: turns off forces except one
        if False:
            while system.getNumForces() > 1:
                for i, force in enumerate(system.getForces()):
                    if not isinstance(force, CustomNonbondedForce):
                        system.removeForce(i)
                        break
        
        #   add pulling force
        if False:
            pull_force = add_ext_force_all(pdb.positions, system)

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

        #   turn on to freeze mm atoms in place
        if False:
            for n in range(system.getNumParticles()):
                if n not in qm_atoms:
                    print("FREEZE: ", n)
                    system.setParticleMass(n, 0*dalton)

        #   remove bonded forces between QM and MM system
        adjust_forces(system, simulation.context, pdb.topology, qm_atoms, outfile=outfile)

        #   output files and reporters
        stats_reporter = StatsReporter('stats.txt', 1, options, qm_atoms=qm_atoms, vel_file_loc=args.repv, force_file_loc=args.repf)
        simulation.reporters.append(HDF5Reporter('output.h5', 1))
        qm_atoms_reporter = QMatomsReporter('qm_atoms.txt')

        #   set up files
        scratch = os.path.join(os.path.curdir, 'qm_mm_scratch/')
        os.makedirs(scratch, exist_ok=True)
        os.makedirs(qc_scratch, exist_ok=True)
        atoms = list(pdb.topology.atoms())
        elements = [x.element.symbol for x in atoms]

        #   test to make sure that all qm_forces are fine
        print_initial_forces(simulation, qm_atoms, pdb.topology, outfile)

        if options.jobtype == 'opt':
            opt = GradientMethod(options.time_step*0.001)

        if options.jobtype != 'opt' and not args.state:
            print(" Setting initial velocities to temperature of {:5f} K: ".format(options.aimd_temp/kelvin), file=outfile)
            simulation.context.setVelocitiesToTemperature(options.aimd_temp, options.aimd_temp_seed)

        #   for sanity checking
        print(' Integrator: ', type(integrator))
        sys.stdout.flush()

        simulation.saveState('initial_state.xml')

        water_filler = WaterFiller(pdb.topology, forcefield, simulation)

        spin_mult = options.mult
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

            state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
            pos = state.getPositions(True)

            # update QM atom list and water positions
            if n % 10 == 0:
                if True:
                    pos = water_filler.fill_void(pos, qm_atoms, outfile=outfile)

                qm_sphere_atoms = get_qm_spheres(qm_origin_atoms, qm_fixed_atoms, options.qm_mm_radius/angstroms, pos/angstrom, pdb.topology)
                qm_atoms = qm_fixed_atoms + qm_sphere_atoms
                qm_atoms = update_mm_forces(qm_atoms, system, simulation.context, pos, pdb.topology, outfile=outfile)

            if len(qm_atoms) > 0:

                qm_energy, qm_gradient = get_qm_force(pos/angstrom, charges, elements, qm_atoms, outfile, pdb.topology, options, rem_lines=rem_lines, step_number=n, outfile=outfile, total_chg=options.charge, spin_mult=options.mult)
                update_ext_force(simulation, qm_atoms, qm_gradient, ext_force, pos/nanometers, charges, qm_energy=qm_energy, outfile=outfile)


            #   additional force updates
            if options.ratchet_pawl:
                update_rachet_pawl_force(ratchet_pawl_force, simulation.context, pos/nanometers, outfile=outfile)
            print(options.oxy_bound)
            if options.oxy_bound:
                oxygen_force.update(simulation.context, pos, outfile=outfile)
            stats_reporter.report(simulation, qm_atoms)


            #   call reporter before taking a step. OpenMM calls reporters after taking a step, but this
            #   will not report the correct force and energy as the current parameters are only valid 
            #   for the current positions, not after
            qm_atoms_reporter.report(simulation, qm_atoms)
            
            if n % 10  == 0:
                simulation.saveState('simulation.xml')

            if options.jobtype == 'opt':
                opt.step(simulation, outfile=outfile)


            simulation.step(1)


    return simulation
              
if __name__ == "__main__":

    simulation = main(sys.argv[1:])

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
        


