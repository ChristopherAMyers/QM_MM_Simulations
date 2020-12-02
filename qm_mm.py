from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm import *
from openmmtools.integrators import VelocityVerletIntegrator
from simtk.openmm.openmm import *
from simtk.unit import *
import argparse
import numpy as np
import sys
import itertools
from multiprocessing import cpu_count
from subprocess import run
import time
import shutil
from scipy import optimize
from optimize import GradientMethod

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

qchem_path = '/network/rit/lab/ChenRNALab/bin/Q-Chem5.2-GPU'
scratch = '/tmp'
qc_scratch = '/tmp'
job_name = ''
n_procs = cpu_count()


def parse_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-pdb', help='pdb molecule file to use')
    parser.add_argument('-rem', help='rem arguments to use with Q-Chem')
    parser.add_argument('-idx', help='list of atoms to treat as QM')
    parser.add_argument('-out', help='output file', default='output.txt')
    return parser.parse_args()

def parse_idx(idx_file_loc):
    idx_list = []
    with open(idx_file_loc, 'r') as file:
        for line in file.readlines():
            sp = line.split()
            #   assume that just a column of numbers is used
            if len(sp) == 1:
                idx_list.append(int(sp[0]) - 1)
            #   assume that the output from pymol is used
            elif len(sp) == 2:
                idx = sp[-1].split('`')[-1].split(')')[0]
                idx_list.append(int(idx) - 1)
            else:
                print("ERROR: Can't determin index format")
    idx_list = sorted(idx_list)
    return idx_list

def find_all_qm_atoms(mat_idx_list, bondedToAtom, topology):
    qm_idx_list = []
    atoms = list(topology.atoms())
    for idx in mat_idx_list:
        qm_idx_list.append(idx)
        for bonded_to in bondedToAtom[idx]:
            if atoms[bonded_to].element.symbol not in ['Se', 'Zn']:
                pass

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
    for force in forces:
        if  isinstance(force, HarmonicBondForce):
            for n in range(force.getNumBonds()):
                a, b, r, k = force.getBondParameters(n)
                #force.setBondParameters(n, a, b, r, k*0.000)
                #if 51 in [a, b]:
                #    print(a, b, r, k, atoms[a].name, atoms[b].name, atoms[a].residue.id, atoms[b].residue.id)
                if a in qm_atoms and b in qm_atoms:
                    force.setBondParameters(n, a, b, r, k*0.000)
                    num_bonds_removed += 1
                if 55 in [a, b]:
                    print(a, b, r, k)
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
            print("FOUND")
            #force.setUseDispersionCorrection(False)
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

def add_ext_force(qm_atoms, system):
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

def create_qc_input(coords, charges, elements, qm_atoms, total_chg=0, rem_lines=[], step_number=0, ghost_atoms=[], jobtype=None):
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
        if step_number != 0:
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
        file.write('    {:d}  1 \n'.format(int(total_chg)))
        for line in mol_lines:
            file.write(line)
        file.write('$end \n\n')

        #   write external charges
        file.write('$external_charges \n')
        for line in chg_lines:
            file.write(line)
        file.write('$end \n\n')

        return input_file_loc

def calc_qm_force(coords, charges, elements, qm_atoms, output_file, total_chg=0, rem_lines=[], step_number=0, copy_input=False):
    global scratch, qc_scratch
    redo = True
    failures = 0
    use_rem_lines = copy.copy(rem_lines)
    while redo:
        input_file_loc = create_qc_input(coords, charges, elements, qm_atoms, total_chg=total_chg, rem_lines=use_rem_lines, step_number=step_number)
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
            shutil.copytree('/network/rit/lab/ChenRNALab/awesomeSauce/2d_materials/ZnSe/quant_espres/znse_2x2/qm_mm/multi_solvent/6w_2o_no_ligs/save_files', os.path.join(qc_scratch, 'save_files'))
            shutil.copyfile(os.path.join(qc_scratch, 'save_files/output'), output_file_loc)
        else:
            run(cmd.split())
        end = time.time()

        #   forward Q-Chem output to output file and search for completion
        found_thank_you = False
        found_scf_failure = False
        with open(output_file_loc, 'r') as file:
            for line in file.readlines():
                if "Thank you" in line:
                    found_thank_you = True
                if "SCF failed to converge" in line:
                    found_scf_failure = True
                outfile.write(line)

        #   if SCF failed, try once more with RCA 
        if found_scf_failure and failures == 0:
            redo = True
            failures += 1
            use_rem_lines.append('scf_algorithm RCA')
        elif found_thank_you or failures > 0:
            redo = False
        

    gradient = []
    energy = 0.0
    grad_file_loc = os.path.join(qc_scratch, 'save_files/GRAD')
    if os.path.isfile(grad_file_loc):
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

    #   convert gradient to kJ/mol/nm and energy to kJ/mol
    gradient = (gradient * 2625.5009 * kilojoules_per_mole / nanometer / bohrs.conversion_factor_to(nanometer))
    energy = energy * 2625.5009 * kilojoules_per_mole
    return (energy, gradient)

def update_qm_force(context, gradient, ext_force, qm_coords_in_nm, qm_energy=0.0):
    total = np.sum(qm_coords_in_nm * gradient)
    context.setParameter('k', total / len(qm_atoms))
    context.setParameter('qm_energy', qm_energy/len(qm_atoms))
    print("K: ", total / len(qm_atoms))
    print("QM_ENERGY: ", qm_energy)
    for n in range(ext_force.getNumParticles()):
        idx = ext_force.getParticleParameters(n)[0]
        atom_idx = qm_atoms
        ext_force.setParticleParameters(n, idx, gradient[n])
    ext_force.updateParametersInContext(context)

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
            else:
                rem_lines.append(line)
            #else:
            #    print("ERROR: rem option < %s > is not supported" % option)
            #    print("       Script will now terminate")
            #    exit()

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
    outfile.write(' jobtype:               {:>10s} \n'.format(opts.jobtype) )
    outfile.write(' integrator:            {:>10s} \n'.format(opts.integrator) )
    outfile.write(' time step:             {:>10.2f} fs \n'.format(opts.time_step/femtoseconds) )
    outfile.write(' number of steps:       {:>10d} \n'.format(opts.aimd_steps) )
    if opts.jobtype == 'aimd':
        outfile.write(' temperature:           {:>10.2f} K \n'.format(opts.aimd_temp/kelvin) )
    if opts.aimd_thermostat:
        outfile.write(' thermostat:            {:>10s} \n'.format(opts.aimd_thermostat) )
        outfile.write(' langevin frequency:  1/{:>10.2f} fs \n'.format(opts.aimd_langevin_timescale / femtoseconds) )
    outfile.write('--------------------------------------------\n')
    outfile.flush()
    return rem_lines, opts

def add_nonbonded_force(qm_atoms, system, bonds, outfile=sys.stdout):
    forces = system.getForces()
    forceString = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + 138.935458 * q/r; "
    forceString += "sigma=0.5*(sigma1+sigma2); "
    forceString += "epsilon=sqrt(epsilon1*epsilon2); "
    forceString += "q=q1*q2; "
    customForce = CustomNonbondedForce(forceString)
    customForce.addPerParticleParameter("q")
    customForce.addPerParticleParameter("sigma")
    customForce.addPerParticleParameter("epsilon")

    #   get list of bonds for exclusions
    bond_idx_list = []
    for bond in bonds:
        #if bond.atom1.index == 51 or bond.atom2.index == 51:
        #    print(bond)
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
                    chg = chg*0
                else:
                    mm_atoms.append(n)
                customForce.addParticle([chg, sig, eps])

            system.removeForce(i)
            customForce.addInteractionGroup(qm_atoms, mm_atoms)
            customForce.addInteractionGroup(mm_atoms, mm_atoms)
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
    return charges

def get_integrator(opts):
    if opts.jobtype == 'aimd':
        if opts.integrator.lower() == 'langevin':
            return LangevinIntegrator(opts.aimd_temp, 1/opts.aimd_langevin_timescale, opts.time_step)
        else:
            return VerletIntegrator(opts.time_step)
    elif opts.jobtype == 'grad':
        integrator = CustomIntegrator(opts.time_step)
        integrator.addGlobalVariable("step_size", opts.time_step / nanometers)
        integrator.addUpdateContextState()
        integrator.addConstrainPositions()
        integrator.addComputePerDof("x", "x + step_size*step_size*f")
        return integrator
    else:
        if True:
            integrator = CustomIntegrator(opts.time_step)

            #timestep = 1.0 * unit.femtoseconds
            integrator.addGlobalVariable("step_size", opts.time_step / nanometers)
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

            integrator.addUpdateContextState()
            integrator.addConstrainPositions()

            integrator.addComputeGlobal("energy_new", "energy")
            integrator.addComputeSum("fnorm2", "f^2")
            integrator.addComputePerDof("dir", "f/sqrt(fnorm2 + delta(fnorm2))")
            integrator.addComputeGlobal("delta_energy", "energy_new-energy_old")
            integrator.addComputeGlobal("accept", "step(-delta_energy) * delta(energy - energy_new)")
            integrator.addComputeGlobal("accept", "1")
            #integrator.addComputeGlobal("step_size", "step_size * (1.2*accept + 0.5*(1-accept))")
            integrator.addComputeGlobal("energy_old", "energy*accept + (1-accept)*energy_old")
            integrator.addComputePerDof("dir_old",    "dir*accept + (1-accept)*dir_old")
            integrator.addComputePerDof("x_old", "    x*accept + (1-accept)*x_old")
            integrator.addComputePerDof("x", "accept*(x + step_size*dir) + (1-accept)*(x_old + step_size*dir_old)")

        else:
            integrator = CustomIntegrator(opts.time_step)
            #integrator.addComputePerDof("v", "0.0")
            #integrator.addComputePerDof("x", "x + 0.5*dt*dt*f/m")
            #integrator.addComputePerDof("x", "x + dt*f")
            #return integrator
            """
            Construct a simple gradient descent minimization integrator.
            An adaptive step size is used.
            """
            #timestep = 1.0 * unit.femtoseconds
            integrator.addGlobalVariable("step_size", opts.time_step / nanometers)
            integrator.addGlobalVariable("energy_old", 0)
            integrator.addGlobalVariable("energy_new", 0)
            integrator.addGlobalVariable("delta_energy", 0)
            integrator.addGlobalVariable("accept", 0)
            integrator.addGlobalVariable("fnorm2", 0)
            integrator.addPerDofVariable("x_old", 0)

            integrator.addUpdateContextState()
            integrator.addConstrainPositions()
            # Store old energy and positions.
            integrator.addComputeGlobal("energy_old", "energy")
            integrator.addComputePerDof("x_old", "x")
            # Compute sum of squared norm.
            integrator.addComputeSum("fnorm2", "f^2")
            # Take step.
            integrator.addComputePerDof("x", "x+step_size*f/sqrt(fnorm2 + delta(fnorm2))")
            integrator.addConstrainPositions()
            # Ensure we only keep steps that go downhill in energy.
            integrator.addComputeGlobal("energy_new", "energy")
            integrator.addComputeGlobal("delta_energy", "energy_new-energy_old")
            # Accept also checks for NaN
            integrator.addComputeGlobal("accept", "step(-delta_energy) * delta(energy - energy_new)")
            integrator.addComputePerDof("x", "accept*x + (1-accept)*x_old")

            # Update step size.
            integrator.addComputeGlobal("step_size", "step_size * (1.2*accept + 0.5*(1-accept))")

        return integrator
   
def gen_qchem_opt(options, simulation, charges, elements, qm_atoms, qm_bonds, qm_angles, outfile):
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
    
    exit()

def apply_pull_force(coords, system):
    pull_force = CustomExternalForce('px*x + py*y + pz*z')
    pull_force.addPerParticleParameter('px')
    pull_force.addPerParticleParameter('py')
    pull_force.addPerParticleParameter('pz')
    
    direction = coords[98] - coords[226]
    norm = direction / np.linalg.norm(direction)
    force_mag = 5000
    pull_force.addParticle(98, -norm*force_mag)
    system.addForce(pull_force)
    return pull_force

if __name__ == "__main__":
    args = parse_args()
    with open(args.out, 'w') as outfile:
        rem_lines, options = get_rem_lines(args.rem, outfile)
        pdb = PDBFile(args.pdb)
        pdb_to_qc.add_bonds(pdb)
        data, bondedToAtom = pdb_to_qc.determine_connectivity(pdb.topology)
        ff_loc = '/network/rit/lab/ChenRNALab/awesomeSauce/2d_materials/ZnSe/quant_espres/znse_2x2/qm_mm/forcefield'
        forcefield = ForceField(os.path.join(ff_loc, 'forcefields/forcefield2.xml'), 'tip3p.xml')
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

        integrator = get_integrator(options)
        qm_atoms = parse_idx(args.idx)
        system = forcefield.createSystem(pdb.topology)
        #   re-map nonbonded forces so QM only interacts with MM through vdW
        charges = add_nonbonded_force(qm_atoms, system, pdb.topology.bonds(), outfile=outfile)
        #   "external" force is for updating QM forces
        ext_force = add_ext_force(qm_atoms, system)
        #   add pulling force
        if True:
            pull_force = apply_pull_force(pdb.positions, system)

        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        if False:
            for n in range(system.getNumParticles()):
                if n not in qm_atoms:
                    print("FREEZE: ", n)
                    system.setParticleMass(n, 0*dalton)

        #   remove bonded forces between QM and MM system
        qm_bonds, qm_angles = adjust_forces(system, simulation.context, pdb.topology, qm_atoms, outfile=outfile)
        
        #   output files and reporters
        stats_reporter = StatsReporter('stats.txt', 1, options, qm_atoms=qm_atoms)
        simulation.reporters.append(PDBReporter('output.pdb', 1))
        simulation.reporters.append(stats_reporter)

        scratch = os.path.join(os.path.curdir, 'qm_mm_scratch/')
        #   set up files
        if 'QCSCRATCH' in os.environ:
            qc_scratch = os.environ.get('QCSCRATCH')
            print(" QCSCRATCH set as " + scratch)
        os.makedirs(scratch, exist_ok=True)
        atoms = list(pdb.topology.atoms())
        elements = [x.element.symbol for x in atoms]

        #   test to make sure that all qm_forces are fine
        state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
        forces = state.getForces()
        print(" Initial Potential energy: ", state.getPotentialEnergy(), file=outfile)
        print(" Initial forces exerted on QM atoms: ", file=outfile)
        for n in qm_atoms:
            f = forces[n]/forces[n].unit
            print(" Force on atom {:3d}: {:10.2f}  {:10.2f}  {:10.2f} kJ/mol/nm"
            .format((n+1), f[0], f[1], f[2]), file=outfile)
        print(" Check to make sure that all forces are ~10^3 or less.", file=outfile)
        print(" Larger forces may indicate an inproper force field parameterization.", file=outfile)

        #if options.jobtype == 'opt':
        #    gen_qchem_opt(options, simulation, charges, elements, qm_atoms, qm_bonds, qm_angles, outfile)
        if options.jobtype == 'opt':
            opt = GradientMethod(options.time_step*0.001)

        if options.jobtype != 'opt':
            simulation.context.setVelocitiesToTemperature(options.aimd_temp)

        print(' Integrator: ', integrator)
        sys.stdout.flush()

        #   run simulation
        base_qm_energy = 0.0 * kilojoules_per_mole
        for n in range(options.aimd_steps):
            state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)  
            pos = state.getPositions(True)
            if len(qm_atoms) > 0:
                qm_energy, qm_gradient = calc_qm_force(pos/angstrom, charges, elements, qm_atoms, outfile, rem_lines=rem_lines, step_number=n)
                update_qm_force(simulation.context, qm_gradient, ext_force, pos[qm_atoms]/nanometer, qm_energy=qm_energy)
            
            
            if False:
                state = simulation.context.getState(getEnergy=True, getForces=True)   
                print("Forces Before Step:", file=outfile)
                forces = state.getForces()
                for n in qm_atoms:
                    f = forces[n]/forces[n].unit
                    print(" Force         {:3s}: {:13.2f}  {:13.2f}  {:13.2f} kJ/mol/nm"
                    .format(elements[n], f[0], f[1], f[2]), file=outfile)

            simulation.step(1)
            if options.jobtype == 'opt':
                opt.step(simulation, outfile=outfile)

            if False:
                state = simulation.context.getState(getPositions=True)
                print("Positions After Step:", file=outfile)
                new_pos = state.getPositions(True)
                for n in qm_atoms:
                    f = new_pos[n]/angstroms
                    print(" Position      {:3s}: {:15.8f}  {:15.8f}  {:15.8f} Ang."
                    .format(elements[n], f[0], f[1], f[2]), file=outfile)




            

