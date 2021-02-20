import copy
import numpy as np
import os
import time
import shutil
from subprocess import run
from simtk.unit import *
from spin_mult import determine_mult_from_coords
from multiprocessing import cpu_count
from qm_fragments import QM_Fragments

# pylint: disable=no-member
import simtk.unit as unit
picosecond = picoseconds = unit.picosecond
nanometer = nanometers = unit.nanometer
femtoseconds = unit.femtoseconds
# pylint: enable=no-member

class QChemRunner():
    def __init__(self, rem_lines, topology, charges, options, outfile, scratch, fragments_file=None):

        self._rem_lines = rem_lines
        self._topology = topology
        self._options = copy.copy(options)
        self._outfile = outfile
        self._charges = copy.copy(charges)
        self._elements = [x.element.symbol for x in topology.atoms()]
        self._scratch = scratch

        if fragments_file:
            self._use_qm_fragments = True
            self._qm_fragments = QM_Fragments(fragments_file, self._topology)
        else:
            self._use_qm_fragments = False
            self._qm_fragments = None

        #   make sure Q-Chem is available, exit otherwise
        if 'QC' in os.environ:
            self._qchem_path = os.environ.get('QC')
            self._qc_scratch = os.environ.get('QCSCRATCH')
            print(" QC set as ", self._qchem_path)
            print(" QCSCRATCH set as ", self._qc_scratch)
        else:
            print(" Error: environment variable QC not defined. Cannot find Q-Chem directory")
            exit()

        #   set available CPU resources
        if 'SLURM_NTASKS' in os.environ.keys():
            self._n_procs = int(os.environ['SLURM_NTASKS'])
        else:
            #   if not running a slurm job, use number of cores
            self._n_procs = cpu_count()

    def _create_qc_input(self, coords, qm_atoms, rem_lines, total_chg, spin_mult=1, scf_read=True, ghost_atoms=[], jobtype=None):
        input_file_loc = os.path.join(self._scratch, 'input')
        with open(input_file_loc, 'w') as file:
            #   copy over rem job lines
            file.write('$rem \n')
            for line in rem_lines:
                if line[-1] != '\n':
                    line = line + '\n'
                file.write(line)

            #   use the previous jobs orbitals as an initial guess
            if scf_read:
                file.write('    scf_guess read \n')
            if not jobtype:
                file.write('    jobtype     force \n')
                file.write('    qm_mm       true \n')
            else:
                file.write('    jobtype     {:s} \n'.format(jobtype))
            file.write('    sym_ignore  true \n')
            file.write('$end \n\n')

            #   write additional q-chem options, if present
            other_file = os.path.join(self._scratch, 'other')
            if os.path.isfile(other_file):
                with open(other_file, 'r') as other_file:
                    for line in other_file.readlines():
                        file.write(line)

            #   mm_atoms are represented as external charges
            mol_lines = []
            chg_lines = []
            for n, coord in enumerate(coords):
                if n in qm_atoms:
                    mol_lines.append('    {:2s}  {:15.8f}  {:15.8f}  {:15.8f} \n'
                    .format(self._elements[n], coord[0], coord[1], coord[2]))
                elif n in ghost_atoms:
                    mol_lines.append('    {:2s}  {:15.8f}  {:15.8f}  {:15.8f} \n'
                    .format('@H', coord[0], coord[1], coord[2]))
                else:
                    chg_lines.append('    {:15.8f}  {:15.8f}  {:15.8f}  {:15.8f} \n'
                    .format(coord[0], coord[1], coord[2], self._charges[n]))
            

            #   write molecule section
            file.write('$molecule \n')
            file.write('    {:d}  {:d} \n'.format(int(total_chg), spin_mult))
            if self._use_qm_fragments:
                mol_lines = self._qm_fragments.convert_molecule_lines(total_chg, spin_mult, mol_lines, qm_atoms)
            for line in mol_lines:
                file.write(line)
            file.write('$end \n\n')

            #   write external charges
            file.write('$external_charges \n')
            for line in chg_lines:
                file.write(line)
            file.write('$end \n\n')

            return input_file_loc

    def _calc_qm_force(self, coords, qm_atoms, step_number=0, copy_input=False, spin_mult=1):
        outfile = self._outfile
        redo = True
        failures = 0
        use_rem_lines = copy.copy(self._rem_lines)
        while redo:
            outfile.flush()

            scf_read = True
            if failures > 0 or (step_number % 10 == 0):
                scf_read = False

            input_file_loc = self._create_qc_input(coords, qm_atoms, use_rem_lines, self._options.total_charge, spin_mult=spin_mult, scf_read=scf_read)
            output_file_loc = os.path.join(self._scratch, 'output')
            cmd = os.path.join(self._qchem_path, 'bin/qchem') + ' -save -nt {:d}  {:s}  {:s} save_files'.format(self._n_procs, input_file_loc, output_file_loc)

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
                shutil.copyfile(os.path.join(self._qc_scratch, 'save_files/output'), output_file_loc)
            else:
                run(cmd.split())
            end = time.time()
            #   forward Q-Chem output to output file and search for completion
            found_thank_you = False
            with open(output_file_loc, 'r', errors='ignore') as file:
                for line in file.readlines():
                    if "Thank you" in line:
                        found_thank_you = True
                    outfile.write(line)

            if not found_thank_you:
                failures += 1

            #   if SCF failed, try once more with different algorithm 
            if failures == 1 and not found_thank_you:
                redo = True
                if spin_mult == 1:
                    use_rem_lines.append('scf_algorithm RCA')
                    use_rem_lines.append('unrestricted False')
                else:
                    use_rem_lines.append('scf_algorithm DIIS_GDM')
            elif found_thank_you or failures > 1:
                redo = False

        gradient = []
        energy = 0.0
        grad_file_loc = os.path.join(self._qc_scratch, 'save_files/GRAD')
        bohr_to_nm = bohrs.conversion_factor_to(nanometer)
        
        if os.path.isfile(grad_file_loc):
            #shutil.copyfile(grad_file_loc, os.path.join(scratch, 'GRAD'))
            energy = np.loadtxt(grad_file_loc, skiprows=1, max_rows=1)
            gradient = np.loadtxt(grad_file_loc, skiprows=3, max_rows=len(qm_atoms))

            if self._use_qm_fragments:
                frag_gradient = np.copy(gradient)
                for n, idx in enumerate(self._qm_fragments.gradient_order):
                    gradient[n] = frag_gradient[idx]

            #   convert gradient to kJ/mol/nm and energy to kJ/mol
            gradient = (gradient * 2625.5009 * kilojoules_per_mole / nanometer / bohr_to_nm)
            energy = energy * 2625.5009 * kilojoules_per_mole

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
                u = grad.unit
                outfile.write('    {:3d}  {:2s}  {:13.2f}  {:13.2f}  {:13.2f} \n'
                .format(n + 1, self._elements[qm_atoms[n]], grad[0]/u, grad[1]/u, grad[2]/u))
            outfile.write(' -----------------------------------------------------------\n')
        else:
            outfile.write(' Gradient fle NOT found \n')
            outfile.write('      TERMINATING PROGRAM \n')
            outfile.write(' -----------------------------------------------------------\n')
            exit()

        #   debug only: save all generated input files
        #shutil.copyfile(os.path.join(scratch, 'input'), os.path.join(scratch, 'input_{:d}'.format(step_number)))


        return (energy, gradient)

    def get_qm_force(self, coords, qm_atoms, step_number, copy_input=False):
        outfile = self._outfile
        spin_mult = self._options.spin_mult


        qm_energy, qm_gradient = self._calc_qm_force(coords, qm_atoms, step_number, copy_input, spin_mult)

        #   adaptive spin multiplicity for O2
        if (self._options.adapt_mult or self._options.mc_spin) and (step_number % self._options.qm_mm_update_freq == 0 and self._options.qm_mm_update):
            #   spin mult based on number of O2 molecules
            if self._options.adapt_mult:
                new_mult = determine_mult_from_coords(coords*10*nanometers, self._topology, spin_mult, qm_atoms)
                if new_mult != spin_mult:
                    new_energy, new_gradient = self._calc_qm_force(coords, qm_atoms, step_number, copy_input, new_mult)

                if new_energy < qm_energy:
                    outfile.write("\n Step {:d}: Changing spin multiplicity from {:d} to {:d} \n".format(step_number, spin_mult, new_mult))
                    outfile.write(" Energy difference: {:.5f} kJ/mol \n\n".format((new_energy - qm_energy)/kilojoules_per_mole))
                    qm_energy = new_energy
                    qm_gradient = new_gradient
                    #   change spin mult of system to the new multiplicity
                    self._options.mult = new_mult

            #   marcov chain spin flip
            elif self._options.mc_spin:
                up_or_down = np.random.randint(0, 2)
                if up_or_down == 0:
                    new_mult = spin_mult - 2
                else:
                    new_mult = spin_mult + 2
                new_mult = max(1, new_mult)
                

                new_energy, new_gradient = self._calc_qm_force(coords, qm_atoms, step_number, copy_input, new_mult)
                energy_diff = (new_energy - qm_energy)/kilojoules_per_mole

                print(" Attempted new spin multiplicity at step {:d}".format(step_number), file=outfile)
                print(" Old spin multiplicity: {:d}".format(spin_mult), file=outfile)
                print(" New spin multiplicity: {:d}".format(new_mult), file=outfile)
                print(" Energy difference: {:15.5f} kJ/mol".format(energy_diff), file=outfile)


                accept = False
                if energy_diff < 0:
                    print(" Accepted new multiplicity", file=outfile)
                    accept = True
                else:
                    y = np.random.rand()
                    kT = self._options.aimd_temp/kelvin * 0.00831446261815324
                    if y < np.exp(-energy_diff/kT):
                        accept = True
                        print(" Accepted new multiplicity from Boltzmann probability", file=outfile)
                    else:
                        print(" Denied new multiplicity from Boltzmann probability", file=outfile)
                
                if accept:
                    qm_energy = new_energy
                    qm_gradient = new_gradient
                    self._options.mult = new_mult

        return qm_energy, qm_gradient