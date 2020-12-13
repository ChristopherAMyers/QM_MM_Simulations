#from .. import qm_mm #pylint: disable=relative-beyond-top-level
import sys
import os
import qm_mm
import numpy as np
from simtk.openmm.app import PDBFile
from simtk.unit import *

# pylint: disable=no-member
import simtk.unit as unit
picosecond = picoseconds = unit.picosecond
nanometer = nanometers = unit.nanometer
femtoseconds = unit.femtoseconds
# pylint: enable=no-member

def bool_to_str(x):
    if x:
        return "True"
    else:
        return "False"

def bool_to_pass(x):
    if x:
        return "PASS"
    else:
        return "FAIL"

def run_aimd_test(outfile=sys.stdout):
    orig_dir = os.path.abspath(os.curdir)
    os.chdir('aimd')

    #   call QM_MM program
    arg_string = '-pdb opt.pdb -idx id_list -rem rem_aimd -repf force.txt -repv vel.txt'
    #qm_mm.main(arg_string.split())

    #   load outputs and reference values
    ref_force = np.loadtxt('ref_force.txt')
    ref_vel = np.loadtxt('ref_vel.txt')
    ref_pos = np.loadtxt('ref_pos.txt')
    forces = np.loadtxt('force.txt')
    vels = np.loadtxt('vel.txt')
    pos = PDBFile('output.pdb').getPositions(True)/nanometer

    #   compare to reference data
    max_vel = np.max(np.linalg.norm(ref_force, axis=1))
    max_vel_diff = np.max(np.linalg.norm(vels - ref_vel, axis=1))
    max_force = np.max(np.linalg.norm(forces, axis=1))
    max_force_diff = np.max(np.linalg.norm(forces - ref_force, axis=1))
    max_pos_diff = np.max(np.linalg.norm(pos - ref_pos, axis=1))

    pass_vel = True
    pass_force = True
    pass_pos = True
    #if max_vel_diff / max_vel > 0.0001:
    #    pass_vel = False
    if max_force_diff / max_force > 0.0001:
        pass_force = False
    if max_pos_diff > 0.01:
        pass_pos = False

    #print("     Maximum velocity deviation:    {:10.5e}  {:s}".format(max_vel_diff, bool_to_pass(pass_vel)), file=outfile)
    print("     Maximum force deviation:       {:10.5e}  {:s}".format(max_force_diff, bool_to_pass(pass_force)), file=outfile)
    print("     Maximum position deviation:    {:10.5e}  {:s}".format(max_pos_diff, bool_to_pass(pass_pos)), file=outfile) 

    os.chdir(orig_dir)
    return pass_force*pass_vel*pass_pos

if __name__ == "__main__":

    print(" Running Tests")
    with open('test_results.txt', 'w') as outfile:

        print(' Running AIMD with Q-Chem test', file=outfile)
        result = run_aimd_test(outfile)
        print('     AIMD overall test Result: {:s}'.format(bool_to_pass(result)), file=outfile)