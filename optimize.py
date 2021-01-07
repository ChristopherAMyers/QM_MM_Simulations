import numpy as np
from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm import * #pylint: disable=unused-wildcard-import
from simtk.unit import * #pylint: disable=unused-wildcard-import
from sys import stdout

# pylint: disable=no-member
import simtk.unit as unit
picosecond = picoseconds = unit.picosecond
nanometer = nanometers = unit.nanometer
femtoseconds = unit.femtoseconds
# pylint: enable=no-member


class GradientMethod(object):
    def __init__(self, stepsize, outfile_loc='opt.pdb'):
        self._stepsize = stepsize / picosecond
        self._force_old = None
        self._step_old = None
        self._energy_old = None
        self._pos_old = None
        self._initialed = False
        self._max_step_size = 0.02 # nanometers
        #self._pdb_file = open(outfile_loc, 'w')
        self._pdb_file_loc = outfile_loc

    #def __del__(self):
    #    self._pdb_file.close()

    def step(self, simulation, outfile=stdout):
        
        dim = simulation.topology.getNumAtoms()
        state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
        energy = state.getPotentialEnergy()
        forces = state.getForces(True) / (kilojoules_per_mole / nanometer)
        pos = state.getPositions(True).reshape(dim*3) / nanometer
        max_force_mag = np.max(np.linalg.norm(forces, axis=1))
        forces = forces.reshape(dim*3)

        outfile.flush()
        print(' -----------------------------------------------------------', file=outfile)
        print(" Performing Conjugate-Gradient Step ", file=outfile)
        print(" Force Factor: ", 1/np.sqrt(np.sum(forces**2)))
        print(" Maximum force magnitude on atom: ", max_force_mag, file=outfile)
        print(" Energy old:        ", self._energy_old, file=outfile)
        print(" Energy new:        ", energy, file=outfile)
        if self._energy_old != None:
            print(" Energy difference: ", energy - self._energy_old, file=outfile)
    
        if not self._initialed:
            #   if first step, make sure energy conditions passes
            self._energy_old = energy + 100 * kilojoules_per_mole
            step = forces
            self._initialed = True
            beta = 0.0
            print(" Polak-Ribiere step factor: ", beta, file=outfile)
        else:
            #   Polak-Ribiere step
            force_diff = forces - self._force_old
            denom = np.dot(self._force_old, self._force_old)
            numer = np.dot(forces, force_diff)
            beta = numer/denom
            beta = np.max([0.0, beta])
            step = forces +  beta * self._step_old
            print(" Polak-Ribiere step factor: ", beta, file=outfile)
        
        #   last step was successfull, try increased stepsize
        if energy < self._energy_old:
            print(" Energy condition passed ", file=outfile)
            self._stepsize = self._stepsize * 1.2
            self._step_old = step
            self._pos_old = pos
            self._force_old = forces
            self._energy_old = energy

            #   shrink stepsize to ensure max_step_size is enforced
            max_step_length = np.max(np.sqrt(np.dot(step, step))) * self._stepsize
            if max_step_length > self._max_step_size:
                shrink = (self._max_step_size / max_step_length)
                self._stepsize = self._stepsize * shrink
                print(" Max step size exceded, shrinking stepsize by {:.5f}".format(shrink), file=outfile)

            PDBFile.writeFile(simulation.topology, pos.reshape((dim, 3))*nanometers, file=open(self._pdb_file_loc, 'w'))
            new_pos = pos + self._stepsize * step
        #   shrink stepsize if last step was too big and retry step
        else:
            print(" Energy condition failed ", file=outfile)
            print(" Shrinking stepsize by 0.5", file=outfile)
            self._stepsize = self._stepsize * 0.5
            new_pos = self._pos_old + self._stepsize * self._step_old
        print(" Using stepsize of {:15.5E}".format(self._stepsize), file=outfile)

        print(' -----------------------------------------------------------', file=outfile)
        outfile.flush()
        simulation.context.setPositions(new_pos.reshape(dim, 3) * nanometers)
        simulation.currentStep += 1






