from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm import *
from simtk.openmm.openmm import *
from simtk.unit import *
from os import urandom
import numpy as np
import time
import copy

# pylint: disable=no-member
import simtk
picoseconds = simtk.unit.picoseconds
picosecond = picoseconds
nanometer = simtk.unit.nanometer
femtoseconds = simtk.unit.femtoseconds
# pylint: enable=no-member



class StatsReporter(object):
    def __init__(self, file, reportInterval, options, qm_atoms = [], vel_file_loc=None, force_file_loc=None):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self._qm_atoms = copy.copy(qm_atoms)
        self._has_initialted = False
        self._total_steps = options.aimd_steps
        self._init_clock_time = time.time()
        self._init_qm_energy = 0
        self._init_pot_energy = 0
        self._jobtype = options.jobtype
        self._options = copy.copy(options)
        self._vel_file=None
        self._force_file=None
        
        if vel_file_loc:
            self._vel_file = open(vel_file_loc, 'w')
        if force_file_loc:
            self._force_file = open(force_file_loc, 'w')
        
        #   aimd and optimization jobs have diffeent stats
        if self._jobtype == 'aimd':
            self._out.write(' Step Pot-Energy QM-Energy MM-Temp(K) QM-Temp(K) All-Temp(K) Time-Elapsed(s) Time-Remaining \n')
        else:
            self._out.write(' Step Pot-Energy QM-Energy Time-Elapsed(s) Time-Remaining Time-Step RMS-Forces  Max-Forces\n')


    def __del__(self):
        self._out.close()

        if self._vel_file:
            self._vel_file.close()
        if self._force_file:
            self._force_file.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, True, True, True, None)

    def report(self, simulation, state=None):

        #   if called outside of OpenMM
        if not state:
            state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
            if not (self._reportInterval - simulation.currentStep%self._reportInterval):
                return

        qm_atoms = self._qm_atoms
        qm_energy = simulation.context.getParameter('qm_energy')*len(qm_atoms)
        pot_energy = state.getPotentialEnergy()/kilojoules_per_mole
        step = simulation.currentStep
        if not self._has_initialted:
            self._init_steps = step
            self._init_clock_time = time.time()
            self._has_initialted = True
            self._init_qm_energy = qm_energy
            self._init_pot_energy = pot_energy
        elapsed_seconds, time_rem = self._get_remaining_time(simulation)
        qm_energy -= self._init_qm_energy*0
        pot_energy -= self._init_pot_energy*0

        #   aimd and optimization jobs have diffeent stats
        if self._jobtype == 'aimd':
            #   calculate kinetic energies
            system = simulation.context.getSystem()
            vel = state.getVelocities(True).in_units_of(meters/second)
            v2 = np.linalg.norm(vel, axis=1)**2
            masses = np.array([system.getParticleMass(n)/dalton for n in range(len(vel))])*(1.67377E-27)
            k = 1.380649E-23
            mm_atoms = []
            for x in range(len(v2)):
                if x not in qm_atoms:
                    mm_atoms.append(x)

            #   calculate temperatures from kinetic energies
            temp_mm = np.sum(v2[mm_atoms]*masses[mm_atoms])/(len(mm_atoms) * k * 3)
            temp_qm = np.sum(v2[qm_atoms]*masses[qm_atoms])/(len(qm_atoms) * k * 3)
            temp_all = np.sum(v2*masses)/(len(v2) * k * 3)

            #   rescale temperature if too high
            if temp_all > (100 + self._options.aimd_temp/kelvin):
                scale = np.sqrt(self._options.aimd_temp/temp_qm)
                new_vel = vel
                new_vel[qm_atoms] = new_vel[qm_atoms]*scale
                #self._out.write(' Scalling QM velocities by {:8.3f} \n'.format(scale))
                simulation.context.setVelocities(new_vel)

            #   print velocities to file
            if self._vel_file:
                for v in vel:
                    v = v / v.unit
                    self._vel_file.write('{:15.8E}  {:15.8E}  {:15.8E} \n'.format(v[0], v[1], v[2]))
                self._vel_file.flush()

            #   print forces to file
            if self._force_file:
                for f in state.getForces(True):
                    f = f / f.unit
                    self._force_file.write('{:15.8E}  {:15.8E}  {:15.8E} \n'.format(f[0], f[1], f[2]))
                self._force_file.flush()

            self._out.write(' {:4d}  {:10.1f}  {:10.3f}  {:7.2f}  {:7.2f}  {:7.2f}  {:10.1f}  {:10s} \n'
                .format(step, pot_energy, qm_energy, temp_mm, temp_qm, temp_all, elapsed_seconds, time_rem))

        else:
            forces = state.getForces(True)
            force_norms = np.linalg.norm(forces, axis=1)
            rms_forces = np.sqrt(np.mean(force_norms**2))
            max_forces = np.max(force_norms)
            timestep = simulation.integrator.getGlobalVariable(0)
            self._out.write(' {:4d}  {:10.1f}  {:10.3f}  {:10.1f}  {:10s}  {:10.6f}  {:10.3f}  {:10.3f}\n'
                .format(step, pot_energy, qm_energy, elapsed_seconds, time_rem, timestep, rms_forces, max_forces))

        self._out.flush()


    def _get_remaining_time(self, simulation):
        '''Copied from simtk/openmm/app/statedatareporter.py '''
        elapsedSeconds = time.time()-self._init_clock_time
        elapsedSteps = simulation.currentStep-self._init_steps
        if elapsedSteps == 0:
            value = '--'
        else:
            estimatedTotalSeconds = (self._total_steps-self._init_steps)*elapsedSeconds/elapsedSteps
            remainingSeconds = int(estimatedTotalSeconds-elapsedSeconds)
            remainingDays = remainingSeconds//86400
            remainingSeconds -= remainingDays*86400
            remainingHours = remainingSeconds//3600
            remainingSeconds -= remainingHours*3600
            remainingMinutes = remainingSeconds//60
            remainingSeconds -= remainingMinutes*60
            if remainingDays > 0:
                value = "%d:%d:%02d:%02d" % (remainingDays, remainingHours, remainingMinutes, remainingSeconds)
            elif remainingHours > 0:
                value = "%d:%02d:%02d" % (remainingHours, remainingMinutes, remainingSeconds)
            elif remainingMinutes > 0:
                value = "%d:%02d" % (remainingMinutes, remainingSeconds)
            else:
                value = "0:%02d" % remainingSeconds
        return elapsedSeconds, value

class JobOptions(object):
    def __init__(self):
        self.time_step = 1.0 * femtoseconds
        self.jobtype = 'aimd'
        self.aimd_steps = 2000
        self.aimd_temp = 300 * kelvin
        self.aimd_thermostat = None
        self.aimd_langevin_timescale = 100 * femtoseconds
        self.integrator = 'Verlet'
        #   32-bit random number seed shifted to c++ min/max integer limits
        self.aimd_temp_seed = int(urandom(4).hex(), 16) - 2147483647
        self.aimd_langevin_seed = int(urandom(4).hex(), 16) - 2147483647

        self.qm_mm_radius = 5 * angstroms
        self.ratched_pawl = False
        self.ratched_pawl_force = 0
