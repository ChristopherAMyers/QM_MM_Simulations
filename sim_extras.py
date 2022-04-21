from openmm.app import * #pylint: disable=unused-wildcard-import
from openmm import * #pylint: disable=unused-wildcard-import
from openmm.openmm import * #pylint: disable=unused-wildcard-import
from openmm.unit import * #pylint: disable=unused-wildcard-import
from os import urandom
import numpy as np
import time
import copy
import openmm.openmm as mm
import openmm.unit as unit

# pylint: disable=no-member
picoseconds = unit.picoseconds
picosecond = picoseconds
nanometer = nanometers = unit.nanometer
femtoseconds = unit.femtoseconds
# pylint: enable=no-member

class QMatomsReporter(object):
    """
    Prints out the current lsit of QM atoms to file
    This reporter is NOT suited as an OpenMM reporter
    """
    def __init__(self, file, topology):
        self._out = open(file, 'w')
        self._atoms = list(topology.atoms())

    def __del__(self):
        self._out.close()

    def report(self, simulation, qm_atoms):
        ids = []
        for atom in simulation.topology.atoms():
        #for atom in self._atoms:
            if atom.index in qm_atoms:
                ids.append(int(atom.id))
        ids = sorted(ids)
        self._out.write(" Step {:d}: ".format(simulation.currentStep))
        for id in ids:
            self._out.write('{:d} '.format(id))
        self._out.write('\n')
        self._out.flush()
     


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
        self._last_pos = np.array([])
        self._increased_step_size = False
        self._smallest_force = 1E8
        

        if vel_file_loc:
            self._vel_file = open(vel_file_loc, 'w')
        if force_file_loc:
            self._force_file = open(force_file_loc, 'w')
        
        #   aimd and optimization jobs have diffeent stats
        if self._jobtype == 'aimd':
            self._out.write(' Step Pot-Energy QM-Energy MM-Temp(K) QM-Temp(K) All-Temp(K) Time-Elapsed(s) Time-Remaining qm_atoms\n')
        else:
            self._out.write(' Step Pot-Energy QM-Energy Time-Elapsed(s) Time-Remaining Time-Step Max-Disp RMS-Forces  Max-Forces  Max-ID\n')


        self._masses = None
        self._fixed_dof = 0
        self._constraint_idx = np.empty((0, 2))
        self._num_qm_constraints = 0

        #   TODO: recycle already generated states
        self._previous_state = None

    def __del__(self):
        self._out.close()

        if self._vel_file:
            self._vel_file.close()
        if self._force_file:
            self._force_file.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, True, True, True, None)

    def report(self, simulation, qm_atoms, state=None):

        #   if called outside of OpenMM
        if not state:
            state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
            if not (self._reportInterval - simulation.currentStep%self._reportInterval):
                return

        #qm_atoms = self._qm_atoms
        qm_energy = simulation.context.getParameter('qm_energy')
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

        system = simulation.context.getSystem()
        masses = np.array([system.getParticleMass(n)/dalton for n in range(system.getNumParticles())])

        #   aimd and optimization jobs have diffeent stats
        if self._jobtype == 'aimd':
            #   calculate kinetic energies
            
            vel = state.getVelocities(True).in_units_of(meters/second)
            v2 = np.linalg.norm(vel, axis=1)**2
            masses = masses * 1.67377E-27
            k = 1.380649E-23
            mm_atoms = []
            for x in range(len(v2)):
                if x not in qm_atoms:
                    mm_atoms.append(x)

            #   calculate temperatures from kinetic energies
            # temp_mm = np.sum(v2[mm_atoms]*masses[mm_atoms])/(len(mm_atoms) * k * 3)
            # if len(qm_atoms) > 0:
            #     temp_qm = np.sum(v2[qm_atoms]*masses[qm_atoms])/(len(qm_atoms) * k * 3)
            # else:
            #     temp_qm = 0.0
            # temp_all = np.sum(v2*masses)/(len(v2) * k * 3)

            temp_mm, temp_qm, temp_all = self._calc_temperature(simulation, state, qm_atoms)

            #   rescale temperature if too high
            if temp_all > (100 + self._options.aimd_temp/kelvin) and False:
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

            self._out.write(' {:4d}  {:10.1f}  {:10.3f}  {:7.2f}  {:7.2f}  {:7.2f}  {:10.1f}  {:10s}  {:3d}\n'
                .format(step, pot_energy, qm_energy, temp_mm, temp_qm, temp_all, elapsed_seconds, time_rem, len(qm_atoms)))

        else:
            forces = state.getForces(True)
            force_norms = np.linalg.norm(forces, axis=1)
            rms_forces = np.sqrt(np.mean(force_norms**2))
            max_forces = np.max(force_norms)
            max_force_idx = np.argmax(force_norms)
            max_force_id = int(list(simulation.topology.atoms())[max_force_idx].id)
            self._smallest_force = min(self._smallest_force, max_forces)

            max_disp = 0.0
            pos = state.getPositions(True)/angstroms
            if len(self._last_pos) > 0:
                max_disp = np.max(np.linalg.norm(pos - self._last_pos, axis=1))

            if self._jobtype == 'friction':
                if max_forces > 2500:
                    simulation.integrator.setGlobalVariable(0, 0.5)
                    stepsize = min(np.sqrt(0.005/max_forces), 0.003)*picoseconds
                    #stepsize = min(0.01/max_forces, 0.000003)*picoseconds
                    simulation.integrator.setStepSize(stepsize)
                else:
                    simulation.integrator.setGlobalVariable(0, 0.9)
                    simulation.integrator.setStepSize(self._options.time_step)
                    #simulation.integrator.setStepSize(0.000004)
                    self._increased_step_size = True

            self._last_pos = pos
            stepsize = simulation.integrator.getStepSize()
            self._out.write(' {:4d}  {:10.1f}  {:10.1f}  {:10.1f}  {:10s} {:11.6g} {:10.6f}  {:10.3f}  {:10.3f}  {:4d}\n'
                .format(step, pot_energy, qm_energy, elapsed_seconds, time_rem, stepsize/picoseconds, max_disp, rms_forces, max_forces, max_force_id))

        self._out.flush()

    def _calc_temperature(self, simulation, state, qm_atoms):
        system = simulation.system
        qm_atoms = np.array(qm_atoms, dtype=int)

        # Compute the number of degrees of freedom.
        if self._masses is None:
            self._masses = np.array([system.getParticleMass(n)/dalton for n in range(system.getNumParticles())])
            self._masses = self._masses * 1.67377E-27
            for i in range(system.getNumParticles()):
                if self._masses[i] > 0:
                    self._fixed_dof += 3
            if any(type(system.getForce(i)) == mm.CMMotionRemover for i in range(system.getNumForces())):
                self._fixed_dof -= 3
            for i in range(system.getNumConstraints()):
                params = system.getConstraintParameters(i)
                self._constraint_idx = np.append(self._constraint_idx, [params[:2]], axis=0)

            #   get number of constraints associated with QM atoms only
            num_total_constraints = system.getNumConstraints()
            self._num_qm_constraints = 0
            for n in range(num_total_constraints):
                p1, p2, dist = system.getConstraintParameters(n)
                if p1 in qm_atoms or p2 in qm_atoms:
                    self._num_qm_constraints += 1


        dof = self._fixed_dof
        #   TODO: get the constraints associated with the qm_atoms, not the entire system
        #dof -= self._num_qm_constraints
        dof -= system.getNumConstraints()
        self._dof = dof

        is_qm_1 = np.zeros(len(self._constraint_idx), dtype=int)
        is_qm_2 = np.zeros(len(self._constraint_idx), dtype=int)
        for atom in qm_atoms:
            is_qm_1 += self._constraint_idx[:, 0] == atom
            is_qm_2 += self._constraint_idx[:, 1] == atom
        n_qm_constr = np.sum(is_qm_1 * is_qm_2)
        dof_qm = np.sum(self._masses[qm_atoms] != 0)*3
        dof_qm -= n_qm_constr
        dof_mm = dof - dof_qm

        #   total system temperature
        #   Note that this might be a half timestep off based on the integrator,
        #   so we use the real kinetic energy for total system only
        vel = state.getVelocities(True).in_units_of(meters/second)
        v2 = np.linalg.norm(vel, axis=1)**2
        k = 1.380649E-23
        kinetic_all = 0.5 * np.sum(v2*self._masses)
        temp_all = 2 * kinetic_all/(dof * k)
        temp_all = (2*state.getKineticEnergy()/(dof*unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin)

        #   calculate QM temperatures
        if len(qm_atoms) > 0:
            kinetic_qm = 0.5 * np.sum(v2[qm_atoms]*self._masses[qm_atoms])
            temp_qm = 2 * kinetic_qm / (dof_qm * k)
        else:
            kinetic_qm = 0.0
            temp_qm = 0.0

        #   remaining is MM temperature
        kinetic_mm = kinetic_all - kinetic_qm
        temp_mm = 2 * kinetic_mm / (dof_mm * k)
    
        return temp_mm, temp_qm, temp_all



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
        self.constrain_hbonds = False

        #   32-bit random number seed shifted to c++ min/max integer limits
        self.aimd_temp_seed = int(urandom(4).hex(), 16) - 2147483647
        self.aimd_langevin_seed = int(urandom(4).hex(), 16) - 2147483647

        #   adaptive QM atoms
        self.qm_mm_model = 'oniom'
        self.qm_mm_radius = 3.0 * angstroms
        self.qm_mm_update = True
        self.qm_mm_update_freq = 10
        
        #   ratchet and pawl force options
        self.ratchet_pawl = False
        self.ratchet_pawl_force = 0.0
        self.ratchet_pawl_half_dist = 1000.0
        self.ratchet_pawl_switch = 'exp'

        #   simulated annealing options
        self.annealing = False
        self.annealing_peak = 400 * kelvin
        self.annealing_period = 5000 * femtoseconds

        #   oxygen boundry force
        self.oxy_bound = False
        self.oxy_bound_force = 0.0
        self.oxy_bound_dist = 0.4 * nanometers

        #   charge and multiplicity
        self.charge = 0
        self.mult = 1
        self.adapt_mult = False
        self.mc_spin = False
        self.mc_spin_max_mult = 100
        self.mc_spin_min_mult = 0

        #   oxygen-oxygen repulsion force
        self.oxy_repel = False
        self.oxy_repel_dist = 0.25 * nanometers

        #   rangom gaussian kicks
        self.random_kicks = False
        self.random_kicks_scale = 0.20

        #   initial ionization
        self.ionization = False
        self.ionization_num = 1

        #   restraint force
        self.restraints = False
        self.restraints_switch_time = 0*femtoseconds

        #   point restraint force
        self.point_restraints = False

        #   centroid restraint force
        self.cent_restraints = False

        #   constraints
        self.constrain_qmmm_bonds = True

        #   QM fragments
        self.qm_fragments = False

        #   link atoms
        self.link_atoms = False

        #   force field files
        self.force_field_files = []

        #   external scripts to run before system is created
        self.script_file = None

        #   print options
        self.traj_out_freq = 1