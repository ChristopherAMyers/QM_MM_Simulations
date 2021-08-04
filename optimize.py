import numpy as np
from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm import * #pylint: disable=unused-wildcard-import
from simtk.unit import * #pylint: disable=unused-wildcard-import
from sys import stdout

from scipy.optimize import minimize

# pylint: disable=no-member
import simtk.unit as unit
picosecond = picoseconds = unit.picosecond
nanometer = nanometers = unit.nanometer
femtoseconds = unit.femtoseconds
# pylint: enable=no-member

class BFGS(object):
    def __init__(self, stepsize, outfile_loc='opt.pdb'):
        self._alpha_k = 0.1
        self._gfk = None
        self._step_dir_old = np.array([])
        self._pk = np.array([])
        self._energy_old = None
        self._xkp1 = None
        self._xk = None
        self._step_number = 0
        self._max_step_length = 0.01 # nanometers
        #self._pdb_file = open(outfile_loc, 'w')
        self._pdb_file_loc = outfile_loc
        self._num_fails = 0
        
        self._hessian = np.array([])
        self._Hk = np.array([])

        if not is_quantity(stepsize):
            print(" Error in BFGS: Stepsize must have time units")
            exit()
        
        self._absolute_max_step_size = (stepsize/femtoseconds)**2 * 0.5

    #def __del__(self):
    #    self._pdb_file.close()


    def _update_step_direction(self, forces):

        if self._step_number == 0:
            return forces

        y = forces - self._gfk
        s = self._step_dir_old
        B = self._hessian
        
        #   update Hessian
        term1 = np.outer(y, y)/np.dot(y, s)
        Bs = B @ s
        term2 = np.outer(Bs, Bs) / (s @ B @ s)
        self._hessian = self._hessian + term1 + term2

        #   step direction
        p = np.linalg.solve(self._hessian, forces)

        return p

    def _reset_hessian(self, dim, outfile=stdout):
        print(" Resetting Hessian", file=outfile)
        self._hessian = np.eye(dim*3, dtype=int)
        self._Hk = np.eye(dim*3, dtype=int)

    def step(self, simulation, outfile=stdout):
        
        dim = simulation.topology.getNumAtoms()
        state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
        qm_energy = simulation.context.getParameter('qm_energy')
        energy = state.getPotentialEnergy() + qm_energy * kilojoules_per_mole
        forces = state.getForces(True) / (kilojoules_per_mole / nanometer)
        self._xkp1 = state.getPositions(True).reshape(dim*3) / nanometer
        max_force_mag = np.max(np.linalg.norm(forces, axis=1))
        gfkp1 = -forces.reshape(dim*3)

        outfile.flush()
        print(' -----------------------------------------------------------', file=outfile)
        print(" Performing BFGS Step {:d}".format(self._step_number), file=outfile)
        #print(" Force Factor: ", 1/np.sqrt(np.sum(forces**2)))
        print(" Maximum force magnitude on atom: ", max_force_mag, file=outfile)
        print(" Energy old:        ", self._energy_old, file=outfile)
        print(" Energy new:        ", energy, file=outfile)
        if self._energy_old != None:
            print(" Energy difference: ", energy - self._energy_old, file=outfile)
    
        accept_step = False
        update_hess = True
        
        if self._step_number == 0:
            #   first step
            print(" First step: taking gradient descent step ", file=outfile)
            self._reset_hessian(dim, outfile)
            accept_step = True
            self._gfk = gfkp1
            update_hess = False

        elif energy < self._energy_old:
            #   energy condition passed
            accept_step = True
            print(" Energy condition passed ", file=outfile)

        elif self._num_fails >= 5:
            #   reset due to too many failed steps
            print(" 5 consecutive failed steps: resetting", file=outfile)
            self._reset_hessian(dim, outfile)
            accept_step = True
            self._gfk = gfkp1
            update_hess = False

        if max_force_mag > 10000:
            print(" Force is too large to use BFGS update", file=outfile)
            self._reset_hessian(dim, outfile)
            update_hess = False

        if accept_step:
            self._num_fails = 0
            if update_hess:
                ### This section was taken from scipy _minimize_bfgs
                ### anaconda3.7/lib/python3.7/site-packages/scipy/optimize/optimize.py
                sk = self._xkp1 - self._xk
                self._xk = self._xkp1
                yk = gfkp1 - self._gfk
                print(" Wolfe Condition 2: ", np.dot(self._pk, gfkp1) / np.dot(self._pk, self._gfk) , file=outfile)
                self._gfk = gfkp1


                try:  # this was handled in numeric, let it remaines for more safety
                    rhok = 1.0 / (np.dot(yk, sk))
                except ZeroDivisionError:
                    rhok = 1000.0
                    print("Divide-by-zero encountered: rhok assumed large", file=outfile)
                if np.isinf(rhok):  # this is patch for numpy
                    rhok = 1000.0
                    print("Divide-by-zero encountered: rhok assumed large", file=outfile)
                I = np.eye(dim*3, dtype=int)
                A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
                A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
                self._Hk = np.dot(A1, np.dot(self._Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                        sk[np.newaxis, :])
                ### end scipy code

            else:
                self._xk = self._xkp1

            #   new step direction
            self._pk = -np.dot(self._Hk, self._gfk)


            #   try an increased stepsize next time
            self._alpha_k *= 1.20

            #   shrink stepsize to ensure max_step_length is enforced
            step_length = np.max(np.sqrt(self._pk *self._pk)) * self._alpha_k
            if step_length > self._max_step_length:
                shrink = (self._max_step_length / step_length)
                self._alpha_k = self._alpha_k * shrink
                print(" Max step length exceded, shrinking stepsize by {:.10f}".format(shrink), file=outfile)
            #PDBFile.writeFile(simulation.topology, self._xkp1.reshape((dim, 3))*nanometers, file=open(self._pdb_file_loc, 'w'))

            self._xkp1 = self._xk + self._alpha_k * self._pk

            self._energy_old = energy

        else:
            #   shrink stepsize if last step was too big and retry step
            print(" Energy condition failed ", file=outfile)
            print(" Shrinking stepsize by 0.5", file=outfile)
            self._alpha_k = self._alpha_k * 0.5
            self._xkp1 = self._xk + self._alpha_k * self._pk
            self._num_fails += 1
        print(" Using stepsize of {:15.5E}".format(self._alpha_k), file=outfile)

        print(' -----------------------------------------------------------', file=outfile)
        outfile.flush()
        simulation.context.setPositions(self._xkp1.reshape(dim, 3) * nanometers)
        simulation.integrator.setStepSize(self._alpha_k)
        self._step_number += 1
        #simulation.currentStep += 1


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
        self._num_fails = 0
        self._max_foce_mag_old = 0
        self._next_pos = None

    #def __del__(self):
    #    self._pdb_file.close()

    def prepare_step(self, simulation, outfile=stdout):
        
        dim = simulation.topology.getNumAtoms()
        state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
        qm_energy = simulation.context.getParameter('qm_energy')
        energy = state.getPotentialEnergy() + qm_energy * kilojoules_per_mole
        forces = state.getForces(True) / (kilojoules_per_mole / nanometer)
        pos = state.getPositions(True).reshape(dim*3) / nanometer
        max_force_mag = np.max(np.linalg.norm(forces, axis=1))
        forces = forces.reshape(dim*3)

        outfile.flush()
        print(' -----------------------------------------------------------', file=outfile)
        print(" Performing Conjugate-Gradient Step %d" % simulation.currentStep, file=outfile)
        #print(" Force Factor: ", 1/np.sqrt(np.sum(forces**2)))
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
            beta = np.max([np.max([0.0, beta]), 4.0])
            beta = 0.0
            step = forces +  beta * self._step_old
            print(" Polak-Ribiere step factor: ", beta, file=outfile)
        
        #   last step was successfull, try increased stepsize
        if energy < self._energy_old or (energy - self._energy_old) < 10*kilojoules_per_mole or self._num_fails >= 50:
            if self._num_fails >= 50:
                print(" Number of fails exceeded. Continuing anyway with fixed stepsize", file=outfile)
                #   decrease stepsize to compensate for later increase
                #self._stepsize = 1.00E-07 
                self._num_fails = 0
            elif energy < self._energy_old:
                print(" Energy condition passed ", file=outfile)
            else:
                print(" Energy condition failed, but continuing b/c diff. is small kJ/mol", file=outfile)

            #   if we achieved a lower force, try a bigger stepsize
            if (max_force_mag - self._max_foce_mag_old)/max_force_mag < 0.05 and (energy - self._energy_old) < 0*kilojoules_per_mole:
                print(" Increasing stepsize by 1.50", file=outfile)
                self._stepsize = self._stepsize * 1.50
            self._max_foce_mag_old = max_force_mag

            self._step_old = step
            self._pos_old = pos
            self._force_old = forces
            self._energy_old = energy

            #   shrink stepsize to ensure max_step_size is enforced
            max_step_length = np.max(np.sqrt(np.dot(step, step))) * self._stepsize
            if max_step_length > self._max_step_size:
                shrink = (self._max_step_size / max_step_length)
                self._stepsize = self._stepsize * shrink
                print(" Max step length exceded, shrinking stepsize by {:.5f}".format(shrink), file=outfile)

            PDBFile.writeFile(simulation.topology, pos.reshape((dim, 3))*nanometers, file=open(self._pdb_file_loc, 'w'))
            new_pos = pos + self._stepsize * step
        #   shrink stepsize if last step was too big and retry step
        else:
            print(" Energy condition failed ", file=outfile)
            print(" Shrinking stepsize by 0.5", file=outfile)
            norm_forces = np.max(np.linalg.norm(self._force_old.reshape(dim, 3), axis=1))
            self._stepsize = min([0.01/norm_forces,  self._stepsize * 0.5])
            print("NEW: ", 0.01/norm_forces, self._stepsize * 0.5)
            #self._stepsize = self._stepsize * 0.5
            new_pos = self._pos_old + self._stepsize * self._step_old
            self._num_fails += 1
        print(" Using stepsize of {:15.5E}".format(self._stepsize), file=outfile)

        print(' -----------------------------------------------------------', file=outfile)
        outfile.flush()
        
        
        #simulation.context.setPositions(new_pos.reshape(dim, 3) * nanometers)
        self._next_pos = new_pos.reshape(dim, 3) * nanometers
        
        simulation.integrator.setStepSize(self._stepsize)
        #simulation.currentStep += 1

    def step(self, simulation):
        simulation.context.setPositions(self._next_pos)


class MMOnlyBFGS(object):
    def __init__(self, context, constraints=None, progress_pdb=None, topology=None, out_freq=1, reporter=(None, None)):
        from scipy import optimize
        self._optimize = optimize
        self._progress_pdb = None
        self._topology = topology
        self._step_num = 0
        self._constraints = constraints
        self._context = context
        if progress_pdb is not None and topology is not None:
            self._progress_pdb = open(progress_pdb, 'w')
        self._out_freq = out_freq


        self._constr_2_idx = {
                'X': [0], 'Y': [1], 'Z': [2],
                'XY': [0, 1], 'XZ': [1,2], 'YZ': [1,2],
                'YX': [0, 1], 'ZX': [1,2], 'ZY': [1,2],
                'XYZ': [0, 1, 2]
            }

        self._reporter, self._reporter_args = reporter

    def _callback(self, pos):
        print("STEP: ", self._step_num)
        if self._progress_pdb is not None and (self._step_num % self._out_freq == 0):
            PDBFile.writeModel(self._topology, pos.reshape(-1,3)*nanometer, file=self._progress_pdb, modelIndex=self._step_num)
        self._step_num += 1

        if self._reporter is not None:
            self._reporter(*self._reporter_args)


    def minimize(self):
        #constraints = dict(zip(np.arange(64), ['Z']*64))

        init_state = self._context.getState(getForces=True, getEnergy=True, getPositions=True)
        init_pos = init_state.getPositions(True).value_in_unit(nanometer)
        init_energy, init_forces = self._target_func(init_pos, self._context, self._constraints)
        force_norms = [np.linalg.norm(f) for f in init_forces]
        print(" Initial max. force: {:15.3f} kJ/mol".format(np.max(force_norms)))
        print(" Initial energy:     {:15.3f} kJ/mol/nm".format(init_energy))


        self._step_num = 0
        args = (self._context, self._constraints)
        self._callback(init_pos)
        res = self._optimize.minimize(self._target_func, init_pos, args=args, method='L-BFGS-B', jac=True, callback=self._callback,
        options=dict(maxiter=200, disp=False, gtol=5))
        final_pos = res.x.reshape(-1,3)

        final_energy, final_forces = self._target_func(final_pos, self._context, self._constraints)
        force_norms = [np.linalg.norm(f) for f in final_forces]
        print(" Final max. force:   {:15.3f} kJ/mol".format(np.max(force_norms)))
        print(" Final energy:       {:15.3f} kJ/mol/nm".format(final_energy))


    def _target_func(self, pos, context, constraints=None):
        context.setPositions(pos.reshape(-1,3))
        state = context.getState(getEnergy=True, getForces=True)
        forces = state.getForces(asNumpy=True)
        energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        forces = forces.value_in_unit(kilojoule_per_mole/nanometer)

        if constraints is not None:
            for n, constr in constraints.items():
                for idx in self._constr_2_idx[constr.upper()]:
                    forces[n][idx] *= 0

        return energy, -forces.flatten()