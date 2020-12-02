from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm import *
from openmmtools.integrators import VelocityVerletIntegrator
from simtk.openmm.openmm import *
from simtk.unit import *
import argparse
import numpy as np
import sys

# pylint: disable=no-member
import simtk
picoseconds = simtk.unit.picoseconds
picosecond = picoseconds
nanometer = simtk.unit.nanometer
femtoseconds = simtk.unit.femtoseconds
# pylint: enable=no-member

from sim_extras import *

def print_pressure(simulation, masses_in_kg):
    state = simulation.context.getState(getPositions=True, getForces=True, getVelocities=True, getEnergy=True)

        #   calculate kinetic energies
    system = simulation.context.getSystem()
    vel = state.getVelocities(True).in_units_of(meters/second)
    v2 = np.linalg.norm(vel, axis=1)**2
    k = 1.380649E-23

    #   calculate temperatures from kinetic energies
    temp_all = np.sum(v2*masses_in_kg)/(len(v2) * k * 3)

    #   calculate the center of mass
    pos = state.getPositions(True)
    weighted_pos = masses_in_kg[:, None] * pos
    com = np.sum(weighted_pos, axis=0)/np.sum(masses_in_kg)

    pos -= com
    forces = state.getForces(True)

    pressure_int = -0.5*np.sum(pos * forces)
    print(pressure_int)
    print(state.getKineticEnergy())
    exit()


if __name__ == "__main__":

    pdb = PDBFile(sys.argv[1])
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
            nonbondedCutoff=1*nanometer, constraints=HBonds)
    integrator = LangevinIntegrator(300*kelvin, 1/(100*femtoseconds), 0.001*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(300*kelvin)
    print(" Minimizing")
    simulation.minimizeEnergy()
    simulation.reporters.append(PDBReporter('output.pdb', 4))
    simulation.reporters.append(StateDataReporter('stats.txt', 1, step=True,
        potentialEnergy=True, temperature=True, separator=' ', ))

    masses_in_kg = np.array([system.getParticleMass(n)/dalton for n in range(simulation.topology.getNumAtoms())])*(1.67377E-27)
    
    print(" Running")
    for n in range(1000):
        simulation.step(1)
        simulation.topology.getNumAtoms
        print_pressure(simulation, masses_in_kg)