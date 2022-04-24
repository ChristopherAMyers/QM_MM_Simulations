from openmm.app import *
from openmm import *
from openmm.unit import nanometers, kelvin, picoseconds
from sys import stdout

FAKE_FF = ForceField('/network/rit/lab/ChenRNALab/awesomeSauce/dynamic_charge/simulations/umbrella/a_u/qm_mm/no_qm_atoms/chen-garcia.script.xml')
FAKE_PDB = PDBFile('/network/rit/lab/ChenRNALab/awesomeSauce/dynamic_charge/simulations/umbrella/a_u/qm_mm/no_qm_atoms/input.pdb')
FAKE_SYSTEM = FAKE_FF.createSystem(FAKE_PDB.topology, nonbondedMethod=NoCutoff,
            nonbondedCutoff=1*nanometers, constraints=HBonds)

FAKE_INTEGRATOR = VerletIntegrator(0.002*picoseconds)
FAKE_SIM = Simulation(FAKE_PDB.topology, FAKE_SYSTEM, FAKE_INTEGRATOR)

def fake_qchem(pos, qm_atoms):
    
    
    atom_pos = [pos[x] for x in qm_atoms]
    FAKE_SIM.context.setPositions(atom_pos)
    state = FAKE_SIM.context.getState(getForces=True, getEnergy=True)

    forces = state.getForces(True)
    energy = state.getPotentialEnergy()

    print("FAKE Q-CHEM: ", energy)

    return energy/energy.unit, -forces/forces.unit