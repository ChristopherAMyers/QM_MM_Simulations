
from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm import *
from openmmtools.integrators import VelocityVerletIntegrator
from simtk.openmm.openmm import CustomNonbondedForce, CustomExternalForce
from simtk.unit import *
import argparse
import numpy as np
import sys
import itertools

# pylint: disable=no-member
import simtk
picoseconds = simtk.unit.picoseconds
picosecond = picoseconds
nanometer = simtk.unit.nanometer
femtoseconds = simtk.unit.femtoseconds
# pylint: enable=no-member

sys.path.insert(1, "/network/rit/lab/ChenRNALab/bin/Pymol2.3.2/pymol/lib/python3.7/site-packages")
from pymol import cmd, stored # pylint: disable=import-error

import create_mat
import pdb_to_qc


def get_bonds(file_loc):
    cmd.load(file_loc, 'obj')
    print("N-Atoms: ", cmd.count_atoms('obj'))
    #bonds = cmd.find_pairs('e. se', 'e. zn', cutoff=2.7)
    #bonds += cmd.find_pairs('e. n',  'e. c',  cutoff=1.8)
    #bonds += cmd.find_pairs('e. c',  'e. c',  cutoff=1.8)
    #bonds += cmd.find_pairs('e. c',  'e. h',  cutoff=1.3)
    bonds = cmd.find_pairs('e. n',  'e. h',  cutoff=1.3)
    #bonds += cmd.find_pairs('e. n',  'e. zn', cutoff=2.4)
    
    rtn_bonds = []
    for bond in bonds:
        print(bond, cmd.distance(selection1=bond[0], selection2=bond[1]))
        print(cmd.get_coords(bond[0]), cmd.get_coords(bond[1]))
        rtn_bonds.append([bond[0][1], bond[1][1]])
    

    return rtn_bonds
    
def list_in_list(list1, list2):
    for sub_list in list2:
        if list1 == sub_list or list(reversed(list1)) == sub_list:
            return True
    return False

def get_bond_types(pdb):
    topol = pdb.topology
    coords = pdb.positions
    atoms = list(pdb.topology.atoms())
    residues = [a.residue.name for a in atoms]
    combos = []
    print("    <HarmonicBondForce>")
    for bond in topol.bonds():
        type1 = residues[bond.atom1.index] + "-" + bond.atom1.name.upper()
        type2 = residues[bond.atom2.index] + "-" + bond.atom2.name.upper()
        bond_list = [type1, type2]
        if not list_in_list(bond_list, combos):
            combos.append(bond_list)
            coord1 = coords[bond.atom1.index]
            coord2 = coords[bond.atom2.index]
            dist = np.linalg.norm(coord1 - coord2).in_units_of(nanometer)
            print('        <Bond k="{:.2f}" length="{:.5f}" type1={:10s} type2={:10s}/>'
                .format(200000.0, dist/nanometer, '"' + type1 + '"', '"' + type2 + '"'))

    print("    </HarmonicBondForce>")

def find_bonded_atoms(bonds, atom1_idx, exclude_idx):
    rtn_list = []
    for bond in bonds:
        a1 = bond.atom1.index
        a2 = bond.atom2.index
        if atom1_idx == a1 and exclude_idx != a2:
            rtn_list.append(bond.atom2)
        elif atom1_idx == a2 and exclude_idx != a1:
            rtn_list.append(bond.atom1)
    return rtn_list


def get_angle_types(pdb, coords):
    topol = pdb.topology
    combos = []
    print("    <HarmonicAngleForce>")
    for n, bond in enumerate(topol.bonds()):
        type1 = "MAT-" + bond.atom1.name.upper()
        type2 = "MAT-" + bond.atom2.name.upper()
        nbr_atoms_1 = find_bonded_atoms(topol.bonds(), bond.atom1.index, bond.atom2.index)
        nbr_atoms_2 = find_bonded_atoms(topol.bonds(), bond.atom2.index, bond.atom1.index)

        for atom3 in nbr_atoms_1:
            type3 = "MAT-" + atom3.name.upper()
            angle_list = [type3, type1, type2]
            if not list_in_list(angle_list, combos):
                combos.append(angle_list)
                coord1 = coords[bond.atom1.index]
                coord2 = coords[bond.atom2.index]
                coord3 = coords[     atom3.index]
                diff_31 = coord3 - coord1
                diff_21 = coord2 - coord1
                n1 = diff_31 / np.linalg.norm(diff_31)
                n2 = diff_21 / np.linalg.norm(diff_21)
                angle = np.arccos(np.dot(n1, n2))

                print('        <Angle k="{:.2f}" angle="{:.5f}" type1="{:8s} type2="{:8s}  type3="{:8s}"/>'
                        .format(1000.0, angle, type3, type1, type2))


        for atom3 in nbr_atoms_2:
            type3 = "MAT-" + atom3.name.upper()
            angle_list = [type1, type2, type3]
            if not list_in_list(angle_list, combos):
                combos.append(angle_list)
                coord1 = coords[bond.atom1.index]
                coord2 = coords[bond.atom2.index]
                coord3 = coords[     atom3.index]
                diff_12 = coord1 - coord2
                diff_32 = coord3 - coord2
                n1 = diff_12 / np.linalg.norm(diff_12)
                n2 = diff_32 / np.linalg.norm(diff_32)
                angle = np.arccos(np.dot(n1, n2))

                print('        <Angle k="{:.2f}" angle="{:.5f}" type1="{:s} type2="{:s}  type3="{:s}"/>'
                        .format(1000.0, angle, type1, type2, type3))

    print("    <\\HarmonicAngleForce>")
            



def calc_angle(coord1, coord2, coord3):
    diff_12 = coord1 - coord2
    diff_32 = coord3 - coord2
    n1 = diff_12 / np.linalg.norm(diff_12)
    n2 = diff_32 / np.linalg.norm(diff_32)
    angle = np.arccos(np.dot(n1, n2))
    return angle

def calc_dihedral(coord1, coord2, coord3, coord4):
    u0 = (coord1 - coord2) / nanometer
    u1 = (coord3 - coord2) / nanometer
    u2 = (coord3 - coord4) / nanometer

    v0 = np.cross(u0, u1) 
    v1 = np.cross(u1, u2) 

    cos_angle =  np.dot(v0, v1)  / (np.linalg.norm(v0) * np.linalg.norm(v1))
    angle = np.arccos(cos_angle)

    if np.dot(u0, v1) < 0:
        angle *= -1.0

    return angle

def print_angle_force_field(pdb, angles):
    coords = pdb.positions
    atoms = list(pdb.topology.atoms())
    residues = [a.residue.name for a in atoms]
    unique_angles = []
    print("    <HarmonicAngleForce>")
    q = '"'
    for n, angle in enumerate(angles):
        types = [residues[i] + "-" + atoms[i].name.upper() for i in angle]
        if not list_in_list(types, unique_angles):
            unique_angles.append(types)
            angle_val = calc_angle(*tuple([coords[i] for i in angle]))
            print('        <Angle k="{:.2f}" angle="{:8.5f}" type1={:10s} type2={:10s}  type3={:10s}/>'
                        .format(1000.0, angle_val, q + types[0] + q, q + types[1] + q, q + types[2] + q))
    print("    </HarmonicAngleForce>")

def print_torsion_force_field(pdb, propers, impropers):
    coords = pdb.positions
    atoms = list(pdb.topology.atoms())
    residues = [a.residue.name for a in atoms]
    unique_angles = []
    print("    <PeriodicTorsionForce>")
    q = '"'
    for proper in propers:
        types = [residues[i] + "-" + atoms[i].name.upper() for i in proper]
        if atoms[proper[0]].element.atomic_number == 1 and atoms[proper[-1]].element.atomic_number == 1:
            k = 1.0
        else:
            k = 10.0
        if not list_in_list(types, unique_angles):
            unique_angles.append(types)
            angle = calc_dihedral(*tuple([coords[i] for i in proper])) - np.pi
            print('        <Proper k1="{:.2f}" periodicity1="{:d}" phase1="{:9.5f}" type1={:10s} type2={:10s}  type3={:10s}  type4={:10s}/>'
                        .format(k, 1, angle, q + types[0] + q, q + types[1] + q, q + types[2] + q, q + types[3] + q))

    for improper in impropers:
        types = [residues[i] + "-" + atoms[i].name.upper() for i in improper]
        k = 10.0
        if not list_in_list(types, unique_angles):
            unique_angles.append(types)
            angle = calc_dihedral(*tuple([coords[i] for i in improper])) - np.pi
            print('        <Improper k1="{:.2f}" periodicity1="{:d}" phase1="{:9.5f}" type1={:10s} type2={:10s}  type3={:10s}  type4={:10s}/>'
                        .format(k, 1, angle, q + types[0] + q, q + types[1] + q, q + types[2] + q, q + types[3] + q))

    print("    </PeriodicTorsionForce>")

def set_ff_params(system, context):
    for force in system.getForces():
        if isinstance(force, HarmonicAngleForce) and True:
            for n in range(force.getNumAngles()):
                params = force.getAngleParameters(n)
                params[-1] *= 10
                force.setAngleParameters(n, *params)
            force.updateParametersInContext(context)

        if isinstance(force, PeriodicTorsionForce):
            for n in range(force.getNumTorsions()):
                params = force.getTorsionParameters(n)
                params[-1] *= 10
                #params[-2] -= np.pi * radians
                force.setTorsionParameters(n, *params)
            force.updateParametersInContext(context)




if __name__ == "__main__":
    #create_mat.create_struct(4, 4, 'test.pdb')
    #create_mat.create_struct_from_template(3, 3, 'test.pdb', lig=True)
    #pdb = PDBFile('test.pdb')
    pdb = PDBFile(sys.argv[1])
    pdb_to_qc.add_bonds(pdb)
    data, bondedToAtom = pdb_to_qc.determine_connectivity(pdb.topology)

    if True:
        get_bond_types(pdb)
        print_angle_force_field(pdb, data.angles)
        print_torsion_force_field(pdb, data.propers, data.impropers)
        exit()

    forcefield = ForceField('forcefields/forcefield2.xml')
    
    pdb_to_qc.pdb_to_qc(pdb, 'test.qc', bondedToAtom, 'forcefields/forcefield.prm')
    unmatched_residues = forcefield.getUnmatchedResidues(pdb.topology)
    [templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(pdb.topology)

    for n, template in enumerate(templates):
        residue = residues[n]
        for atom in template.atoms:
            print(n, residue.name + "-" + atom.name.upper(), residue.index)
            atom.type = residue.name + "-" + atom.name.upper()

        # Register the template with the forcefield.
        template.name += str(n)
        forcefield.registerResidueTemplate(template)

    #   print out bonded terms for force field file
        

    system, ff_data = forcefield.createSystem(pdb.topology)

    #   check to make sure all bonded forces are accounted for
    for force in system.getForces():
        if  isinstance(force, HarmonicBondForce):
            print("HarmonicBondForce: ", force.getNumBonds(), len(data.bonds))
        elif isinstance(force, HarmonicAngleForce):
            print("HarmonicAngleForce: ", force.getNumAngles(), len(data.angles))
        elif isinstance(force, PeriodicTorsionForce):
            print("PeriodicTorsionForce: ", force.getNumTorsions(), len(data.propers) + len(data.impropers))


    #integrator = VerletIntegrator(1*femtoseconds)
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.001*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    #set_ff_params(system, simulation.context)
    simulation.context.setPositions(pdb.positions)
    simulation.reporters.append(PDBReporter('output.pdb', 10))
    simulation.reporters.append(StateDataReporter(sys.stdout, 100, step=True,
            potentialEnergy=True, temperature=True))
    print("Running")
    simulation.minimizeEnergy(maxIterations=10)
    simulation.step(10000)

    atoms = list(pdb.topology.atoms())
    coords = list(pdb.positions)
    f = system.getForces()[1]