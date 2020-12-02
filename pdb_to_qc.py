from simtk.openmm.app import PDBFile, ForceField
from simtk.unit import *
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

def determine_connectivity(topology):
    '''
        taken from simtk/openmm/app/forcefield.py
    '''
    data = ForceField._SystemData()
    data = ForceField._SystemData()
    data.atoms = list(topology.atoms())
    for atom in data.atoms:
        data.excludeAtomWith.append([])

    # Make a list of all bonds
    for bond in topology.bonds():
        data.bonds.append(ForceField._BondData(bond[0].index, bond[1].index))

    # Record which atoms are bonded to each other atom
    bondedToAtom = []
    for i in range(len(data.atoms)):
        bondedToAtom.append(set())
        data.atomBonds.append([])
    for i in range(len(data.bonds)):
        bond = data.bonds[i]
        bondedToAtom[bond.atom1].add(bond.atom2)
        bondedToAtom[bond.atom2].add(bond.atom1)
        data.atomBonds[bond.atom1].append(i)
        data.atomBonds[bond.atom2].append(i)

     # Make a list of all unique angles
    uniqueAngles = set()
    for bond in data.bonds:
        for atom in bondedToAtom[bond.atom1]:
            if atom != bond.atom2:
                if atom < bond.atom2:
                    uniqueAngles.add((atom, bond.atom1, bond.atom2))
                else:
                    uniqueAngles.add((bond.atom2, bond.atom1, atom))
        for atom in bondedToAtom[bond.atom2]:
            if atom != bond.atom1:
                if atom > bond.atom1:
                    uniqueAngles.add((bond.atom1, bond.atom2, atom))
                else:
                    uniqueAngles.add((atom, bond.atom2, bond.atom1))
    data.angles = sorted(list(uniqueAngles))

    # Make a list of all unique proper torsions
    uniquePropers = set()
    for angle in data.angles:
        for atom in bondedToAtom[angle[0]]:
            if atom not in angle:
                if atom < angle[2]:
                    uniquePropers.add((atom, angle[0], angle[1], angle[2]))
                else:
                    uniquePropers.add((angle[2], angle[1], angle[0], atom))
        for atom in bondedToAtom[angle[2]]:
            if atom not in angle:
                if atom > angle[0]:
                    uniquePropers.add((angle[0], angle[1], angle[2], atom))
                else:
                    uniquePropers.add((atom, angle[2], angle[1], angle[0]))
    data.propers = sorted(list(uniquePropers))

    # Make a list of all unique improper torsions
    for atom in range(len(bondedToAtom)):
        bondedTo = bondedToAtom[atom]
        if len(bondedTo) > 2:
            for subset in itertools.combinations(bondedTo, 3):
                data.impropers.append((atom, subset[0], subset[1], subset[2]))

    return data, bondedToAtom

def get_bonds_from_coords(elm1, elm2, coords, elms, cutoff):
    sub_coords1 = []
    sub_coords2 = []
    idx_list_1 = []
    idx_list_2 = []
    for n in range(len(coords)):
        if elms[n] == elm1:
            sub_coords1.append(coords[n] / angstrom)
            idx_list_1.append(n)
        if elms[n] == elm2:
            sub_coords2.append(coords[n] / angstrom)
            idx_list_2.append(n)


    sub_coords1 = np.array(sub_coords1)
    sub_coords2 = np.array(sub_coords2)
    idx_list_1 = np.array(idx_list_1)
    idx_list_2 = np.array(idx_list_2)
    if len(sub_coords1) == 0 or len(sub_coords2) == 0:
        return []
    distMat = np.linalg.norm(sub_coords1[:, None] - sub_coords2, axis=-1)
    bonds = []
    for n, row in enumerate(distMat):
        where_within = np.where(row <= cutoff)[0]
        idx_list = idx_list_2[where_within]
        for idx in idx_list:
            if idx_list_1[n] != idx:
                bonds.append([idx_list_1[n], idx])
    return bonds

def add_bonds(pdb):

    coords = pdb.getPositions(asNumpy=True).in_units_of(angstroms)
    elements = np.array([x.element.symbol.lower() for x in pdb.topology.atoms()])
    bond_list =  get_bonds_from_coords('n', 'h', coords, elements, 1.3)
    bond_list += get_bonds_from_coords('c', 'h', coords, elements, 1.3)
    bond_list += get_bonds_from_coords('c', 'n', coords, elements, 1.8)
    bond_list += get_bonds_from_coords('c', 'c', coords, elements, 1.8)
    bond_list += get_bonds_from_coords('se', 'zn', coords, elements, 2.7)
    bond_list += get_bonds_from_coords('zn', 'n', coords, elements, 2.4)
    

    atoms = list(pdb.topology.atoms())
    for bond in bond_list:
        pdb.topology.addBond(atoms[bond[0]], atoms[bond[1]])

def pdb_to_qc(pdb, file_out_loc, bondedToAtom, qc_forcefield):

    coords = pdb.positions.in_units_of(angstroms)

    #   get atom type indicies from qc_forcefield file
    atom_types = {}
    atom_chg = {}
    with open(qc_forcefield, 'r') as file:
        for line in file.readlines():
            sp = line.split()
            if sp[0].lower() == "atom":
                key = sp[4]
                vdw = int(sp[1])
                chg = float(sp[2])
                atom_types[key] = vdw
                atom_chg[key] = chg

    #   store all lines to write to file and update total charge
    total_chg = 0.0
    file_lines = []
    
    file_lines.append("$molecule\n")
    file_lines.append("0 1 \n")
    for n, atom in enumerate(pdb.topology.atoms()):
        elm = atom.element.symbol
        idx = atom.index
        bonds = np.zeros(4, dtype=int)
        for m, i in enumerate(bondedToAtom[n]):
            bonds[m] = i + 1
        print(n + 1, bonds)
        type_idx = atom_types[atom.residue.name + '-' + atom.name]
        total_chg += atom_chg[atom.residue.name + '-' + atom.name]
        file_lines.append("    {:2s}  {:10.5f}  {:10.5f}  {:10.5f}   {:2d}  {:3d}  {:3d}  {:3d}  {:3d} \n" \
            .format(elm, coords[n][0]/angstrom, coords[n][1]/angstrom, coords[n][2]/angstrom, 
            -type_idx, bonds[0], bonds[1], bonds[2], bonds[3]))
    file_lines.append("$end\n")

    #   update total charge line
    print("Total charge: {:8.3f}".format(total_chg))
    #file_lines[1] = "{:8.3f} 1 \n".format(total_chg)

    #   write to file
    with open(file_out_loc, 'w') as file:
        for line in file_lines:
            file.write(line)




if __name__ == "__main__":
    pdb = PDBFile(sys.argv[1])
    add_bonds(pdb)
    data, bondedToAtom = determine_connectivity(pdb.topology)
    pdb_to_qc(pdb, sys.argv[2], bondedToAtom, 'forcefield.prm')