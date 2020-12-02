import numpy as np
from simtk.openmm.app import PDBFile
from simtk.unit import *


orig_coords = np.array([[0.000, 3.232,	0.034],
                        [1.145,	0.000,  2.025],
                        [4.482,	0.000,	0.034],
                        [3.337,	3.232,	2.025],
                        [0.119,	0.852,	0.000],
                        [1.025,	4.084,	2.060],
                        [3.456,	0.852,	2.060],
                        [4.363,	4.084,	0.000],
                        ])
orig_atoms = np.array(["Se", "Se", "Se", "Se", 
                        "Zn", "Zn", "Zn", "Zn"])
orig_names= np.array(["Se1", "Se2", "Se3", "Se4", 
                      "Zn1", "Zn2", "Zn3", "Zn4"])
cell_dims = np.array([[6.6746, 0.0000, 0.0000],
                      [0.0000, 6.4642, 0.0000],
                      [0.0000, 0.0000, 0.0000]])


def create_struct_from_template(nx, ny, file_name, mat=True, lig=True):
    template = PDBFile('ethyl_template.pdb')
    n_res = 0
    pdb_lines = []
    n_atoms = 0
    for i in range(nx):
        for j in range(ny):
            shift = cell_dims[0]*i + cell_dims[1]*j

            for res in template.topology.residues():
                if (res.name == "MAT" and mat) or (res.name == "ETH" and lig):
                    n_res += 1
                    for atom in res.atoms():
                        n_atoms += 1
                        coord = template.positions[atom.index].in_units_of(angstrom)/angstrom
                        coord = np.array(coord) + shift
                        x, y, z = coord

                        elm = atom.element.symbol
                        name = atom.name

                        pdb_line = "{:6s}{:<5d} {:4s} {:3s}  {:4d}    {:8.3f}{:8.3f}{:8.3f}                      {:2s}"\
                            .format("ATOM", n_atoms, name, res.name, n_res, 
                            x, y, z, elm)
                        pdb_lines.append(pdb_line)

    with open(file_name, 'w') as file:
            for line in pdb_lines:
                file.write(line + '\n')



def create_struct(nx, ny, file_name):
    coords = np.array([])
    pdb_lines = []
    n_res = 0
    n_atoms = 0
    for i in range(nx):
        for j in range(ny):
            n_res += 1
            shift = cell_dims[0]*i + cell_dims[1]*j
            coords = shift + orig_coords
            
            for n in range(8):
                n_atoms += 1
                pdb_line = "{:6s}{:5d} {:4s} {:3s} 1{:4d}    {:8.3f}{:8.3f}{:8.3f}                      {:2s}"\
                    .format("ATOM", n_atoms, orig_names[n], "MAT", n_res, 
                            coords[n][0], coords[n][1], coords[n][2], orig_atoms[n])
                pdb_lines.append(pdb_line)

    with open(file_name, 'w') as file:
        for line in pdb_lines:
            file.write(line + '\n')






    