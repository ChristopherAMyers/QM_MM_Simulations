class QM_Fragments():
    def __init__(self, frag_file_loc, topology):
        self.frag_file_loc = frag_file_loc
        self.atom_ids = []
        self.frag2_qm_atoms = []
        self.gradient_order = None

        #   import fragment charge, spin, and atom ids to use from file
        line_num = 0
        with open(frag_file_loc) as file:
            for line in file:
                sp = line.split()
                if len(sp) == 0: continue

                if line_num == 0:
                    self.frag2_charge = int(sp[0])
                    self.frag2_spin_mult = int(sp[1])
                else:
                    self.atom_ids.append(int(sp[0]))

                line_num += 1

        #   convert atom ids to indicies
        for atom in topology.atoms():
            if int(atom.id) in self.atom_ids:
                self.frag2_qm_atoms.append(atom.index)


    def convert_molecule_lines(self, total_chg, spin_mult, mol_lines, qm_atoms):

        frag1_charge = total_chg - self.frag2_charge
        frag1_mult = spin_mult
    
        #   assumes molecule lines are in same order as qm_atoms
        frag1_lines = []
        frag2_lines = []
        frag1_lines.append('    {:d}  {:d} \n'.format(frag1_charge, frag1_mult))
        frag2_lines.append('    {:d}  {:d} \n'.format(self.frag2_charge, self.frag2_spin_mult))
        frag1_order = []
        frag2_order = []
        for n, line in enumerate(mol_lines):
            idx = qm_atoms[n]
            if idx in self.frag2_qm_atoms:
                frag2_lines.append(line)
                frag2_order.append(n)
            else:
                frag1_lines.append(line)
                frag1_order.append(n)

        new_mol_lines = []
        new_mol_lines += ['-- \n']
        new_mol_lines += frag1_lines
        new_mol_lines += ['-- \n']
        new_mol_lines += frag2_lines

        pre_order = frag1_order + frag2_order
        self.gradient_order = [0]*len(pre_order)
        for n in range(len(pre_order)):
            self.gradient_order[pre_order[n]] = n

        return new_mol_lines
        



        

