# QM-MM Simulations

This is a small program that interfaces with Q-Chem Quantum Chemistry software and the OpenMM molecular dynamics library to run AIMD (*ab initio* molecular dynamics) simulations. *This is still very much a work in progress, so please pardon the mess!*

## Running the program
The main module file is `qm_mm.py`. To see the available command line optins availabe, you can run `$python3 qm_mm.py --help`, but the main usage is
```
python3 qm_mm.py -ipt input.txt -pdb molcule.pdb -out output.txt
```
`-ipt` is the input file that contains all of the options and lists of QM atoms to use in the system (see below for more info). <br>
`-pdb` is the molecular PDB file to run the simulation on <br>
`-out` specifies the output file to print results and program functionality to. <br>

## Inputs

The input file (specified by `-ipt`) uses a format similar to Q-Chem, with the main section begins with `$rem` and ends with `$end`. Any options *not* included in the following tabels are consided to be a Q-Chem $rem option. For this reason, many of the options that controll the simulation will be identical to those found in the Q-Chem manual. 

The following options are availabe for use:

| $rem option      | Description | Default value |
| :---        |    :---   |          ---: |
| time_step      | Integration step size in a.u.       |   42 a.u. $\approx$ 1 fs |
| jobtype   | 'aimd' for simulation or 'opt' for geometry optimization | 'aimd' |
| aimd_steps | number of AIMD steps to perform | 2000 |
| aimd_temp | simulation temperature (Kelvin) | 300 |
| aimd_temp_seed | random seed for generating initial velocities | n/a |
| aimd_langevin_seed | random seed for langevin integration | n/a |
| aimd_thermostat | 'none' uses a Verlet integrator <br> 'langevin' uses a Langevin integration integration | 'None' |
| aimd_langevin_timescale | Langevin correlation time | 100 fs|
| constrain_hbonds | constraint hydrogen-heavy atom bond lengths | False |



## QM/MM radius
Atoms that should be treated with QM are identified in the $idx section. This section lists the atom ID's in the PDB file that are to imported into a Q-Chem job. Any atom with an asterisk (*) listed after it (e.g. 16\*, 590\*, etc.) also includes a QM sphere centered on that atom. Any other atom that falls within the radius of that sphere is then also treated as a QM atom. This can be useful for treating solvent atoms that are near a solvute as QM, while those outside the sphere are treated with MM. If any atom falls within this radius, the *entire residue* is treated as QM, so use with care.

The following `$rem` options control the QM sphere
| $rem option      | Description | Default value |
| :---        |    :---   |          ---: |
| qm_mm_update | whether or not to update QM atoms within sphere <br> If False, initial atoms inside the sphere are kept as QM for the entire simulation. <br> If True, atoms are updated every `qm_mm_update_freq` time steps. | True |
| qm_mm_radius | radius in angstroms of QM sphere | 3.0 Ang. |
| qm_mm_update_freq | Number of time steps between updating QM atoms | 10 |

