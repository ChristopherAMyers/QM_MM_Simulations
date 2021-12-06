# QM-MM Simulations

This is a small program that interfaces with Q-Chem Quantum Chemistry software and the OpenMM molecular dynamics library to run AIMD (*ab initio* molecular dynamics) simulations. OpenMM is used as the MD simulator for for Molecular Mechanics (MM) forces, while Q-Chem is used as the engine for QM forces and energies. *This is still very much a work in progress, so please pardon the mess!*

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
| aimd_temp | Simulation temperature (Kelvin) | 300 |
| aimd_temp_seed | Random seed for generating initial velocities | n/a |
| aimd_langevin_seed | Random seed for langevin integration | n/a |
| aimd_thermostat | 'none' uses a Verlet integrator <br> 'langevin' uses a Langevin integration integration | 'None' |
| aimd_langevin_timescale | Langevin correlation time | 100 fs|
| constrain_hbonds | constraint hydrogen-heavy atom bond lengths | False |

## QM/MM radius
Atoms that should be treated with QM are identified in the $idx section. This section lists the atom ID's in the PDB file that are to imported into a Q-Chem job. Any atom with an asterisk (\*) listed after it (e.g. 16\*, 590\*, etc.) also includes a QM sphere centered on that atom. For example, that section might look like

```
$idx
    50
    51
    52
    53*
    54*
    55*
$end
```
Here, atoms 50-52 do not use QM spheres, but atoms 53-55 do. Any other atom that falls within the radius of that sphere is then also treated as a QM atom. This can be useful for treating solvent atoms that are near a solvute as QM, while those outside the sphere are treated with MM. If any atom falls within this radius, the *entire residue* is treated as QM, so use with care.

The following `$rem` options control the QM sphere
| $rem option      | Description | Default value |
| :---        |    :---   |          ---: |
| qm_mm_update | Whether or not to update QM atoms within sphere. <br> If False, initial atoms inside the sphere are kept as QM for the entire simulation. <br> If True, atoms are updated every `qm_mm_update_freq` time steps. | True |
| qm_mm_radius | radius in angstroms of QM sphere | 3.0 Ang. |
| qm_mm_update_freq | Number of time steps between updating QM atoms | 10 |

## Temperature Annealing
The program also has the ability to apply sinusoidal temperature annealing using a Langevin integrations scheme. If enabled, the thermostat is initially set to `aimd_temp` and gradually increases to the `annealing peak` temperature. Once reached, it then decreases back down to the initial `aimd_temp` and the process repeats every `annealing period`.
| $rem option      | Description | Default value |
| :---        |    :---   |          ---: |
| annealing | Turn on temperature annealing | False |
| annealing_peak | The Maximum temperature to reach durring the annealing process. | 400 Kelvin |
| annealing_period | The period of oscillation of one annealing cycle (initial, peak, initial) | 5 ps |

## Centroid Restraint force
Restraints on groups of atoms are applied using "centroid" restraint forces. Centroids are defined as the center of mass of a group of atoms. The restraint are applied as harmonic cost functions about a fixed distance between two centroids. Centroid restraints are turned on by setting `cent_restraints` equal to `True` in the `$rem` section. <br>

To define the groups of atoms, we use a new `$centroids` input section. Within this section, each group is specified by a list of atom IDs, either as a range of atoms listed one after another. These atoms are placed between the keywords `GROUP`, `ENDGROUP` for each group. For example,
```
$centroids
GROUP
    1:15
ENDGROUP
GROUP
    16:26 25 27 28 29 30 30
ENDGROUP
FORCES
    1 2 3000 0.8
ENDFORCES
$end
```
In this example, two centroids are formed, one with atoms center of mass of atoms 1-15, the other with atoms 16-30. Note that repeated atoms (as in the second group) do not affect the centroid calculations. The force between each centroids are specified between the `FORCES`, `ENDFORCES` keywords. In this example, a harmonic force is specified between groups 1 and 2 with a spring constant of 3000 kJ/mol/nm^2 and a equilibrium distance of 0.8 nm. The groups numbers are implicitly assigned by the order in which they are specified in the input file; group 1 uses atoms 1-15 and group 2 uses atoms 16-30. <br>

Each centroid can also be constraint to a fixed planar position using the `BOUNDARY`, `ENDBOUNDARY` keywords. For example, if the folling was included in the `$centroids` section,
```
BOUNDARY
    1   xyz   1000    0.00
    2   xy    1000    0.20
ENDBOUNDARY
```
Then group 1 is restrained to the origin in all cartesian directions (x, y, and z), while group 2 is restrained to the xy plane a distance 0.20nm from the z-axes. The boundary restraints are not applied unless the centroids are beyone the specified distance. In this example, the second restraint is not applied until distance of group 2 in the xy plane is greater than 0.2nm. Otherwise, the centroid is free to move about. Mathematically, the resraint cost function $E_{cost}$ applied is of the form<br>
$E_{cost} = \frac{1}{2}k(\sqrt{x^2 + y^2} - 0.2nm)^2, \,\, r\ge0.2nm$<br>
$E_{cost} = 0, \,\, r<0.2nm$<br>

where $k$ is the spring constant = 1000 jK/mol/nm nad $x$ and $y$ are the cartesian positions of centroid 2. 

## Point-like Restraints
Individual atoms can also be constraint to a particular cartesian point in space using the `$points` section. Each line is this section must contain 6 elements, including the atom ID (single or as a range), spring constant, boundary type (x, xy, yz, etc.), and the x, y, and z coordinates in nm, *in this order*. For example,
```
$points
    5     1000 xyz 0.10  0.20 0.30
    6:10  2000 yz  0.00 -0.50 0.40
$end
```
In this example, a single restraint force with a spring cosntant 1000 kJ/nm/mol is applied to atom ID 5 and is restraint to the cartesian point (0.1, 0.2, 0.3)nm. Additionally, atoms 6-10 all have the same restraint force with spring constant 2000 kJ/mol/nm to the point y=-0.50nm and z=0.40. Because the boundary type for this restraint is specified as `yz`, the `x` component is ignored. In other words, the restraint cost energy is <br>
$
E_{cost} = \frac{1}{2}k\left((y  - -0.50nm)^2 + (z - 0.40nm^2\right)
$<br>

## Link atoms
A Janus link model can be used to link covalently bonded QM atoms to MM atoms. The link atoms are threated both as MM and QM atoms. In the MM system, they are treat as normal with the same specified charge and bond forces. As a QM atom, however, it is treated as a QM hydrogen atom with a gaussian charge, equal to the MM charge, superimposed to the same hydrogen position. The MM bond force between the link atom and it's covalently bonded QM atom is also appled. At this moment, the harmonic bond force is *not* reduced in order to account for the double counting of forces applied to the atom. <br>

Link atoms are enabled by setting the`$rem` option `link_atoms` to `True`. The atom ID's are specified in the same manner as the `$idx` section, but instead using a `$link` section. 