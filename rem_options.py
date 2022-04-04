from openmm.unit import * #pylint: disable=unused-wildcard-import
from sim_extras import *
from distutils.util import strtobool
import os

def get_rem_lines(rem_file_loc, outfile):
    rem_lines = []
    rem_lines_in = []

    with open(rem_file_loc, 'r') as file:
        for line in file.readlines():
            if "$" not in line:
                rem_lines_in.append(line.replace('=', ' '))

    opts = JobOptions()

    for line in rem_lines_in:
        orig_line = line
        line = line.lower()
        sp_comment = line.split('!')
        if sp_comment[0] != '!':
            sp = line.split()
            if len(sp) == 0:
                continue
            option = sp[0].lower()
            if option == 'time_step':
                opts.time_step = float(sp[1]) * 0.0242 * femtoseconds
            elif option == 'jobtype':
                opts.jobtype = sp[1]
            elif option == 'aimd_steps' or option == 'geom_opt_max_cycles':
                opts.aimd_steps = int(sp[1])
            elif option == 'aimd_temp':
                opts.aimd_temp = float(sp[1]) * kelvin
            elif option == 'aimd_thermostat':
                opts.aimd_thermostat = sp[1]
            elif option == 'aimd_langevin_timescale':
                opts.aimd_langevin_timescale = float(sp[1]) * femtoseconds
            elif option == 'constrain_qmmm_bonds':
                opts.constrain_qmmm_bonds = strtobool(sp[1])
            elif option == 'ff_file':
                opts.force_field_files.append(os.path.abspath(orig_line.split()[1]))
            elif option == 'constrain_hbonds':
                opts.constrain_hbonds = strtobool(sp[1])


            #   adaptive QM atoms
            elif option == 'qm_mm_radius':
                opts.qm_mm_radius = float(sp[1]) * angstroms
            elif option == 'qm_mm_update':
                opts.qm_mm_update = strtobool(sp[1])
            elif option == 'qm_mm_update_freq':
                opts.qm_mm_update_freq = int(sp[1])

            #   temperature anealing
            elif option == 'annealing':
                opts.annealing = bool(strtobool(sp[1]))
            elif option == 'annealing_peak':
                opts.annealing_peak = float(sp[1]) * kelvin
            elif option == 'annealing_period':
                opts.annealing_period = float(sp[1]) * femtoseconds

            #   osygen boundry force
            elif option == 'oxy_bound':
                opts.oxy_bound = bool(strtobool(sp[1]))
            elif option == 'oxy_bound_force':
                opts.oxy_bound_force = float(sp[1])
            elif option == 'oxy_bound_dist':
                opts.oxy_bound_dist = float(sp[1]) * nanometers            

            #   ratchet and pawl force
            elif option == 'ratchet_pawl':
                opts.ratchet_pawl = bool(strtobool(sp[1]))
            elif option == 'ratchet_pawl_force':
                opts.ratchet_pawl_force = float(sp[1])
            elif option == 'ratchet_pawl_half_dist':
                opts.ratchet_pawl_half_dist = float(sp[1])
            elif option == 'ratchet_pawl_switch':
                opts.ratchet_pawl_switch = sp[1].lower()

            #   charge and multiplicity
            elif option == 'mult':
                opts.mult = int(sp[1])
            elif option == 'charge':
                opts.charge = int(sp[1])
            elif option == 'adapt_spin':
                opts.adapt_mult = bool(strtobool(sp[1]))
            elif option == 'mc_spin':
                opts.mc_spin = bool(strtobool(sp[1]))
            elif option == 'mc_spin_max_mult':
                opts.mc_spin_max_mult = int(sp[1])
            elif option == 'mc_spin_min_mult':
                opts.mc_spin_min_mult = int(sp[1])

            #   oxygen repulsion force
            elif option == 'oxy_repel':
                opts.oxy_repel = bool(strtobool(sp[1]))
            elif option == 'oxy_repel_dist':
                opts.oxy_repel_dist = float(sp[1])

            #   random kicks
            elif option == 'random_kicks':
                opts.random_kicks = bool(strtobool(sp[1]))
            elif option == 'random_kicks_scale':
                opts.random_kicks_scale = float(sp[1])

            #   initial ionization
            elif option == 'ionization':
                opts.ionization = bool(strtobool(sp[1]))
            elif option == 'ionization_num':
                opts.ionization_num = int(sp[1])

            #   restraint force
            elif option == 'restraints':
                opts.restraints = bool(strtobool(sp[1]))
            elif option == 'restraints_switch_time':
                opts.restraints_switch_time = float(sp[1]) * femtoseconds

            #   centroid restraints
            elif option == 'cent_restraints':
                opts.cent_restraints = strtobool(sp[1])

            #   point restraints
            elif option == 'point_restraints':
                opts.point_restraints = strtobool(sp[1])

            #   QM fragments
            elif option == 'qm_fragments':
                opts.qm_fragments = strtobool(sp[1])

            #   QM / MM Janus Link atoms
            elif option == 'link_atoms':
                opts.link_atoms = strtobool(sp[1])

            #   random number seeds
            elif option == 'aimd_temp_seed':
                seed = int(sp[1])
                if seed > 2147483647 or seed < -2147483648:
                    raise ValueError('rem AIMD_TEMP_SEED must be between -2147483648 and 2147483647')
                opts.aimd_temp_seed = seed
            elif option == 'aimd_langevin_seed':
                seed = int(sp[1])
                if seed > 2147483647 or seed < -2147483648:
                    raise ValueError('rem AIMD_LANGEVIN_SEED must be between -2147483648 and 2147483647')
                opts.aimd_langevin_seed = seed
            elif option == 'script':
                opts.script_file = os.path.abspath(sp[1])
            else:
                rem_lines.append(line)
            #else:
            #    print("ERROR: rem option < %s > is not supported" % option)
            #    print("       Script will now terminate")
            #    exit()

    #   print rem file to output so user can make sure
    #   it was interpereted correctly
    outfile.write(' Imported rem file \n')
    for line in open(rem_file_loc, 'r').readlines():
        outfile.write(line)
    outfile.write('\n')


    if opts.jobtype == 'aimd':
        if opts.aimd_thermostat:
            opts.integrator = 'Langevin'
        else:
            opts.integrator = 'Verlet'
    elif opts.jobtype.lower() == 'grad':
        opts.integrator = 'Steepest-Descent'
    else:
        opts.integrator = 'Conjugate-Gradient'


    
    outfile.write('--------------------------------------------\n')
    outfile.write('              Script Options                \n')
    outfile.write('--------------------------------------------\n')
    outfile.write(' jobtype:                   {:>10s} \n'.format(opts.jobtype) )
    outfile.write(' integrator:                {:>10s} \n'.format(opts.integrator) )
    outfile.write(' time step:                 {:>10.2f} fs \n'.format(opts.time_step/femtoseconds) )
    outfile.write(' number of steps:           {:>10d} \n'.format(opts.aimd_steps) )
    outfile.write(' Total QM charge:           {:10d} \n'.format(opts.charge))
    outfile.write(' QM Multiplicity:           {:10d} \n'.format(opts.mult))
    outfile.write(' QM/MM radius:              {:>10.2f} Ang. \n'.format(opts.qm_mm_radius/angstroms) )
    outfile.write(' QM/MM update:              {:10d}. \n'.format(opts.qm_mm_update) )
    outfile.write(' QM/MM update frequency:    {:10d} steps. \n'.format(opts.qm_mm_update_freq) )
    outfile.write(' Constrain QM/MM bonds:     {:10d} \n'.format(opts.constrain_qmmm_bonds))

    if opts.adapt_mult:
        outfile.write(' Adaptive Spin:             {:10d} \n'.format(int(opts.adapt_mult)))
    if opts.mc_spin:
        outfile.write(' MCMC Spin:                 {:10d} \n'.format(int(opts.mc_spin)))
        outfile.write(' Max MC Spin Multiplicity:  {:10d} \n'.format(int(opts.mc_spin_max_mult)))
        outfile.write(' Min MC Spin Multiplicity:  {:10d} \n'.format(int(opts.mc_spin_min_mult)))

    if opts.ratchet_pawl:
        outfile.write(' Ratchet-Pawl:              {:10d} \n'.format(int(opts.ratchet_pawl)))
        outfile.write(' Ratchet-Pawl Force:        {:10.1f} \n'.format(float(opts.ratchet_pawl_force)))
        outfile.write(' Ratchet-Pawl Half-Dist:    {:10.4f} \n'.format(opts.ratchet_pawl_half_dist))
        outfile.write(' Ratchet-Pawl Switching:    {:>10s} \n'.format(opts.ratchet_pawl_switch))

    if opts.jobtype == 'aimd':
        outfile.write(' temperature:               {:>10.2f} K \n'.format(opts.aimd_temp/kelvin) )
        outfile.write(' temperature seed:          {:>10d} \n'.format(opts.aimd_temp_seed) )

    if opts.aimd_thermostat:
        outfile.write(' thermostat:                {:>10s} \n'.format(opts.aimd_thermostat) )
        outfile.write(' langevin frequency:      1/{:>10.2f} fs \n'.format(opts.aimd_langevin_timescale / femtoseconds) )
        outfile.write(' langevin seed:            {:11d} \n'.format(opts.aimd_langevin_seed))

    if opts.annealing:
        outfile.write(' Temperature Annealing:     {:10d} \n'.format(int(opts.annealing)))
        outfile.write(' Annealing Peak:            {:10.2f} K\n'.format(opts.annealing_peak / kelvin))
        outfile.write(' Annealing Period:          {:10.1f} fs\n'.format(opts.annealing_period/femtoseconds))

    if opts.oxy_bound:
        outfile.write(' Oxygen boundry:            {:10d} \n'.format(int(opts.oxy_bound)))
        outfile.write(' Oxygen boundry force:      {:10.4f} nm \n'.format(opts.oxy_bound_dist / nanometers))
        outfile.write(' Oxygen boundry distance:   {:10.1f} \n'.format(opts.oxy_bound_force))

    if opts.oxy_repel:
        outfile.write(' Oxy - Oxy Repulsion:       {:10d} \n'.format(int(opts.oxy_repel)))
        outfile.write(' Oxy - Oxy Distance:        {:10.4f} \n'.format(float(opts.oxy_repel_dist)))

    if opts.random_kicks:
        outfile.write(' Random Thermal Kicks:      {:10d} \n'.format(int(opts.random_kicks)))
        outfile.write(' Random Kicks Scale:        {:10.5f} \n'.format(opts.random_kicks_scale))

    if opts.ionization:
        outfile.write(' Initial Ionization:        {:10d} \n'.format(int(opts.ionization)))
        outfile.write(' No. of ionized H2O pairs:  {:10d} \n'.format(opts.ionization_num))

    if opts.restraints:
        outfile.write(' Restraints:                {:10d} \n'.format(int(opts.restraints)))
        outfile.write(' Restraints Switch Time:    {:>10f} fs \n'.format(opts.restraints_switch_time/femtoseconds))

    if opts.cent_restraints:
        outfile.write(' Centroid Restraints:       {:10d} \n'.format(int(opts.cent_restraints)))

    if opts.point_restraints:
        outfile.write(' Point Restraints:          {:10d} \n'.format(int(opts.point_restraints)))

    if opts.qm_fragments:
        outfile.write(' QM Fragments:              {:10d} \n'.format(int(opts.qm_fragments)))

    if opts.link_atoms:
        outfile.write(' QM/MM Link Atoms:          {:10d} \n'.format(int(opts.link_atoms)))

    if opts.constrain_hbonds:
        outfile.write(' H-Bond Constraints:        {:10d} \n'.format(int(opts.constrain_hbonds)))

    outfile.write('--------------------------------------------\n')

    if len(opts.force_field_files) > 0:
        outfile.write(' Additional force fields:   \n')
        for f in opts.force_field_files:
            outfile.write('     %s\n' % f)
        outfile.write('--------------------------------------------\n')

    outfile.flush()
    return rem_lines, opts