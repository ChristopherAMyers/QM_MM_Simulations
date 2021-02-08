#!/usr/bin/env python3
import mdtraj as md
import argparse
import os, sys

def convert(input_file_loc, out_file_loc, n_skip=None, keep_frame=None):
    print(" Reading in trajectory file...", end='', flush=True)
    #   separate out traj object to work better with pylint
    tmp_traj = md.Trajectory.load(input_file_loc)
    n_frames_in =  tmp_traj.n_frames
    if keep_frame:
        tmp_traj.xyz = tmp_traj.xyz[keep_frame - 1]
    elif n_skip and isinstance(skip, int):
        tmp_traj.xyz = tmp_traj.xyz[::skip]
    traj = md.Trajectory(tmp_traj.xyz, tmp_traj.topology)
    print("Done")

    print(" No. frames inported: {:>12d}".format(n_frames_in))
    print(" No. frames kept:     {:>12d}".format(traj.n_frames))

    extension = os.path.splitext(out_file_loc)[-1][1:]
    print(" Requested output: {:s}".format(extension))
    print(" Writing...", end='', flush=True)

    if extension == 'hdf5':
        traj.save_hdf5(out_file_loc)

    elif extension == 'lammpstrj':
        traj.save_lammpstrj(out_file_loc)

    elif extension == 'xyz':
        traj.save_xyz(out_file_loc)

    elif extension == 'pdb':
        traj.save_pdb(out_file_loc)

    elif extension == 'xtc':
        traj.save_xtc(out_file_loc)

    elif extension == 'trr':
        traj.save_trr(out_file_loc)

    elif extension == 'dcd':
        traj.save_dcd(out_file_loc)

    elif extension == 'dtr':
        traj.save_dtr(out_file_loc)

    elif extension == 'binpos':
        traj.save_binpos(out_file_loc)

    elif extension == 'mdcrd':
        traj.save_mdcrd(out_file_loc)

    elif extension == 'netcdf':
        traj.save_netcdf(out_file_loc)

    elif extension == 'netcdfrst':
        traj.save_netcdfrst(out_file_loc)

    elif extension == 'amberrst7':
        traj.save_amberrst7(out_file_loc)

    elif extension == 'lh5':
        traj.save_lh5(out_file_loc)

    elif extension == 'gro':
        traj.save_gro(out_file_loc)

    elif extension == 'tng':
        traj.save_tng(out_file_loc)

    elif extension == 'gsd':
        traj.save_gsd(out_file_loc)
        
    print("Done")


if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    skip = None
    keep_frame = None
    if '-skip' in sys.argv:
        skip = int(sys.argv[sys.argv.index('-skip') + 1])
    if '-n' in sys.argv:
        keep_frame = int(sys.argv[sys.argv.index('-n') + 1])
    
    convert(in_file, out_file, skip, keep_frame)