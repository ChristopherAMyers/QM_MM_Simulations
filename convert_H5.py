#!/usr/bin/env python3
import mdtraj as md
import argparse
import os


def convert(input_file_loc, out_file_loc):
    traj = md.load(input_file_loc)

    extension = os.path.splitext(out_file_loc)[-1][1:]
    print(" Requested output: {:s}".format(extension))
    print(" Writing...")

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
        
    print(" ...Done writing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ipt', required=True, help='input HDF5 file')
    parser.add_argument('-o', '--out', required=True, help='output file (with file extension in name)')
    args = parser.parse_args()
    convert(args.ipt, args.out)