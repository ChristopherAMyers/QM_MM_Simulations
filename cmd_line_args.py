import os.path as path
from os import makedirs
import sys
from shutil import copyfile

class FileSections():
    def __init__(self, start_str, end_str, file_name, arg=None, trim=True) -> None:
        self.start_str = str.lower(start_str)   #   string to indicate section start
        self.end_str = str.lower(end_str)       #   string to indicate section start
        self.file_name = str.lower(file_name)   #   file name to save data to
        self.arg = arg                          #   command line argument to use
        self.lines = []                         #   actual data
        self.trim = trim                        #   keep start_str or end_str in data

def parse_cmd_line_args(scratch_dir):
    if not path.isdir(scratch_dir):
        makedirs(scratch_dir)

    #   check if condensed input file is provided
    if '-ipt' not in sys.argv:
        return sys.argv[1:]
    else:
        file_loc = sys.argv[sys.argv.index('-ipt') + 1]
    file_loc = path.abspath(file_loc)

    return_argv = []
    for argv in sys.argv[1:]:
        return_argv.append(argv)

    sections = [
        FileSections('$rem', '$end', 'rem', '-rem', trim=False),
        FileSections('$velocity', '$end', 'init_velocity', '-velocity'),
        FileSections('$idx', '$end', 'idx', '-idx'),
        FileSections('$hugs', '$end', 'hugs', '-hugs'),
        FileSections('$restraints', '$end', 'restraints', '-rest'),
        FileSections('$pawl', '$end', 'pawl', '-pawl'),
        FileSections('$freeze', '$end', 'freeze', '-freeze'),
        FileSections('$fragments', '$end', 'frags', '-frags'),
        FileSections('$link', '$end', 'link', '-link'),
        FileSections('$centroids', '$end', 'centroid', '-centroid'),
        FileSections('$points', '$end', 'point', '-point'),
        FileSections('$other', '$end', 'other')
    ]

    
    copyfile(file_loc, path.join(scratch_dir, 'qm_mm_input'))

    current_read = None
    with open(file_loc) as file:
        for line in file.readlines():
            orig_line = line
            line = line.lower()
            sp = line.split()
            if len(sp) == 0:
                continue


            #   determine if current line starts a new section
            for sec in sections:
                if sec.start_str == sp[0]:
                    #   make sure we aren't already reading a section
                    if current_read:
                        raise Exception(' Could not find ending string for {:s} section'.format(current_read.start_str))
                    current_read = sec
                    break
            
            #   if we are reading a section, save the lines
            if current_read:
                if not (current_read.trim and (sp[0] == current_read.start_str or sp[0] == current_read.end_str) ):
                    current_read.lines.append(orig_line)
            else:
                sections[-1].lines.append(line)
            
            
            #   if line signals end of section to read
            for sec in sections:
                if sec.end_str == sp[0]:
                    #   terminating string found before starting string
                    #if not current_read:
                    #    raise Exception(' Encountered section end with no starting string')

                    current_read = None
                    break

    #   split data into individual files
    for sec in sections:
        if len(sec.lines) != 0 or sec.file_name == 'idx':
            file_loc = path.abspath(path.join(scratch_dir, sec.file_name))
            if sec.arg:
                return_argv.append(sec.arg)
                return_argv.append(file_loc)
            with open(file_loc, 'w') as file:
                for line in sec.lines:
                    file.write(line)

    return return_argv

            
            
