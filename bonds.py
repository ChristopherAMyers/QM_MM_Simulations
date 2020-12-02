from pymol import cmd, stored
from sys import argv

objects = cmd.get_object_list(selection='(all)')

print(cmd.find_pairs('e. se', 'e. zn', cutoff=2.7))

for obj in objects:
	print(obj)

	#bonds =  cmd.find_pairs('n. se and ' + obj, 'n. o  and ' + obj , cutoff=0.0)
	bonds = cmd.find_pairs('n. se and ' + obj, 'n. zn and ' + obj, cutoff=2.7)
	bonds += cmd.find_pairs('n. n  and ' + obj, 'n. zn and ' + obj, cutoff=2.5)
	#bonds += cmd.find_pairs('n. zn and ' + obj, 'n. o  and ' + obj , cutoff=0.0)
	bonds += cmd.find_pairs('n. zn and ' + obj, 'n. c  and ' + obj , cutoff=2.5)
	bonds += cmd.find_pairs('n. se and ' + obj, 'n. c  and ' + obj , cutoff=2.5)
	bonds += cmd.find_pairs('e. zn and ' + obj, 'e. n  and ' + obj , cutoff=2.5)
	bonds += cmd.find_pairs('e. zn and ' + obj, 'e. se and ' + obj, cutoff=2.7)
	for pair in bonds:
		cmd.bond('index {:d} and {:s}'.format(pair[0][1], obj), 'index {:d} and {:s}'.format(pair[1][1], obj))

for obj in objects:
    print(obj)
    bonds =  cmd.find_pairs('n. se and ' + obj, 'n. o  and ' + obj , cutoff=2.3)
    bonds += cmd.find_pairs('n. zn and ' + obj, 'n. o  and ' + obj , cutoff=2.3)
    #bonds += cmd.find_pairs('n. o and ' + obj, 'n. o  and ' + obj , cutoff=2.3)
    bonds += cmd.find_pairs('n. o and ' + obj, 'n. h  and ' + obj , cutoff=2.3)
    bonds += cmd.find_pairs('n. se and ' + obj, 'n. h  and ' + obj , cutoff=1.5)

    for pair in bonds:
        cmd.unbond('index {:d} and {:s}'.format(pair[0][1], obj), 'index {:d} and {:s}'.format(pair[1][1], obj))


for obj in objects:
    print(obj)
    bonds =  cmd.find_pairs('n. h and ' + obj, 'n. o  and ' + obj , cutoff=1.1)
    bonds += cmd.find_pairs('n. o and ' + obj, 'n. o  and ' + obj , cutoff=2.0)

    for pair in bonds:
        cmd.bond('index {:d} and {:s}'.format(pair[0][1], obj), 'index {:d} and {:s}'.format(pair[1][1], obj))

cmd.show('sticks')
cmd.hide('spheres')
