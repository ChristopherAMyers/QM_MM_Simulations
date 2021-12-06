import numpy as np
from simtk.openmm.app.forcefield import * #pylint: disable=unused-wildcard-import
from simtk.openmm.app import * #pylint: disable=unused-wildcard-import
from simtk.openmm import * #pylint: disable=unused-wildcard-import


if __name__ == "__main__":

    ff_mode = 'ipt'
    #ff_mode = ''
    #ff_mode = 'tinker'

    ff_file = 'forcefields/forcefield2.xml'
    ff = ForceField(ff_file)

    #   figure out whihc force is what
    nb_force = NonbondedGenerator(ff, 0.833, 0.5, False)
    bend_force = HarmonicBondGenerator(ff)
    angle_force = HarmonicAngleGenerator(ff)
    torsion_force = PeriodicTorsionGenerator(ff)
    for force in ff.getGenerators():
        if  isinstance(force, HarmonicBondGenerator):
            bend_force = force
        elif isinstance(force, HarmonicAngleGenerator):
            angle_force = force
        elif isinstance(force, PeriodicTorsionGenerator):
            torsion_force = force
        elif isinstance(force, NonbondedGenerator):
            nb_force = force
    exit()
    #   add index number to param attributes
    paramsForType = nb_force.params.paramsForType
    for n, param in enumerate(paramsForType.values()):
        param['index'] = n + 1


    

    #   create atom and vdW type lines
    idx_shift = 55
    two_1_6 = 2.0**(1.0/6.0)
    atom_lines = []
    vdw_lines = []
    chg_lines = []
    for n, atomType in enumerate(ff._atomTypes):
        params = paramsForType[atomType]
        sigma = params['sigma']*10/2.0*two_1_6
        epsilon = params['epsilon']/4.184
        if ff_mode == 'ipt':
            atom_lines.append("AtomType  {:4d} {:10.5f}  {:10.5f}  {:10.5f} {:s}".format(-(n + 1), params['charge'], sigma, epsilon, atomType) )
        elif ff_mode == 'tinker':
            atom_lines.append("atom  {:4d}   {:4d}    {:4s}   {:10s}  {:2d}".format(n + 1 + 9900, params['index'] + idx_shift, atomType[4:], '"' + atomType + '"', 2) )
            vdw_lines.append('vdw  {:4d}  {:10.5f}  {:10.5f}'.format(params['index'] + idx_shift, sigma, epsilon))
            chg_lines.append('charge  {:4d}  {:10.5f}'.format(n + 1 + 9900, params['charge']))
        else:
            atom_lines.append("Atom  {:4d} {:10.5f} {:4d}  {:s}".format(n + 1, params['charge'], params['index'], atomType) )
            #   convert sigma to angstroms and epsilon to kcal/mol
            vdw_lines.append('vdw  {:4d}  {:10.5f}  {:10.5f}'.format(params['index'], sigma, epsilon))
    
    rule_lines = []
    if ff_mode == 'ipt':
        rule_lines.append('NumAtomTypes {:d}'.format(len(atom_lines)))
    elif ff_mode != 'tinker':
        rule_lines.append('RadiusRule Arithmetic')
        rule_lines.append('EpsilonRule Geometric')
        rule_lines.append('RadiusSize Radius')
        rule_lines.append('ImptorType Trigonometric')
        rule_lines.append('vdw-14-scale 0.5')
        rule_lines.append('chg-14-scale 0.83333')
        rule_lines.append('NAtom {:d}'.format(len(atom_lines)))
        rule_lines.append('Nvdw {:d}'.format(len(vdw_lines)))


    
    #   create bond stretching lines
    bend_lines = []
    for n, k in enumerate(bend_force.k):
        a = paramsForType[bend_force.types1[n][0]]['index']
        b = paramsForType[bend_force.types2[n][0]]['index']
        if a > b:
            a, b = [b, a]
        if ff_mode == 'ipt':
            bend_lines.append('bond  {:4d}  {:4d}  {:10.2f}  {:10.5f}'.format(-a, -b, k/418.4/2.0, bend_force.length[n]*10))
        elif ff_mode == 'tinker':
            bend_lines.append('bond  {:4d}  {:4d}  {:10.2f}  {:10.5f}'.format(a + idx_shift, b + idx_shift, k/418.4/2.0, bend_force.length[n]*10))
        else:
            bend_lines.append('bond  {:4d}  {:4d}  {:10.2f}  {:10.5f}'.format(a, b, k/418.4/2.0, bend_force.length[n]*10))

    #   create angle bending lines
    angle_lines = []
    for n, k in enumerate(angle_force.k):
        a = paramsForType[angle_force.types1[n][0]]['index']
        b = paramsForType[angle_force.types2[n][0]]['index']
        c = paramsForType[angle_force.types3[n][0]]['index']
        if a > c:
            a, c = [c, a]
        k_out =  k/2.0/4.184
        if ff_mode == 'ipt':
            angle_lines.append('angle  {:4d}  {:4d}  {:4d}  {:10.3f}  {:10.5f}'.format(-a, -b, -c, k_out, angle_force.angle[n]*180.0/np.pi))
        elif ff_mode == 'tinker':
            angle_lines.append('angle  {:4d}  {:4d}  {:4d}  {:10.3f}  {:10.5f}'.format(a + idx_shift, b + idx_shift, c + idx_shift, k_out, angle_force.angle[n]*180.0/np.pi))
        else:
            angle_lines.append('angle  {:4d}  {:4d}  {:4d}  {:10.3f}  {:10.5f}'.format(a, b, c, k_out, angle_force.angle[n]*180.0/np.pi))

    #   create torsion lines
    torsion_lines = []
    for n, p in enumerate(torsion_force.proper):
        k = p.k[0]
        if k != 0:
            a = paramsForType[p.types1[0]]['index']
            b = paramsForType[p.types2[0]]['index']
            c = paramsForType[p.types3[0]]['index']
            d = paramsForType[p.types4[0]]['index']
            if a > d:
                a, b, c, d = [d, c, b, a]
            k_out = k/4.184*10
            if ff_mode == 'ipt':
                angle_lines.append('torsion  {:4d}  {:4d}  {:4d}  {:4d}  {:10.4f}  {:10.5f}  1'.format(-a, -b, -c, -d, k_out, p.phase[0]*180.0/np.pi))
            elif ff_mode == 'tinker':
                angle_lines.append('torsion  {:4d}  {:4d}  {:4d}  {:4d}  {:10.4f}  {:10.5f}  1'.format(a + idx_shift, b + idx_shift, c + idx_shift, d + idx_shift, k_out, p.phase[0]*180.0/np.pi))
            else:
                angle_lines.append('torsion  {:4d}  {:4d}  {:4d}  {:4d}  {:10.4f}  {:10.5f}  1'.format(a, b, c, d, k_out, p.phase[0]*180.0/np.pi))

    with open('forcefields/forcefield.prm', 'w') as file: 
        for line in rule_lines:
            file.write(line + '\n')
        for line in atom_lines:
            file.write(line + '\n')
        for line in vdw_lines:
            file.write(line + '\n')
        for line in bend_lines:
            file.write(line + '\n')
        for line in angle_lines:
            file.write(line + '\n')
        for line in torsion_lines:
            file.write(line + '\n')
        for line in chg_lines:
            file.write(line + '\n')




