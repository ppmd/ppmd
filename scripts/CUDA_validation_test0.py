#!/usr/bin/python
import os
import numpy as np
import ctypes


from ppmd import *
from ppmd.cuda import *

# debug level
cuda_runtime.DEBUG.level = 0
#verbosity level
cuda_runtime.VERBOSE.level = 2
#timer level
runtime.TIMER.level = 3
#build timer level
runtime.BUILD_TIMER.level = 3


#cuda on/off
runtime.CUDA_ENABLED.flag = True



if __name__ == '__main__':

    file_dir = './'


    # open the field file
    fh = open(file_dir + '/FIELD', 'r')
    _field = fh.read().split()
    fh.close()

    # get number of moecules from field file
    N = int(_field[_field.index('NUMMOLS') + 1])
    print 'N:', N

    epsilon = float(_field[_field.index('lj') + 1])
    sigma = float(_field[_field.index('lj') + 2])

    print "LJ: epsilon:", epsilon, "sigma:", sigma


    mass = float(_field[_field.index('Ar') + 1])


    # open the control file
    fh = open(file_dir + '/CONTROL', 'r')
    _control = fh.read().split()
    fh.close()

    dt = float(_control[_control.index('timestep') + 1])
    t = float(_control[_control.index('steps') + 1]) * dt
    rc = float(_control[_control.index('cutoff') + 1])

    print "rc =", rc

    # Initialise basic domain
    test_domain = domain.BaseDomainHalo()

    # Initialise LJ potential
    test_potential = potential.LennardJones(sigma=sigma,epsilon=epsilon, rc=rc)

    # Initialise masses
    test_mass_init = simulation.MassInitIdentical(mass)

    # Initialise positions and velocities
    test_pos_init = simulation.PosInitDLPOLYConfig(file_dir + '/CONFIG')
    test_vel_init = simulation.VelInitDLPOLYConfig(file_dir + '/CONFIG')

    # Create simulation class from above initialisations.
    sim1 = simulation.BaseMDSimulation(domain_in=test_domain,
                                       potential_in=test_potential,
                                       particle_pos_init=test_pos_init,
                                       particle_vel_init=test_vel_init,
                                       particle_mass_init=test_mass_init,
                                       n=N,
                                       setup_only=True,
                                       )


    fio.ParticleDat_to_xml(sim1.state.positions, file_dir + 'gpu_ppmd_x0.xml')

    # Check ParticleDat dump is correct
    test = fio.xml_to_ParticleDat(file_dir + 'gpu_ppmd_x0.xml')
    for ix in range(N):
        assert np.all(test.data[ix,0:3:] == sim1.state.positions.data[ix,0:3:])



    COM = cuda_cell.CellOccupancyMatrix()

    # Create a particle dat with the positions in from the sim1.state
    sim1.state.d_positions = cuda_data.ParticleDat(initial_value=sim1.state.positions, name='positions')
    sim1.state.d_forces = cuda_data.ParticleDat(initial_value=sim1.state.forces, name='forces')


    # This what the masses should end up being
    #sim1.state.d_mass = cuda_data.TypedDat(initial_value=sim1.state.mass.data, name='mass')


    sim1.state.d_mass = cuda_data.ParticleDat(initial_value=mass * np.ones([N,1], dtype=ctypes.c_double), name='mass')

    sim1.state.d_velocities = cuda_data.ParticleDat(initial_value=sim1.state.velocities, name='velocities')


    sim1.state.d_u = cuda_data.ScalarArray(ncomp=1, dtype=ctypes.c_double, name='potential_energy')
    sim1.state.d_k = cuda_data.ScalarArray(ncomp=1, dtype=ctypes.c_double, name='ke')
    sim1.state.h_k = data.ScalarArray(ncomp=1, dtype=ctypes.c_double, name='ke')
    sim1.state.h_u = data.ScalarArray(ncomp=1, dtype=ctypes.c_double, name='potential_energy')


    # print sim1.state.positions.max_npart, sim1.state.positions.npart_local, sim1.state.positions.data

    COM.setup(sim1.state.as_func('npart_local'), sim1.state.d_positions, sim1.state.domain)

    COM.sort()
    cuda_halo.HALOS = cuda_halo.CartesianHalo(COM)



    sim1.state.d_positions.halo_exchange()





    neighbour_list = cuda_cell.NeighbourListLayerBased(COM, 1.1 * test_potential.rc)
    neighbour_list.update()


    dat_map = {'P': sim1.state.d_positions(access.R), 'A': sim1.state.d_forces(access.INC0), 'u': sim1.state.d_u(access.INC0)}

    pair_loop = cuda_pairloop.PairLoopNeighbourList(kernel_in=test_potential.kernel, #_gpu.kernel,
                                                    particle_dat_dict=dat_map,
                                                    neighbour_list=neighbour_list)
    print "n =", sim1.state.d_positions.npart_local


    # integration definitons -------------------------------------------

    vv1_code = '''
    //self._V+=0.5*self._dt*self._A
    //self._P+=self._dt*self._V
    const double M_tmp = 1./M(0);

    //const double M_tmp = 1.0;

    V(0) += dht*A(0)*M_tmp;
    V(1) += dht*A(1)*M_tmp;
    V(2) += dht*A(2)*M_tmp;
    P(0) += dt*V(0);
    P(1) += dt*V(1);
    P(2) += dt*V(2);
    '''

    t_vv1_code = '''
    //self._V+=0.5*self._dt*self._A
    //self._P+=self._dt*self._V
    const double M_tmp = 1./M(0);

    //const double M_tmp = 1.0;

    double3 v = {V(0), V(1), V(2)};
    double3 a = {A(0), A(1), A(2)};
    double3 p = {P(0), P(1), P(2)};

    V(0) = __fma_rn(dht*M_tmp, a.x, v.x);
    V(1) = __fma_rn(dht*M_tmp, a.y, v.y);
    V(2) = __fma_rn(dht*M_tmp, a.z, v.z);

    P(0) = __fma_rn(dt, v.x, p.x);
    P(1) = __fma_rn(dt, v.y, p.y);
    P(2) = __fma_rn(dt, v.z, p.z);

    '''



    vv1_map = {'V': sim1.state.d_velocities(access.RW), 'P': sim1.state.d_positions(access.RW), 'A': sim1.state.d_forces(access.R), 'M': sim1.state.d_mass(access.R)}
    vv1_constants = [kernel.Constant('dt',dt), kernel.Constant('dht',0.5 * dt),]







    vv2_code = '''
    //self._V.data()(...,...)+= 0.5*self._dt*self._A.data
    const double M_tmp = 1/M(0);
    //const double M_tmp = 1.0;
    V(0) += dht*A(0)*M_tmp;
    V(1) += dht*A(1)*M_tmp;
    V(2) += dht*A(2)*M_tmp;
    '''


    vv2_map = {'V': sim1.state.d_velocities(access.RW), 'A': sim1.state.d_forces(access.R), 'M': sim1.state.d_mass(access.R)}



    ke_code = '''
    K(0) += 0.5 * M(0) * ( V(0)*V(0) + V(1)*V(1) + V(2)*V(2) );

    '''

    ke_map = {'V': sim1.state.d_velocities(access.R), 'M': sim1.state.d_mass, 'K': sim1.state.d_k(access.INC0)}






    BC_flag = cuda_data.ScalarArray(ncomp=1, dtype=ctypes.c_int)
    h_BC_flag = data.ScalarArray(ncomp=1, dtype=ctypes.c_int)

    _one_proc_pbc_code = '''

    int _F = 0;

    //printf("BEFORE ID %d | x=%f y=%f z=%f \\n", _ix, P(0), P(1), P(2));
    if (abs_md(P(0)) > 0.5*E0){
        const double E0_2 = 0.5*E0;
        const double x = P(0) + E0_2;


        if (x < 0){
            P(0) = (E0 - fmod(abs_md(x) , E0)) - E0_2;
            _F = 1;
        }
        else{
            P(0) = fmod( x , E0 ) - E0_2;
            _F = 1;
        }
    }

    if (abs_md(P(1)) > 0.5*E1){
        const double E1_2 = 0.5*E1;
        const double x = P(1) + E1_2;

        if (x < 0){
            P(1) = (E1 - fmod(abs_md(x) , E1)) - E1_2;
            _F = 1;
        }
        else{
            P(1) = fmod( x , E1 ) - E1_2;
            _F = 1;
        }
    }

    if (abs_md(P(2)) > 0.5*E2){
        const double E2_2 = 0.5*E2;
        const double x = P(2) + E2_2;

        if (x < 0){
            P(2) = (E2 - fmod(abs_md(x) , E2)) - E2_2;
            _F = 1;
        }
        else{
            P(2) = fmod( x , E2 ) - E2_2;
            _F = 1;
        }
    }

    if (_F > 0){
        //printf("f \\n");
        BCFLAG(0) = BCFLAG(0) || _F;
    }

    //printf("AFTER ID %d | x=%f y=%f z=%f \\n", _ix, P(0), P(1), P(2));
    '''


    _one_proc_pbc_kernel = kernel.Kernel('_one_proc_pbc_kernel', _one_proc_pbc_code, None, static_args={'E0':ctypes.c_double, 'E1':ctypes.c_double, 'E2':ctypes.c_double})
    _one_process_pbc_lib = cuda_loop.ParticleLoop(_one_proc_pbc_kernel, {'P': sim1.state.d_positions(access.RW), 'BCFLAG':BC_flag})



    vv1 = cuda_loop.ParticleLoop(kernel.Kernel('vv1', vv1_code, vv1_constants), vv1_map)
    vv2 = cuda_loop.ParticleLoop(kernel.Kernel('vv2', vv2_code, vv1_constants), vv2_map)

    ke = cuda_loop.ParticleLoop(kernel.Kernel('ke', ke_code), ke_map)


    per_printer = method.PercentagePrinter(dt,t,10)
    # Some running ---------------------------------------------


    _E = sim1.state.domain.extent
    #_one_process_pbc_lib.execute(n=sim1.state.d_positions.npart_local, static_args={'E0':ctypes.c_double(_E.data[0]), 'E1':ctypes.c_double(_E.data[1]), 'E2':ctypes.c_double(_E.data[2])})




    timer = runtime.Timer(runtime.Level(1), 0)





    _one_process_pbc_lib.execute(n=sim1.state.d_positions.npart_local,
                                 static_args={'E0':ctypes.c_double(_E.data[0]), 'E1':ctypes.c_double(_E.data[1]), 'E2':ctypes.c_double(_E.data[2])})


    print "START"

    pair_loop.execute(n=sim1.state.d_positions.npart_local)

    timer.start()
    for ix in range(int(t / dt)):

        vv1.execute(n=sim1.state.d_positions.npart_local)


        # boundary conditions here.
        BC_flag.zero()
        _one_process_pbc_lib.execute(n=sim1.state.d_positions.npart_local,
                                     static_args={'E0':ctypes.c_double(_E.data[0]), 'E1':ctypes.c_double(_E.data[1]), 'E2':ctypes.c_double(_E.data[2])})
        cuda_runtime.cuda_mem_cpy(h_BC_flag.ctypes_data, BC_flag.ctypes_data, ctypes.c_size_t(ctypes.sizeof(ctypes.c_int)), 'cudaMemcpyDeviceToHost')

        if ((ix % 10) == 0) or (h_BC_flag.data[0] > 0):
            COM.sort()
        sim1.state.d_positions.halo_exchange()
        if ((ix % 10) == 0) or (h_BC_flag.data[0] > 0):
            neighbour_list.update()


        sim1.state.d_u.zero()
        pair_loop.execute(n=sim1.state.d_positions.npart_local)

        vv2.execute(n=sim1.state.d_positions.npart_local)

        if ix == 0:
            sim1.state.d_k.zero()
            ke.execute(n=sim1.state.d_positions.npart_local)

            cuda_runtime.cuda_mem_cpy(sim1.state.h_k.ctypes_data, sim1.state.d_k.ctypes_data, ctypes.c_size_t(ctypes.sizeof(ctypes.c_double)), 'cudaMemcpyDeviceToHost')
            cuda_runtime.cuda_mem_cpy(sim1.state.h_u.ctypes_data, sim1.state.d_u.ctypes_data, ctypes.c_size_t(ctypes.sizeof(ctypes.c_double)), 'cudaMemcpyDeviceToHost')
            print sim1.state.h_k.data[0], 0.5 * sim1.state.h_u.data[0], sim1.state.h_k.data[0] + 0.5 * sim1.state.h_u.data[0]

        per_printer.tick()

    sim1.state.d_k.zero()
    ke.execute(n=sim1.state.d_positions.npart_local)

    cuda_runtime.cuda_mem_cpy(sim1.state.h_k.ctypes_data, sim1.state.d_k.ctypes_data, ctypes.c_size_t(ctypes.sizeof(ctypes.c_double)), 'cudaMemcpyDeviceToHost')
    cuda_runtime.cuda_mem_cpy(sim1.state.h_u.ctypes_data, sim1.state.d_u.ctypes_data, ctypes.c_size_t(ctypes.sizeof(ctypes.c_double)), 'cudaMemcpyDeviceToHost')
    print sim1.state.h_k.data[0], 0.5 * sim1.state.h_u.data[0], sim1.state.h_k.data[0] + 0.5 * sim1.state.h_u.data[0]



    timer.stop("GPU time")



    cuda_runtime.cuda_mem_cpy(sim1.state.positions.ctypes_data, sim1.state.d_positions.ctypes_data, ctypes.c_size_t(N * 3 * ctypes.sizeof(ctypes.c_double)), 'cudaMemcpyDeviceToHost')




    fio.ParticleDat_to_xml(sim1.state.positions, file_dir + 'gpu_ppmd_x1.xml')

    # check ParticleDat dump is correct.
    test = fio.xml_to_ParticleDat(file_dir + 'gpu_ppmd_x1.xml')
    for ix in range(N):
        assert np.all(test.data[ix,0:3:] == sim1.state.positions.data[ix,0:3:])








