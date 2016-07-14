#!/usr/bin/python

from ppmd import *

# debug level
runtime.DEBUG.level = 0
#verbosity level
runtime.VERBOSE.level = 3
#timer level
runtime.TIMER.level = 3
#build timer level
runtime.BUILD_TIMER.level = 3

runtime.OPT.level = 3

#cuda on/off
runtime.CUDA_ENABLED.flag = False




if __name__ == '__main__':

    file_dir = './'

    _fpprint = pio.pfprint(dirname=file_dir)

    # open the field file
    fh = open(file_dir + '/FIELD', 'r')
    _field = fh.read().split()
    fh.close()

    # get number of moecules from field file
    N = int(_field[_field.index('NUMMOLS') + 1])

    _fpprint.pprint('N:', N)

    epsilon = float(_field[_field.index('lj') + 1])
    sigma = float(_field[_field.index('lj') + 2])

    _fpprint.pprint("LJ: epsilon:", epsilon, "sigma:", sigma)

    mass = float(_field[_field.index('Ar') + 1])

    # open the control file
    fh = open(file_dir + '/CONTROL', 'r')
    _control = fh.read().split()
    fh.close()

    dt = float(_control[_control.index('timestep') + 1])
    t = float(_control[_control.index('steps') + 1]) * dt
    rc = float(_control[_control.index('cutoff') + 1])

    _fpprint.pprint("rc =", rc)

    # Initialise basic domain
    test_domain = domain.BaseDomainHalo()

    # Initialise LJ potential
    test_potential = potential.LennardJones(sigma=sigma,epsilon=epsilon, rc=rc)
    #test_potential = potential.TestPotential3(sigma=sigma,epsilon=epsilon, rc=rc)

    # Initialise masses
    test_mass_init = simulation.MassInitIdentical(mass)

    # Initialise positions and velocities
    test_pos_init = simulation.PosInitDLPOLYConfig(file_dir + '/CONFIG')
    test_vel_init = simulation.VelInitDLPOLYConfig(file_dir + '/CONFIG')

    _pairloop = pairloop.PairLoopNeighbourList
    # _pairloop = pairloop.PairLoopRapaportHalo


    sim1 = simulation.BaseMDSimulation(domain_in=test_domain,
                                       potential_in=test_potential,
                                       particle_pos_init=test_pos_init,
                                       particle_vel_init=test_vel_init,
                                       particle_mass_init=test_mass_init,
                                       n=N,
                                       pairloop_in=_pairloop
                                       )

    # Helper methods
    per_printer = method.PercentagePrinter(dt, t, 10)

    schedule = method.Schedule([1], [per_printer.tick])

    # Create an integrator for above state class.
    test_integrator = method.VelocityVerlet(simulation=sim1, schedule=schedule)

    if mpi.MPI_HANDLE.nproc == 1:
        fio.ParticleDat_to_xml(sim1.state.positions, file_dir + 'ppmd_x0.xml')
    else:
        fio.MPIParticleDat_to_xml(sim1.state.positions, file_dir + 'ppmd_x0.xml', sim1.state.global_ids)


    ###########################################################

    _fpprint.pprint("t", t, "dt", dt)
    test_integrator.integrate(dt=dt, t=t)

    ###########################################################

    if mpi.MPI_HANDLE.nproc == 1:
        fio.ParticleDat_to_xml(sim1.state.positions, file_dir + 'ppmd_x1.xml')
    else:
        fio.MPIParticleDat_to_xml(sim1.state.positions, file_dir + 'ppmd_x1.xml', sim1.state.global_ids)


    _fpprint.pprint('\n--------- GENERAL TIMING ----------\n')

    _fpprint.pprint("Velocity Verlet: \t\t\t", test_integrator.timer.time())
    _fpprint.pprint("Force update total: \t\t\t", sim1.cpu_forces_timer.time())
    _fpprint.pprint("Boundary conditions: \t\t\t", sim1.boundary_method_timer.time())
    _fpprint.pprint("Halo exchange: \t\t\t\t", sim1.state.positions.timer_comm.time())



    _fpprint.pprint('\n--------- PAIRLOOP TIMING ---------\n')

    _fpprint.pprint("PairLoop timer resolution: \t", opt.get_timer_accuracy(), "s")
    _fpprint.pprint("PairLoop cputime (T/Np): \t",
                    sim1._forces_update_lib.loop_timer.cpu_time,
                    '(',
                    sim1._forces_update_lib.loop_timer.cpu_time/mpi.MPI_HANDLE.nproc,
                    ')'
                    )

    _fpprint.pprint("Cell list: \t\t\t", cell.cell_list.timer_sort.time())

    if issubclass(type(sim1._forces_update_lib), pairloop.PairLoopNeighbourList):
        _fpprint.pprint("Neighbour list: \t\t",
                        sim1._forces_update_lib.neighbour_list.timer_update.time())
        _fpprint.pprint("Neighbour list execution: \t\t",
                        sim1._forces_update_lib.neighbour_list._neighbour_lib.execute_timer.time())


    _fpprint.pprint('\n---- BOUNDARY CONDITION TIMING ----\n')

    _fpprint.pprint("Apply: \t\t", sim1._boundary_method.timer_apply.time())
    _fpprint.pprint("Lib overhead: \t", sim1._boundary_method.timer_lib_overhead.time())
    _fpprint.pprint("Search: \t", sim1._boundary_method.timer_search.time())
    _fpprint.pprint("Move: \t\t", sim1._boundary_method.timer_move.time())

    _fpprint.pprint('\n----------- HALO TIMING -----------\n')

    _fpprint.pprint("Total: \t\t", sim1.state.positions.timer_comm.time())
    _fpprint.pprint("Pack: \t\t", sim1.state.positions.timer_pack.time())
    _fpprint.pprint("Transfer: \t", sim1.state.positions.timer_transfer.time())



    _fpprint.close()





