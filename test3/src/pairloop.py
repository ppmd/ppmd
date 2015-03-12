import numpy as np
import particle
import math
import ctypes
import time
import random
import os
import hashlib
import subprocess

class _base():
    def _unique_name_calc(self):
        '''Return name which can be used to identify the pair loop 
        in a unique way.
        '''
        return self._kernel.name+'_'+self.hexdigest()
        
    def hexdigest(self):
        '''Create unique hex digest'''
        m = hashlib.md5()
        m.update(self._kernel.code+self._code)
        if (self._headers != None):
            for header in self._headers:
                m.update(header)
        return m.hexdigest()
        
    def _generate_header_source(self):
        '''Generate the source code of the header file.

        Returns the source code for the header file.
        '''
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H

        %(INCLUDED_HEADERS)s

        void %(KERNEL_NAME)s_wrapper(int n,%(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME':self._unique_name,
             'INCLUDED_HEADERS':self._included_headers(),
             'KERNEL_NAME':self._kernel.name,
             'ARGUMENTS':self._argnames()}
        return (code % d)
    
    def _included_headers(self):
        '''Return names of included header files.'''
        s = ''
        if (self._headers != None):
            s += '\n'
            for x in self._headers:
                s += '#include \"'+x+'\" \n'
        return s    
    
    def _argnames(self):
        '''Comma separated string of argument name declarations.

        This string of argument names is used in the declaration of 
        the method which executes the pairloop over the grid. 
        If, for example, the pairloop gets passed two particle_dats, 
        then the result will be ``double** arg_000,double** arg_001`.`
        '''
        #argnames = ''
        #for i in range(self._nargs):
        #    argnames += 'double **arg_'+('%03d' % i)+','
        #return argnames[:-1]
        
        argnames = ''
        for i,dat in enumerate(self._particle_dat_dict.items()):
            argnames += 'double *arg_'+('%03d' % i)+','
            
            
            #if (dat[1].dattype  == 'scalar'):
                #argnames += 'double *arg_'+('%03d' % i)+','
            
            #if (dat[1].dattype  == 'array'):        
                #argnames += 'double **arg_'+('%03d' % i)+','

        return argnames[:-1]
        

    def _generate_impl_source(self):
        '''Generate the source code the actual implementation.
        '''

        d = {'UNIQUENAME':self._unique_name,
             'KERNEL_METHODNAME':self._kernel_methodname(),
             'KERNEL':self._kernel.code,
             'ARGUMENTS':self._argnames(),
             'LOC_ARGUMENTS':self._loc_argnames(),
             'KERNEL_NAME':self._kernel.name,
             'KERNEL_ARGUMENT_DECL':self._kernel_argument_declarations()}
        return self._code % d    

    def _kernel_methodname(self):
        '''Construct the name of the kernel method.
        
        Return a string of the form 
        ``inline void kernel_name(double **<arg1>, double *<arg2}, ...) {``
        which is used for defining the name of the kernel method.
        '''
        space = ' '*14
        s = 'inline void '+self._kernel.name+'('
        for var_name_kernel, var_name_state  in self._particle_dat_dict.items():
            #print var_name_kernel, var_name_state.dattype()
            
            if (var_name_state.dattype=='array'):
                s += 'double **'+var_name_kernel+', '
            if (var_name_state.dattype=='scalar'):
                s += 'double *'+var_name_kernel+', '
            
            
        s = s[:-2] + ') {'
        return s           

    def _loc_argnames(self):
        '''Comma separated string of local argument names.

        This string is used in the call to the local kernel. If, for
        example, two particle dats get passed to the pairloop, then
        the result will be ``loc_arg_000,loc_arg_001``. Each of these
        is of required type, see method _kernel_argument_declarations()
        '''
        argnames = ''
        for i in range(self._nargs):
            argnames += 'loc_arg_'+('%03d' % i)+','
        return argnames[:-1]




    def _kernel_argument_declarations(self):
        '''Define and declare the kernel arguments.

        For each argument the kernel gets passed a pointer of type
        ``double* loc_argXXX[2]``. Here ``loc_arg[i]`` with i=0,1 is
        pointer to the data which contains the properties of particle i.
        These properties are stored consecutively in memory, so for a 
        scalar property only ``loc_argXXX[i][0]`` is used, but for a vector
        property the vector entry j of particle i is accessed as 
        ``loc_argXXX[i][j]``.

        This method generates the definitions of the ``loc_argXXX`` variables
        and populates the data to ensure that ``loc_argXXX[i]`` points to
        the correct address in the particle_dats.
        '''
        s = '\n'
        for i,dat in enumerate(self._particle_dat_dict.items()):
            
            
            space = ' '*14
            argname = 'arg_'+('%03d' % i)
            loc_argname = 'loc_'+argname
            
            
            if (dat[1].dattype  == 'scalar'):
                s += space+'double *'+loc_argname+' = '+argname+';\n'
            
            if (dat[1].dattype  == 'array'):
                ncomp = dat[1].ncomp
                s += space+'double *'+loc_argname+'[2];\n'
                s += space+loc_argname+'[0] = '+argname+'+'+str(ncomp)+'*i;\n'
                s += space+loc_argname+'[1] = '+argname+'+'+str(ncomp)+'*j;\n'       
        
        return s 

class PairLoopRapaport_tmp():
    '''
    Class to implement rapaport 14 cell looping.
    '''
    def __init__(self,input_state):
        
        
        
        self._input_state = input_state
        
        '''Construct initial cell list'''
        self._q_list = np.zeros([1 + self._input_state.N() + self._input_state.domain().cell_count()], dtype=ctypes.c_int, order='C')
        self.cell_sort_all()
        

        
        '''Determine cell neighbours'''
        self._cell_map=np.zeros([14*self._input_state.domain().cell_count(),5],dtype=ctypes.c_int, order='C')
        for cp in range(1,1 + self._input_state.domain().cell_count()):
            self._cell_map[(cp-1)*14:(cp*14),...] = self.get_adjacent_cells(cp)
        
        
        
        '''Initialise pair_loop code'''
        self._libpair_loop_LJ = np.ctypeslib.load_library('libpair_loop_LJ.so','.')
        self._libpair_loop_LJ.d_pair_loop_LJ.restype = ctypes.c_int
        
        #void d_pair_loop_LJ(int N, int cell_count, double rc, int* cells, int* q_list, double* pos, double* d_extent, double *accel);
        self._libpair_loop_LJ.d_pair_loop_LJ.argtypes = [ctypes.c_int,
                                                        ctypes.c_int,
                                                        ctypes.c_double,
                                                        ctypes.POINTER(ctypes.c_int),
                                                        ctypes.POINTER(ctypes.c_int),
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.POINTER(ctypes.c_double)]
        
        
        
    def _arg_update(self):    
        self._args = [self._cell_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                self._q_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                self._input_state.positions().Dat().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self._input_state.domain().extent().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self._input_state.accelerations().Dat().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self._input_state.U().ctypes_data()]
        
        
    
        
        
     
    def _update_prepare(self):
        #handle perodic bounadies
        self._input_state.domain().boundary_correct(self._input_state)
        #update cell list
        self.cell_sort_all()
        
        
    def execute(self):
        """
        C version of the pair_locate: Loop over all cells update accelerations and potential engery.
        """
        self._update_prepare()
    
        self._input_state.set_accelerations(0.0)
        self._input_state.reset_U() #causes segfault.....
        
        self._arg_update()
            
        self._libpair_loop_LJ.d_pair_loop_LJ(ctypes.c_int(self._input_state.N()), ctypes.c_int(self._input_state.domain().cell_count()), ctypes.c_double(self._input_state._potential._rc), *self._args)   
        
        
        
    def cell_sort_all(self):
        """
        Construct neighbour list, assigning atoms to cells. Using Rapaport alg.
        """
        for cx in range(1,1+self._input_state.domain().cell_count()):
            self._q_list[self._input_state.N() + cx] = 0
        for ix in range(1,1+self._input_state.N()):
            c = self._input_state.domain().get_cell_lin_index(self._input_state.positions().Dat()[ix-1,])
            
            #print c, self._pos[ix-1,], self._domain._extent*0.5
            self._q_list[ix] = self._q_list[self._input_state.N() + c]
            self._q_list[self._input_state.N() + c] = ix
            
    
       
    def get_adjacent_cells(self,ix):
        """
        Returns the 14 neighbouring cells as linear index.
        
        :arg ix: (int) Input index.
        
        """
         
        cell_list = np.zeros([14,5],dtype=ctypes.c_int, order='C')
        #cell_list_boundary=[]
        
        C = self._input_state.domain().cell_index_tuple(ix)
        
        stencil_map = [
            [0,0,0],
            [1,0,0],
            [0,1,0],
            [1,1,0],
            [1,-1,0],
            [-1,1,1],
            [0,1,1],
            [1,1,1],
            [-1,0,1],
            [0,0,1],
            [1,0,1],
            [-1,-1,1],
            [0,-1,1],
            [1,-1,1]
            ]
        
        for ix in range(14):
            ind = stencil_map[ix]
            
            cell_list[ix,] = self._input_state.domain().cell_index_lin_offset(C+ind)

        return cell_list

################################################################################################################
# RAPAPORT LOOP SERIAL
################################################################################################################

class PairLoopRapaport(_base):
    '''
    Class to implement rapaport 14 cell looping.
    '''
    def __init__(self,input_state):
        
        
        
        self._input_state = input_state
        
        '''Construct initial cell list'''
        self._q_list = np.zeros([1 + self._input_state.N() + self._input_state.domain().cell_count()], dtype=ctypes.c_int, order='C')
        self.cell_sort_all()
        

        
        '''Determine cell neighbours'''
        self._cell_map=np.zeros([14*self._input_state.domain().cell_count(),5],dtype=ctypes.c_int, order='C')
        for cp in range(1,1 + self._input_state.domain().cell_count()):
            self._cell_map[(cp-1)*14:(cp*14),...] = self.get_adjacent_cells(cp)
        
        
        ##########
        # End of Rapaport initialisations.
        ##########
        
        self._temp_dir = './build/'
        if (not os.path.exists(self._temp_dir)):
            os.mkdir(self._temp_dir)
        self._kernel = input_state.potential().kernel()
        self._particle_dat_dict = input_state.potential().datdict(input_state)
        self._nargs = len(self._particle_dat_dict)
        self._headers = input_state.potential().headers()

        self._code_init()
        
        self._unique_name = self._unique_name_calc()
        
        self._library_filename  = self._unique_name +'.so'
        
        if (not os.path.exists(os.path.join(self._temp_dir,self._library_filename))):
            self._create_library()
        self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)


    def _generate_header_source(self):
        '''Generate the source code of the header file.

        Returns the source code for the header file.
        '''
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H

        %(INCLUDED_HEADERS)s
        #define LINIDX_2D(NX,iy,ix)   ((NX)*(iy) + (ix))
        #define sign(x) (((x) < 0 ) ?  -1 : ((x) > 0 ))
        
        void %(KERNEL_NAME)s_wrapper(const int n,const int cell_count, int* cells, int* q_list, double* d_extent,%(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME':self._unique_name,
             'INCLUDED_HEADERS':self._included_headers(),
             'KERNEL_NAME':self._kernel.name,
             'ARGUMENTS':self._argnames()}
        return (code % d)
        
        
    def _code_init(self):
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"
        
        %(KERNEL_METHODNAME)s
        %(KERNEL)s
        }
        
        inline void cell_index_offset(int cp, int cpp_i, int* cell_array, double *d_extent, int* cpp, int *flag, double *offset){
        
            const signed char cell_map[14][3] = {   {0,0,0},
                                                    {1,0,0},
                                                    {0,1,0},
                                                    {1,1,0},
                                                    {1,-1,0},
                                                    {-1,1,1},
                                                    {0,1,1},
                                                    {1,1,1},
                                                    {-1,0,1},
                                                    {0,0,1},
                                                    {1,0,1},
                                                    {-1,-1,1},
                                                    {0,-1,1},
                                                    {1,-1,1}};     
                
                
            int Cz = cp/(cell_array[0]*cell_array[1]) + cell_map[cpp_i][2];
            int Cx = cp %% cell_array[0] + cell_map[cpp_i][0];
            int Cy = (int)((cp - (Cz-1)*(cell_array[0]*cell_array[1]))/(cell_array[0])) + cell_map[cpp_i][1];;

            
            int C0 = Cx %% cell_array[0];    
            int C1 = Cy %% cell_array[0];
            int C2 = Cz %% cell_array[0];
                
            if ((Cx != C0) || (Cy != C1) || (Cz != C2)) { 
                flag[0] = 1;
                offset[0] = ((double)sign(Cx - C0))*d_extent[0];
                offset[1] = ((double)sign(Cy - C1))*d_extent[1];
                offset[2] = ((double)sign(Cz - C2))*d_extent[2];
                
            } else {flag[0] = 0;}
            
            
                
                
                
                
                
                
            return;      
        }    
        
        void %(KERNEL_NAME)s_wrapper(const int n, const int cell_count, int* cells, int* q_list, double* d_extent,%(ARGUMENTS)s) { 
        
            int cpp,i,j,cpp_i,cp;
    
            for(cp = 0; cp < cell_count; cp++){
                for(cpp_i=0; cpp_i<14; cpp_i++){
                
                    
                    
                    
                    
                    
                    
                    cpp = cells[LINIDX_2D(5,cpp_i + (cp*14),0)];
                    
                    const double s0 = cells[LINIDX_2D(5,cpp_i + (cp*14),1)]*d_extent[0];
                    const double s1 = cells[LINIDX_2D(5,cpp_i + (cp*14),2)]*d_extent[1];
                    const double s2 = cells[LINIDX_2D(5,cpp_i + (cp*14),3)]*d_extent[2];
                    
                    
                    
                    double r1[3];
                    
                    i = q_list[n+cp];
                    while (i > 0){
                        j = q_list[n+cpp];
                        while (j > 0){
                            if (cp != cpp || i < j){
        
                                %(KERNEL_ARGUMENT_DECL)s
                                %(KERNEL_NAME)s(%(LOC_ARGUMENTS)s);
                                
                                
                            }
                            j = q_list[j];  
                        }
                        i=q_list[i];
                    }
                }
            }
            
            
            return;
        }        
        
        
        '''
    def _kernel_argument_declarations(self):
        '''Define and declare the kernel arguments.

        For each argument the kernel gets passed a pointer of type
        ``double* loc_argXXX[2]``. Here ``loc_arg[i]`` with i=0,1 is
        pointer to the data which contains the properties of particle i.
        These properties are stored consecutively in memory, so for a 
        scalar property only ``loc_argXXX[i][0]`` is used, but for a vector
        property the vector entry j of particle i is accessed as 
        ``loc_argXXX[i][j]``.

        This method generates the definitions of the ``loc_argXXX`` variables
        and populates the data to ensure that ``loc_argXXX[i]`` points to
        the correct address in the particle_dats.
        '''
        s = '\n'
        for i,dat in enumerate(self._particle_dat_dict.items()):
            
            
            space = ' '*14
            argname = 'arg_'+('%03d' % i)
            loc_argname = 'loc_'+argname
            
            
            if (dat[1].dattype  == 'scalar'):
                s += space+'double *'+loc_argname+' = '+argname+';\n'
            
            if (dat[1].dattype  == 'array'):
                if (dat[1].name  == 'positions'):
                    s += space+'double *'+loc_argname+'[2];\n'
                    
                    
                    s += space+'if (cells[LINIDX_2D(5,cpp_i + (cp*14),4)] > 0){ \n'

                    #s += space+'double r1[3];\n'
                    s += space+'r1[0] ='+argname+'[LINIDX_2D(3,j-1,0)] + s0; \n'
                    s += space+'r1[1] ='+argname+'[LINIDX_2D(3,j-1,1)] + s1; \n'
                    s += space+'r1[2] ='+argname+'[LINIDX_2D(3,j-1,2)] + s2; \n'
                    s += space+loc_argname+'[1] = r1;\n'
                    
                    s += space+'}else{ \n'
                    s += space+loc_argname+'[1] = '+argname+'+3*(j-1);\n' 
                    s += space+'} \n'
                    s += space+loc_argname+'[0] = '+argname+'+3*(i-1);\n'
                    
                else:
                    ncomp = dat[1].ncomp
                    s += space+'double *'+loc_argname+'[2];\n'
                    s += space+loc_argname+'[0] = '+argname+'+'+str(ncomp)+'*(i-1);\n'
                    s += space+loc_argname+'[1] = '+argname+'+'+str(ncomp)+'*(j-1);\n'       
        
        return s       
        
     
    def _update_prepare(self):
        #handle perodic bounadies
        self._input_state.domain().boundary_correct(self._input_state)
        #update cell list
        self.cell_sort_all()
        
        
    def execute(self):
        """
        C version of the pair_locate: Loop over all cells update accelerations and potential engery.
        """
        self._update_prepare()
    
        self._input_state.set_accelerations(ctypes.c_double(0.0))
        self._input_state.reset_U() 
        
        args = [self._cell_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        self._q_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        self._input_state.domain().extent().ctypes.data_as(ctypes.POINTER(ctypes.c_double))]
                
        for dat in self._particle_dat_dict.values():
            args.append(dat.ctypes_data())
            
        method = self._lib[self._kernel.name+'_wrapper']
        
        method(ctypes.c_int(self._input_state.N()), ctypes.c_int(self._input_state.domain().cell_count()), *args)           
        
        
        
        
        
        
        #self._arg_update()
            
        #self._libpair_loop_LJ.d_pair_loop_LJ(ctypes.c_int(self._input_state.N()), ctypes.c_int(self._input_state.domain().cell_count()), ctypes.c_double(self._input_state._potential._rc), *self._args)   
        
        
        
    def cell_sort_all(self):
        """
        Construct neighbour list, assigning atoms to cells. Using Rapaport alg.
        """
        for cx in range(1,1+self._input_state.domain().cell_count()):
            self._q_list[self._input_state.N() + cx] = 0
        for ix in range(1,1+self._input_state.N()):
            c = self._input_state.domain().get_cell_lin_index(self._input_state.positions().Dat()[ix-1,])
            
            #print c, self._pos[ix-1,], self._domain._extent*0.5
            self._q_list[ix] = self._q_list[self._input_state.N() + c]
            self._q_list[self._input_state.N() + c] = ix
            
    
       
    def get_adjacent_cells(self,ix):
        """
        Returns the 14 neighbouring cells as linear index.
        
        :arg ix: (int) Input index.
        
        """
         
        cell_list = np.zeros([14,5],dtype=ctypes.c_int, order='C')
        #cell_list_boundary=[]
        
        C = self._input_state.domain().cell_index_tuple(ix)
        
        stencil_map = [
            [0,0,0],
            [0,1,0],
            [0,0,1],
            [0,1,1],
            [0,-1,1],
            [1,0,0],
            [1,0,1],
            [-1,0,1],
            [1,1,0],
            [1,-1,0],
            [1,1,1],
            [1,-1,1],
            [-1,1,1],
            [-1,-1,1]
            ]
        
        for ix in range(14):
            ind = stencil_map[ix]
            
            cell_list[ix,] = self._input_state.domain().cell_index_lin_offset(C+ind)

        return cell_list
     
     
    def _create_library(self):
        '''
        Create a shared library from the source code.
        '''
        
        filename_base = os.path.join(self._temp_dir,self._unique_name)
        header_filename = filename_base+'.h'
        impl_filename = filename_base+'.c'
        with open(header_filename,'w') as f:
            print >> f, self._generate_header_source()        
        with open(impl_filename,'w') as f:
            print >> f, self._generate_impl_source()
        object_filename = filename_base+'.o'
        library_filename = filename_base+'.so'        
        cflags = ['-O3','-fpic']
        cc = 'gcc'
        ld = 'gcc'
        compile_cmd = [cc,'-c','-fpic']+cflags+['-I',self._temp_dir] \
                       +['-o',object_filename,impl_filename]
        link_cmd = [ld,'-shared']+['-o',library_filename,object_filename]
        stdout_filename = filename_base+'.log'
        stderr_filename = filename_base+'.err'
        with open(stdout_filename,'w') as stdout:
            with open(stderr_filename,'w') as stderr:
                stdout.write('Compilation command:\n')
                stdout.write(' '.join(compile_cmd))
                stdout.write('\n\n')
                p = subprocess.Popen(compile_cmd,
                                     stdout=stdout,
                                     stderr=stderr)
                p.communicate()
                stdout.write('Link command:\n')
                stdout.write(' '.join(link_cmd))
                stdout.write('\n\n')
                p = subprocess.Popen(link_cmd,
                                     stdout=stdout,
                                     stderr=stderr)
                p.communicate()      
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
        
################################################################################################################
# SINGLE PARTICLE LOOP SERIAL
################################################################################################################
class SingleAllParticleLoop():
    """
    Class to loop over all particles once.
    """
    def __init__(self, kernel, particle_dat_dict, headers=None):
        self._temp_dir = './build/'
        if (not os.path.exists(self._temp_dir)):
            os.mkdir(self._temp_dir)
        self._kernel = kernel
        self._particle_dat_dict = particle_dat_dict
        self._nargs = len(self._particle_dat_dict)
        self._headers = headers

        self._code_init()
        
        self._unique_name = self._unique_name_calc()
        
        self._library_filename  = self._unique_name +'.so'
        
        if (not os.path.exists(os.path.join(self._temp_dir,self._library_filename))):
            self._create_library()
        self._lib = np.ctypeslib.load_library(self._library_filename, self._temp_dir)

     
        

    def _create_library(self):
        '''
        Create a shared library from the source code.
        '''
        
        filename_base = os.path.join(self._temp_dir,self._unique_name)
        header_filename = filename_base+'.h'
        impl_filename = filename_base+'.c'
        with open(header_filename,'w') as f:
            print >> f, self._generate_header_source()        
        with open(impl_filename,'w') as f:
            print >> f, self._generate_impl_source()
        object_filename = filename_base+'.o'
        library_filename = filename_base+'.so'        
        cflags = ['-O3','-fpic']
        cc = 'gcc'
        ld = 'gcc'
        compile_cmd = [cc,'-c','-fpic']+cflags+['-I',self._temp_dir] \
                       +['-o',object_filename,impl_filename]
        link_cmd = [ld,'-shared']+['-o',library_filename,object_filename]
        stdout_filename = filename_base+'.log'
        stderr_filename = filename_base+'.err'
        with open(stdout_filename,'w') as stdout:
            with open(stderr_filename,'w') as stderr:
                stdout.write('Compilation command:\n')
                stdout.write(' '.join(compile_cmd))
                stdout.write('\n\n')
                p = subprocess.Popen(compile_cmd,
                                     stdout=stdout,
                                     stderr=stderr)
                p.communicate()
                stdout.write('Link command:\n')
                stdout.write(' '.join(link_cmd))
                stdout.write('\n\n')
                p = subprocess.Popen(link_cmd,
                                     stdout=stdout,
                                     stderr=stderr)
                p.communicate()                
        
    def execute(self):
        '''Execute the kernel over all particle pairs.'''
        args = []
        for dat in self._particle_dat_dict.values():
            args.append(dat.ctypes_data())
            n = dat.npart
        method = self._lib[self._kernel.name+'_wrapper']
        method(n,*args)   
    

       
    def _generate_impl_source(self):
        '''Generate the source code the actual implementation.
        '''

        d = {'UNIQUENAME':self._unique_name,
             'KERNEL_METHODNAME':self._kernel_methodname(),
             'KERNEL':self._kernel.code,
             'ARGUMENTS':self._argnames(),
             'LOC_ARGUMENTS':self._loc_argnames(),
             'KERNEL_NAME':self._kernel.name,
             'KERNEL_ARGUMENT_DECL':self._kernel_argument_declarations()}
        return self._code % d        
        
    
    def _kernel_methodname(self):
        '''Construct the name of the kernel method.
        
        Return a string of the form 
        ``inline void kernel_name(double *<arg1>, double *<arg2}, ...) {``
        which is used for defining the name of the kernel method.
        '''
        space = ' '*14
        s = 'inline void '+self._kernel.name+'('
        for var_name in self._particle_dat_dict.keys():
            s += 'double *'+var_name+', '
        s = s[:-2] + ') {'
        return s

    def _loc_argnames(self):
        '''Comma separated string of local argument names.

        This string is used in the call to the local kernel. If, for
        example, two particle dats get passed to the pairloop, then
        the result will be ``loc_arg_000,loc_arg_001``. Each of these
        is of type ``double* [2]``, see method _kernel_argument_declarations()
        '''
        argnames = ''
        for i in range(self._nargs):
            argnames += 'loc_arg_'+('%03d' % i)+','
        return argnames[:-1]

    def _code_init(self):
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

        %(KERNEL_METHODNAME)s
        %(KERNEL)s
        }

        void %(KERNEL_NAME)s_wrapper(const int n,%(ARGUMENTS)s) { 
          int i;
          for (i=0; i<n; ++i) {
              %(KERNEL_ARGUMENT_DECL)s
              %(KERNEL_NAME)s(%(LOC_ARGUMENTS)s);
            }
        }
        '''
    
    def _kernel_argument_declarations(self):
        '''Define and declare the kernel arguments.

        For each argument the kernel gets passed a pointer of type
        ``double* loc_argXXX[2]``. Here ``loc_arg[i]`` with i=0,1 is
        pointer to the data which contains the properties of particle i.
        These properties are stored consecutively in memory, so for a 
        scalar property only ``loc_argXXX[i][0]`` is used, but for a vector
        property the vector entry j of particle i is accessed as 
        ``loc_argXXX[i][j]``.

        This method generates the definitions of the ``loc_argXXX`` variables
        and populates the data to ensure that ``loc_argXXX[i]`` points to
        the correct address in the particle_dats.
        '''
        s = '\n'
        for i,dat in enumerate(self._particle_dat_dict.values()):
            
            
            ncomp = dat.ncomp
            
            
            space = ' '*14
            argname = 'arg_'+('%03d' % i)
            loc_argname = 'loc_'+argname
            #s += space+'double *'+loc_argname+'[2];\n'
            s += space+'double *'+loc_argname+';\n'
            s += space+loc_argname+' = '+argname+'+'+str(ncomp)+'*i;\n'
            #s += space+loc_argname+' = &'+argname+'['+str(ncomp)+'*i];\n'
            
            #s += space+loc_argname+'[1] = '+argname+'+'+str(ncomp)+'*j;\n'
        return s


        
    def _argnames(self):
        '''Comma separated string of argument name declarations.

        This string of argument names is used in the declaration of 
        the method which executes the pairloop over the grid. 
        If, for example, the pairloop gets passed two particle_dats, 
        then the result will be ``double* arg_000,double* arg_001`.`
        '''
        argnames = ''
        for i in range(self._nargs):
            argnames += 'double *arg_'+('%03d' % i)+','
        return argnames[:-1]        
        
        
        
    def _generate_header_source(self):
        '''Generate the source code of the header file.

        Returns the source code for the header file.
        '''
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H

        %(INCLUDED_HEADERS)s

        void %(KERNEL_NAME)s_wrapper(int n,%(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME':self._unique_name,
             'INCLUDED_HEADERS':self._included_headers(),
             'KERNEL_NAME':self._kernel.name,
             'ARGUMENTS':self._argnames()}
        return (code % d)

    def _included_headers(self):
        '''Return names of included header files.'''
        s = ''
        if (self._headers != None):
            s += '\n'
            for x in self._headers:
                s += '#include \"'+x+'\" \n'
        return s
        

    def _unique_name_calc(self):
        '''Return name which can be used to identify the pair loop 
        in a unique way.
        '''
        return self._kernel.name+'_'+self.hexdigest()
        
    def hexdigest(self):
        '''Create unique hex digest'''
        m = hashlib.md5()
        m.update(self._kernel.code+self._code)
        if (self._headers != None):
            for header in self._headers:
                m.update(header)
        return m.hexdigest()

################################################################################################################
# SINGLE PARTICLE LOOP OPENMP
################################################################################################################

class SingleAllParticleLoopOpenMP(SingleAllParticleLoop):
    """
    OpenMP version of single pass pair loop
    """
    SingleAllParticleLoop._create_library
    
    def _code_init(self):
        self._code = '''
        #include \"%(UNIQUENAME)s.h\"
        #include <omp.h>

        %(KERNEL_METHODNAME)s
        %(KERNEL)s
        }

        void %(KERNEL_NAME)s_wrapper(const int n,%(ARGUMENTS)s) { 
          int i;
          #pragma omp parallel for
          for (i=0; i<n; ++i) {
              %(KERNEL_ARGUMENT_DECL)s
              %(KERNEL_NAME)s(%(LOC_ARGUMENTS)s);
            }
        }
        '''
    
          
    
    def _create_library(self):
        '''
        Create a shared library from the source code.
        '''
        
        filename_base = os.path.join(self._temp_dir,self._unique_name)
        header_filename = filename_base+'.h'
        impl_filename = filename_base+'.c'
        with open(header_filename,'w') as f:
            print >> f, self._generate_header_source()        
        with open(impl_filename,'w') as f:
            print >> f, self._generate_impl_source()
        object_filename = filename_base+'.o'
        library_filename = filename_base+'.so'        
        cflags = ['-O3','-fpic','-fopenmp','-lgomp','-lpthread','-lc','-lrt']
        cc = 'gcc'
        ld = 'gcc'
        link_flags = ['-fopenmp','-lgomp','-lpthread','-lc','-lrt']
        compile_cmd = [cc,'-c','-fpic']+cflags+['-I',self._temp_dir] \
                       +['-o',object_filename,impl_filename]
        link_cmd = [ld,'-shared']+link_flags+['-o',library_filename,object_filename]
        stdout_filename = filename_base+'.log'
        stderr_filename = filename_base+'.err'
        with open(stdout_filename,'w') as stdout:
            with open(stderr_filename,'w') as stderr:
                stdout.write('Compilation command:\n')
                stdout.write(' '.join(compile_cmd))
                stdout.write('\n\n')
                p = subprocess.Popen(compile_cmd,
                                     stdout=stdout,
                                     stderr=stderr)
                p.communicate()
                stdout.write('Link command:\n')
                stdout.write(' '.join(link_cmd))
                stdout.write('\n\n')
                p = subprocess.Popen(link_cmd,
                                     stdout=stdout,
                                     stderr=stderr)
                p.communicate() 
        
        
        


















        
    
    
