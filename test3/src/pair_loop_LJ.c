#include "pair_loop_LJ.h"
#include <math.h>
#include <stdio.h>

#define LINIDX_2D(NX,iy,ix)   ((NX)*(iy) + (ix))


void d_pair_loop_LJ(int N, int cell_count, double rc, int* cells, int* q_list, double* pos, double* d_extent, double *accel, double* U){

    
   
    
    //rapaport vars
    
    
    double rc2 = rc*rc;
    
    
    

    // LJ variables
    
    
    
    int cpp,ip,ipp,cpp_i,cp;
    
    for(cp = 1; cp < cell_count+1; cp++){
        for(cpp_i=0; cpp_i<14; cpp_i++){
            cpp = cells[LINIDX_2D(5,cpp_i + ((cp-1)*14),0)];
            ip = q_list[N+cp];
            while (ip > 0){
                ipp = q_list[N+cpp];
                while (ipp > 0){
                    if (cp != cpp || ip < ipp){
                    
                    
                    
                        double rv[3];
                        if (cells[LINIDX_2D(5,cpp_i + ((cp-1)*14),4)] > 0){
                        
                            rv[0] = pos[LINIDX_2D(3,ipp-1,0)] - pos[LINIDX_2D(3,ip-1,0)] + cells[LINIDX_2D(5,cpp_i + ((cp-1)*14),1)]*d_extent[0];
                            rv[1] = pos[LINIDX_2D(3,ipp-1,1)] - pos[LINIDX_2D(3,ip-1,1)] + cells[LINIDX_2D(5,cpp_i + ((cp-1)*14),2)]*d_extent[1];
                            rv[2] = pos[LINIDX_2D(3,ipp-1,2)] - pos[LINIDX_2D(3,ip-1,2)] + cells[LINIDX_2D(5,cpp_i + ((cp-1)*14),3)]*d_extent[2];
                        }
                        else {
                            rv[0] = pos[LINIDX_2D(3,ipp-1,0)] - pos[LINIDX_2D(3,ip-1,0)];
                            rv[1] = pos[LINIDX_2D(3,ipp-1,1)] - pos[LINIDX_2D(3,ip-1,1)];
                            rv[2] = pos[LINIDX_2D(3,ipp-1,2)] - pos[LINIDX_2D(3,ip-1,2)];                              
                        }
                        
                        
                        
                        double r2 = pow(rv[0],2) + pow(rv[1],2) + pow(rv[2],2);
                        
                        
                        
                        if (r2 < rc2){

                            /* Lennard-Jones */
                            double C_F = -48.0;
                            double r_m2 = 1/r2;
                            double r_m6 = pow(r_m2,3);
                            double f_tmp = C_F*(pow(r_m2,7) - 0.5*pow(r_m2,4) );
                            U[0]+= 4.0*((r_m6-1.0)*r_m6 + 0.25);
                            
                            
                            accel[LINIDX_2D(3,ip-1,0)]+=f_tmp*rv[0];
                            accel[LINIDX_2D(3,ip-1,1)]+=f_tmp*rv[1];
                            accel[LINIDX_2D(3,ip-1,2)]+=f_tmp*rv[2];
                            
                            accel[LINIDX_2D(3,ipp-1,0)]-=f_tmp*rv[0];
                            accel[LINIDX_2D(3,ipp-1,1)]-=f_tmp*rv[1];
                            accel[LINIDX_2D(3,ipp-1,2)]-=f_tmp*rv[2];

                        }
                        //END OF KERNEL CODE
  
  
  
  
  
                    }
                    ipp = q_list[ipp];  
                }
                ip=q_list[ip];
            }
        }
    }
    
    
    
    
    return;
}
