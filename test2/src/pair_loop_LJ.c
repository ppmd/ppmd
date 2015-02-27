#include "pair_loop_LJ.h"
#include <math.h>
#include <stdio.h>

#define LINIDX_2D(NX,iy,ix)   ((NX)*(iy) + (ix))


void d_pair_loop_LJ(int N, int cp, double rc, int* cells, int* q_list, double* pos, double* d_extent, double *accel, double* U){
    /*
    python involves
    
    int N
    int cp
    
    double rc
    
    int* [14,5] as pointer with cells and boundaries
    int* [1,N+c] linked list of cell contents
    
    double* [N,3] positions
    double* [1,3] domain_extent
    double* accelerations
    double* U (potential energy)

    */
    
    int cpp,ip,ipp,cpp_i;
    double rv[3], r2;
    double rc2 = rc*rc;
    
    int count = 0,j;

    // LJ variables
    
    double C_F = 48.0;
    double r_m2,r_m6;
    double f_tmp;
    
    
    for(cpp_i=0;cpp_i<14;cpp_i++){
    
        cpp = cells[LINIDX_2D(5,cpp_i,0)];
        
        ip = q_list[N+cp];
        
        while (ip > 0){
            
            ipp = q_list[N+cpp];
            
            while (ipp > 0){
                
                if (cp != cpp || ip < ipp){
                    
                    if (cells[LINIDX_2D(5,cpp_i,4)] > 0){
                        rv[0] = pos[LINIDX_2D(3,ip-1,0)] - pos[LINIDX_2D(3,ipp-1,0)] + cells[LINIDX_2D(5,cpp_i,1)]*d_extent[0];
                        rv[1] = pos[LINIDX_2D(3,ip-1,1)] - pos[LINIDX_2D(3,ipp-1,1)] + cells[LINIDX_2D(5,cpp_i,2)]*d_extent[1];
                        rv[2] = pos[LINIDX_2D(3,ip-1,2)] - pos[LINIDX_2D(3,ipp-1,2)] + cells[LINIDX_2D(5,cpp_i,3)]*d_extent[2];
                    }
                    else {
                        rv[0] = pos[LINIDX_2D(3,ip-1,0)] - pos[LINIDX_2D(3,ipp-1,0)];
                        rv[1] = pos[LINIDX_2D(3,ip-1,1)] - pos[LINIDX_2D(3,ipp-1,1)];
                        rv[2] = pos[LINIDX_2D(3,ip-1,2)] - pos[LINIDX_2D(3,ipp-1,2)];                              
                    }
                    r2 = pow(rv[0],2) + pow(rv[1],2) + pow(rv[2],2);
                    
                    
                    if (r2 < rc2){
                        count++;
                        
                        
                        r_m2 = 1/r2;
                        f_tmp = C_F*(pow(r_m2,7) - 0.5*pow(r_m2,4) );
                        
                        accel[LINIDX_2D(3,ip-1,0)]+=f_tmp*rv[0];
                        accel[LINIDX_2D(3,ip-1,1)]+=f_tmp*rv[1];
                        accel[LINIDX_2D(3,ip-1,2)]+=f_tmp*rv[2];
                        
                        accel[LINIDX_2D(3,ipp-1,0)]-=f_tmp*rv[0];
                        accel[LINIDX_2D(3,ipp-1,1)]-=f_tmp*rv[1];
                        accel[LINIDX_2D(3,ipp-1,2)]-=f_tmp*rv[2];
                        
                        
                        
                        
                        r_m6 = pow(r_m2,3);
                        
                        U[0]+= 4.0*((r_m6-1.0)*r_m6 + 0.25);
                        
                    }
                }
                ipp = q_list[ipp];  
            }
            ip=q_list[ip];
        }
    }
    //printf("count: %d \n", count);
    
    
    
    
    
    
    
    return;
}
