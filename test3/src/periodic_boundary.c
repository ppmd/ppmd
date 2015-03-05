#include <omp.h>
#include "periodic_boundary.h"

#include <math.h>
#include <stdio.h>


#define LINIDX_2D_3(iy,ix)   ((3)*(iy) + (ix))

void d_periodic_boundary(int N, double* extent, double *r_in){
int ix;
double x;

#pragma omp parallel for
for(ix=0;ix<N;ix++){    
            
            if (abs(r_in[LINIDX_2D_3(ix,0)]) > 0.5*extent[0]){
                x=r_in[LINIDX_2D_3(ix,0)]+0.5*extent[0];

                if (x < 0){
                    r_in[LINIDX_2D_3(ix,0)] = (extent[0] - fmod(abs(x) , extent[0])) - 0.5*extent[0];
                }else{
                    r_in[LINIDX_2D_3(ix,0)] = fmod( x , extent[0] ) - 0.5*extent[0];
                }
            
            }
            
              
            if (abs(r_in[LINIDX_2D_3(ix,1)]) > 0.5*extent[1]){
                x=r_in[LINIDX_2D_3(ix,1)]+0.5*extent[1];
       
                if (x < 0){
                    r_in[LINIDX_2D_3(ix,1)] = (extent[1] - fmod(abs(x) , extent[1])) - 0.5*extent[1];
                }else{
                    r_in[LINIDX_2D_3(ix,1)] = fmod( x , extent[1] ) - 0.5*extent[1];
                }
            }
            
            if (abs(r_in[LINIDX_2D_3(ix,2)]) > 0.5*extent[2]){
                x=r_in[LINIDX_2D_3(ix,2)]+0.5*extent[2];
       
                if (x < 0){
                    r_in[LINIDX_2D_3(ix,2)] = (extent[2] - fmod(abs(x) , extent[2])) - 0.5*extent[2];
                }else{
                    r_in[LINIDX_2D_3(ix,2)] = fmod( x , extent[2] ) - 0.5*extent[2];
                }            
            }


}










}


