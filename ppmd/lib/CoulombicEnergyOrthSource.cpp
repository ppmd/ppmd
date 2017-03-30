
/*
#define REAB(a,b) ((a)*(a) - (b)*(b))
#define IMAB(a,b) ((a)*(b)*2)
#define ABS2(a,b) ((a)*(a) + (b)*(b))
#define LINIDX(ix, iy, nx) ((iy)*(nx) + (ix))
*/


inline void COMP_AB(
    const double *a,
    const double *b,
    const double *x,
    const double *y,
    double *g,
    double *h
){
    // Compute (a + bi) * (x + yi)
    *g = (*a)*(*x) - (*b)*(*y);
    *h = (*a)*(*y) + (*b)*(*x);
}



inline void COMP_AB_PACKED(
    const double *a,
    const double *x,
    double *gh
){
    // Compute (a + bi) * (x + yi)
    gh[0] = a[0]*x[0] - a[1]*x[1];
    gh[1] = a[0]*x[1] + a[1]*x[0];
}



inline void COMP_EXP_PACKED(
    const double *x,
    double *gh
){
    gh[0] = cos(*x);
    gh[1] = sin(*x);
}



extern "C" int CoulombicEnergyOrth(
    const double * RESTRICT positions,
    const double * RESTRICT charges,
    const int N_LOCAL,
    const double alpha,
    const double max_recip,
    const int * RESTRICT nmax_vec, // len = 3
    const double * RESTRICT recip_vec, //      = np.zeros((3,3), dtype=ctypes.c_double)
    const double * RESTRICT recip_consts, //   = np.zeros(3, dtype=ctypes.c_double)
    const int recip_axis_len,               // length alloced for each dimension
    double * RESTRICT recip_axis           // space to hold the values on the axis to multiple together
){
    int err = 0;


    for (int lx=0 ; lx<N_LOCAL ; lx++){

        for(int dx=0 ; dx<3 ; dx++){
            // because domain is orthoganal
            const double gi = recip_vec[dx*4];
            
            double ri = positions[lx*3+dx]*gi;

            COMP_EXP_PACKED(&ri, &recip_axis[2*dx*recip_axis_len]);

            for(int ex=1 ; ex<nmax_vec[dx] ; ex++){
                
                COMP_AB_PACKED(&recip_axis[2*(dx*recip_axis_len)],
                               &recip_axis[2*(dx*recip_axis_len + ex - 1)],
                               &recip_axis[2*(dx*recip_axis_len + ex)]);

            }
        }










    }
  

    return err;
}




