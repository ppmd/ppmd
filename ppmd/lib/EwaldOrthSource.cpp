
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

inline void COMP_CONG(
    const double *x,
    double *y
){
    // y = x*
    y[0] = x[0];
    y[1] = -1. * x[1];
}

// reciprocal axis macros
// real recip axis index
#define RERAI(indx, alen, adir) (adir*(2*alen + 1) + indx + alen)
// imag recip axis index
#define IMRAI(indx, alen, adir) ((adir+3)*(2*alen + 1) + indx + alen)
// complex exp in axis
#define RAEXP(x, ptr, indx, alen, adir) (ptr[RERAI(indx,alen,adir)]=cos(x);\
                                         ptr[IMRAI(indx,alen,adir)]=sin(x))


extern "C" int ReciprocalContributions(
    const double * RESTRICT positions,
    const double * RESTRICT charges,
    const int N_LOCAL,
    const int * RESTRICT nmax_vec, // len = 3
    const double * RESTRICT recip_vec, //      = np.zeros((3,3), dtype=ctypes.c_double)
    const int recip_axis_len,               // number of k points is 2*recip_axis_length + 1
    double * RESTRICT recip_axis,           // space to hold the values on the axis to multiple together
    double * RESTRICT recip_space
){
    int err = 0;
    double *ra = recip_axis;




    for (int lx=0 ; lx<N_LOCAL ; lx++){

        for(int dx=0 ; dx<3 ; dx++){
            // because domain is orthoganal

            double ri = -1. * positions[lx*3+dx]*recip_vec[dx*4];
            recip_axis[2*dx*recip_axis_len] = 1.0;
            recip_axis[2*dx*recip_axis_len] = 0.0;
            COMP_EXP_PACKED(&ri, &recip_axis[2*dx*(recip_axis_len+1)]);
            COMP_CONG(&recip_axis[2*dx*recip_axis_len - 2], &recip_axis[2*dx*(recip_axis_len+1)]);

            for(int ex=1 ; ex<nmax_vec[dx] ; ex++){
                
                COMP_AB_PACKED(&recip_axis[2*(dx*recip_axis_len)],
                               &recip_axis[2*(dx*recip_axis_len + ex - 1)],
                               &recip_axis[2*(dx*recip_axis_len + ex)]);

            }
        }










    }
  

    return err;
}




