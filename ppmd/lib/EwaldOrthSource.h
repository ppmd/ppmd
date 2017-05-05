#include <math.h>


// defines requiring subsitutions ----------------------



// recip vectors assuming orthonganal box
#define GX (%(SUB_GX)s)
#define GY (%(SUB_GY)s)
#define GZ (%(SUB_GZ)s)

// max of all recip axis
#define NKMAX (%(SUB_NKMAX)s)
// len of each axis
#define NK (%(SUB_NK)s)
#define NL (%(SUB_NL)s)
#define NM (%(SUB_NM)s)
// Nkaxis = NKMAX+1
#define NKAXIS (%(SUB_NKAXIS)s)
#define LEN_QUAD (%(SUB_LEN_QUAD)s)



// other defines  -------------------------------------
//space allocated for planes in reciprocal space
#define PLANE_SIZE (8*(NK*NL)+8*(NL*NM)+8*(NM*NK))

// maps from quadrant indexes to axis indexes
#define XQR (0)
#define XQI (3)
#define YQR (1)
#define YQI (4)
#define ZQR (2)
#define ZQI (5)

// handle complex conjugate efficently
const double CC_COEFF[2] = {1.0, -1.0};
#define CC_MAP_X(qx) ( CC_COEFF[(((qx)+1) >> 1) & 1])
#define CC_MAP_Y(qx) ( CC_COEFF[((qx) >> 1) & 1]    )
#define CC_MAP_Z(qx) ( CC_COEFF[((qx) >> 2)]    )


// slower than above
/*
const double CC_COEFF_X[8] = {1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0};
const double CC_COEFF_Y[8] = {1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0};
const double CC_COEFF_Z[8] = {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0};
#define CC_MAP_X(qx) ( CC_COEFF_X[(qx)] )
#define CC_MAP_Y(qx) ( CC_COEFF_Y[(qx)] )
#define CC_MAP_Z(qx) ( CC_COEFF_Z[(qx)] )
*/





 
// double* RRecipSpace
// double* IRecipSpace 
// maps from k,l,m,quadrant into real and imaginary recip space assuming above pointers
#define RRS_INDEX(k,l,m,q) (RRecipSpace[  8*( (k) + NK*((l) + NL*(m)) )  + (q)])
#define IRS_INDEX(k,l,m,q) (IRecipSpace[  8*( (k) + NK*((l) + NL*(m)) )  + (q)])

// double * RecipSpace
// maps onto reciprocal space axis assuming above pointer
#define RRAXIS(ax, ex) (RecipSpace[(ax)*NKAXIS+(ex)])
#define IRAXIS(ax, ex) (RecipSpace[(6 + (ax))*NKAXIS+(ex)])


//double* PlaneSpace
//maps onto plane space assuming above pointer

#define RRPLANE_0(quad, x1 , x2) ( PlaneSpace[ (x2)*(NK*4) + (x1)*4 + (quad) ] )
#define IRPLANE_0(quad, x1 , x2) ( PlaneSpace[ (4*NK*NL) + (x2)*(NK*4) + (x1)*4 + (quad) ] )

#define RRPLANE_1(quad, x1 , x2) ( PlaneSpace[ (NK*NL*8) + (x2)*(NL*4) + (x1)*4 + (quad) ] )
#define IRPLANE_1(quad, x1 , x2) ( PlaneSpace[ (NK*NL*8) + (4*NL*NM) + (x2)*(NL*4) + (x1)*4 + (quad) ] )

#define RRPLANE_2(quad, x1 , x2) ( PlaneSpace[ (NK*NL*8) + (8*NL*NM) + (x2)*(NM*4) + (x1)*4 + (quad) ] )
#define IRPLANE_2(quad, x1 , x2) ( PlaneSpace[ (NK*NL*8) + (8*NL*NM) + (4*NM*NK) + (x2)*(NM*4) + (x1)*4 + (quad) ] )

// complex part coefficient for planes
const double CC_COEFF_PLANE_X1[4] = {1.0, -1.0, -1.0, 1.0};
const double CC_COEFF_PLANE_X2[4] = {1.0, 1.0, -1.0, -1.0};



// temporary space on the stack for the recip axis should be alright in terms of stack size....
double TMP_RECIP_AXES[6][NKMAX];


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





