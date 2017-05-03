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
 
// double* RRecipSpace
// double* IRecipSpace 
// maps from k,l,m,quadrant into real and imaginary recip space assuming above pointers
# define RRS_INDEX(k,l,m,q) (RRecipSpace[  8*( (k) + NK*((l) + NL*(m)) )  + (q)])
# define IRS_INDEX(k,l,m,q) (IRecipSpace[  8*( (k) + NK*((l) + NL*(m)) )  + (q)])

// double * RecipSpace
// maps onto reciprocal space axis assuming above pointer
#define RRAXIS(ax, ex) (RexipSpace[(ax)*NKAXIS+(ex)])
#define IRAXIS(ax, ex) (RexipSpace[(6 + (ax))*NKAXIS+(ex)])




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





