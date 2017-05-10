/*
args
~~~~
double * RecipSpace
const double * Positions
const double * Charges
*/



// kernel start --------------------------------------

double* PlaneSpace = &RecipSpace[0] + 12*NKAXIS;
double* RRecipSpace = PlaneSpace + PLANE_SIZE;
double* IRecipSpace = RRecipSpace + 8*LEN_QUAD;

const double ri[4] = {-1.0*Positions.i[0]*GX, -1.0*Positions.i[1]*GX, -1.0*Positions.i[2]*GX, 0.0};
const double charge_i = Charges.i[0];

double re_exp[4];
double im_exp[4];

// could pad to 4 for avx call instead of an sse call
for(int ix=0 ; ix<4 ; ix++) { im_exp[ix] = sin(ri[ix]); re_exp[ix] = cos(ri[ix]); }


// populate first entries in reciprocal axis
//RE technically these should be 1.0 but it makes our life easier
// if it is 0.0

//RE
TMP_RECIP_AXES[XQR][0] = re_exp[0];
TMP_RECIP_AXES[YQR][0] = re_exp[1];
TMP_RECIP_AXES[ZQR][0] = re_exp[2];
//IM
TMP_RECIP_AXES[XQI][0]  = im_exp[0];
TMP_RECIP_AXES[YQI][0]  = im_exp[1];
TMP_RECIP_AXES[ZQI][0]  = im_exp[2];


// multiply out x dir
const double re_p1x = TMP_RECIP_AXES[XQR][0];
const double im_p1x = TMP_RECIP_AXES[XQI][0];
for(int ix=1 ; ix<NK ; ix++) {
    COMP_AB(&re_p1x, &im_p1x, &TMP_RECIP_AXES[XQR][ix-1], &TMP_RECIP_AXES[XQI][ix-1], &TMP_RECIP_AXES[XQR][ix], &TMP_RECIP_AXES[XQI][ix]);
}
// multiply out y dir
const double re_p1y = TMP_RECIP_AXES[YQR][0];
const double im_p1y = TMP_RECIP_AXES[YQI][0];
for(int ix=1 ; ix<NL ; ix++) {
    COMP_AB(&re_p1y, &im_p1y, &TMP_RECIP_AXES[YQR][ix-1], &TMP_RECIP_AXES[YQI][ix-1], &TMP_RECIP_AXES[YQR][ix], &TMP_RECIP_AXES[YQI][ix]);
}
// multiply out z dir
const double re_p1z = TMP_RECIP_AXES[ZQR][0];
const double im_p1z = TMP_RECIP_AXES[ZQI][0];
for(int ix=1 ; ix<NM ; ix++) {
    COMP_AB(&re_p1z, &im_p1z, &TMP_RECIP_AXES[ZQR][ix-1], &TMP_RECIP_AXES[ZQI][ix-1], &TMP_RECIP_AXES[ZQR][ix], &TMP_RECIP_AXES[ZQI][ix]);
}


// now can multiply out and add to recip space
// start with the axes

// X
for( int ii=0 ; ii<NK ; ii++ ){
    RRAXIS(0, ii) += charge_i*TMP_RECIP_AXES[XQR][ii];
    IRAXIS(0, ii) += charge_i*TMP_RECIP_AXES[XQI][ii];
    RRAXIS(2, ii) += charge_i*TMP_RECIP_AXES[XQR][ii];
    IRAXIS(2, ii) -= charge_i*TMP_RECIP_AXES[XQI][ii];
}


// Y
for( int ii=0 ; ii<NL ; ii++ ){
    RRAXIS(1, ii) += charge_i*TMP_RECIP_AXES[YQR][ii];
    IRAXIS(1, ii) += charge_i*TMP_RECIP_AXES[YQI][ii];
    RRAXIS(3, ii) += charge_i*TMP_RECIP_AXES[YQR][ii];
    IRAXIS(3, ii) -= charge_i*TMP_RECIP_AXES[YQI][ii];
}
// Z
for( int ii=0 ; ii<NM ; ii++ ){
    RRAXIS(4, ii) += charge_i*TMP_RECIP_AXES[ZQR][ii];
    IRAXIS(4, ii) += charge_i*TMP_RECIP_AXES[ZQI][ii];
    RRAXIS(5, ii) += charge_i*TMP_RECIP_AXES[ZQR][ii];
    IRAXIS(5, ii) -= charge_i*TMP_RECIP_AXES[ZQI][ii];
}


// now the planes between the axes
// double loop over x,y

for(int iy=0 ; iy<NL ; iy++){
    const double ap = TMP_RECIP_AXES[YQR][iy];
    const double bp = TMP_RECIP_AXES[YQI][iy];
    for(int ix=0 ; ix<NK ; ix++ ){
        const double xp = TMP_RECIP_AXES[XQR][ix];
        const double yp = TMP_RECIP_AXES[XQI][ix];
        for(int qx=0 ; qx<4 ; qx++){
            RRPLANE_0(qx, ix, iy) += charge_i*(xp*ap - CC_COEFF_PLANE_X1[qx]*yp * CC_COEFF_PLANE_X2[qx]*bp);
            IRPLANE_0(qx, ix, iy) += charge_i*(xp * CC_COEFF_PLANE_X2[qx]*bp + CC_COEFF_PLANE_X1[qx]*yp*ap);
        }
    }
}


// double loop over y,z
for(int iy=0 ; iy<NM ; iy++){
    const double ap = TMP_RECIP_AXES[ZQR][iy];
    const double bp = TMP_RECIP_AXES[ZQI][iy];
    for(int ix=0 ; ix<NL ; ix++ ){
        const double xp = TMP_RECIP_AXES[YQR][ix];
        const double yp = TMP_RECIP_AXES[YQI][ix];
        for(int qx=0 ; qx<4 ; qx++){
            RRPLANE_1(qx, ix, iy) += charge_i*(xp*ap - CC_COEFF_PLANE_X1[qx]*yp * CC_COEFF_PLANE_X2[qx]*bp);
            IRPLANE_1(qx, ix, iy) += charge_i*(xp * CC_COEFF_PLANE_X2[qx]*bp + CC_COEFF_PLANE_X1[qx]*yp*ap);
        }
    }
}


// double loop over z,x
for(int iy=0 ; iy<NK ; iy++){
    const double ap = TMP_RECIP_AXES[XQR][iy];
    const double bp = TMP_RECIP_AXES[XQI][iy];
    for(int ix=0 ; ix<NM ; ix++ ){
        const double xp = TMP_RECIP_AXES[ZQR][ix];
        const double yp = TMP_RECIP_AXES[ZQI][ix];
        for(int qx=0 ; qx<4 ; qx++){
            RRPLANE_2(qx, ix, iy) += charge_i*(xp*ap - CC_COEFF_PLANE_X1[qx]*yp * CC_COEFF_PLANE_X2[qx]*bp);
            IRPLANE_2(qx, ix, iy) += charge_i*(xp * CC_COEFF_PLANE_X2[qx]*bp + CC_COEFF_PLANE_X1[qx]*yp*ap);
        }
    }
}


// finally loop over axes and quadrants
//RRS_INDEX(k,l,m,q)
for(int iz=0 ; iz<NM ; iz++ ){
    const double gp = TMP_RECIP_AXES[ZQR][iz];
    const double hp = TMP_RECIP_AXES[ZQI][iz];
    for(int iy=0 ; iy<NL ; iy++){
        const double ap = TMP_RECIP_AXES[YQR][iy];
        const double bp = TMP_RECIP_AXES[YQI][iy];
        for(int ix=0 ; ix<NK ; ix++ ){
            const double xp = TMP_RECIP_AXES[XQR][ix];
            const double yp = TMP_RECIP_AXES[XQI][ix];
            double* r_base_index = &RRS_INDEX(ix,iy,iz,0);
            double* i_base_index = &IRS_INDEX(ix,iy,iz,0);
            for(int qx=0 ; qx<8 ; qx++){
                const double ycp = yp * CC_MAP_X(qx);
                const double bcp = bp * CC_MAP_Y(qx);
                const double hcp = hp * CC_MAP_Z(qx);
                const double xa_m_yb = xp*ap - ycp*bcp;
                const double xb_p_ya = xp*bcp + ycp*ap;
                *(r_base_index+qx) += charge_i * (gp*xa_m_yb - hcp*xb_p_ya);
                *(i_base_index+qx) += charge_i * (xa_m_yb*hcp + xb_p_ya*gp);
            }
        }
    }
}

// kernel end -----------------------------------------
