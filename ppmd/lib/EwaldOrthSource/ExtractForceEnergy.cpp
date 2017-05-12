/*
args
~~~~
double * RecipSpace
const double * Positions
const double * Charges
*/


// kernel start --------------------------------------

const double* PlaneSpace = &RecipSpace[0] + 12*NKAXIS;
const double* RRecipSpace = PlaneSpace + PLANE_SIZE;
const double* IRecipSpace = RRecipSpace + 8*LEN_QUAD;

const double ri[4] = {Positions.i[0]*GX, Positions.i[1]*GY, Positions.i[2]*GZ, 0.0};
const double charge_i = Charges.i[0];
const double charge_i_o2 = charge_i*0.5;

double tmp_energy[1] = {0.0};


double re_exp[4];
double im_exp[4];

// could pad to 4 for avx call instead of an sse call
for(int ix=0 ; ix<4 ; ix++) { im_exp[ix] = sin(ri[ix]); re_exp[ix] = cos(ri[ix]); }


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


// start with the axes
// X
for( int ii=0 ; ii<NK ; ii++ ){
    const double coeff = COEFF_SPACE(ii+1, 0, 0) * charge_i_o2;
    tmp_energy[0] += coeff * ((RRAXIS(XP, ii) + RRAXIS(XN, ii))*TMP_RECIP_AXES[XQR][ii] + (IRAXIS(XN, ii) - IRAXIS(XP, ii))*TMP_RECIP_AXES[XQI][ii]);
} 
// Y
for( int ii=0 ; ii<NL ; ii++ ){
    const double coeff = COEFF_SPACE(0, ii+1, 0) * charge_i_o2;
    tmp_energy[0] += coeff * ((RRAXIS(YP, ii) + RRAXIS(YN, ii))*TMP_RECIP_AXES[YQR][ii] + (IRAXIS(YN, ii) - IRAXIS(YP, ii))*TMP_RECIP_AXES[YQI][ii]);
}
// Z
for( int ii=0 ; ii<NM ; ii++ ){
    const double coeff = COEFF_SPACE(0, 0, ii+1) * charge_i_o2;
    tmp_energy[0] += coeff * ((RRAXIS(ZP, ii) + RRAXIS(ZN, ii))*TMP_RECIP_AXES[ZQR][ii] + (IRAXIS(ZN, ii) - IRAXIS(ZP, ii))*TMP_RECIP_AXES[ZQI][ii]);
}


// now the planes between the axes
// double loop over x,y

for(int iy=0 ; iy<NL ; iy++){
    const double ap = TMP_RECIP_AXES[YQR][iy];
    const double bp = TMP_RECIP_AXES[YQI][iy];
    for(int ix=0 ; ix<NK ; ix++ ){
        const double xp = TMP_RECIP_AXES[XQR][ix];
        const double yp = TMP_RECIP_AXES[XQI][ix];
        const double coeff = COEFF_SPACE(ix+1, iy+1, 0) * charge_i_o2;
        for(int qx=0 ; qx<4 ; qx++){
            const double reali = xp*ap - CC_COEFF_PLANE_X1[qx]*yp * CC_COEFF_PLANE_X2[qx]*bp;
            const double imagi = xp * CC_COEFF_PLANE_X2[qx]*bp + CC_COEFF_PLANE_X1[qx]*yp*ap;
            tmp_energy[0] += coeff*(RRPLANE_0(qx, ix, iy)*reali - IRPLANE_0(qx, ix, iy)*imagi);
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
        const double coeff = COEFF_SPACE(0, ix+1, iy+1) * charge_i_o2;
        for(int qx=0 ; qx<4 ; qx++){
            const double reali = xp*ap - CC_COEFF_PLANE_X1[qx]*yp * CC_COEFF_PLANE_X2[qx]*bp;
            const double imagi = xp * CC_COEFF_PLANE_X2[qx]*bp + CC_COEFF_PLANE_X1[qx]*yp*ap;
            tmp_energy[0] += coeff*(RRPLANE_1(qx, ix, iy)*reali - IRPLANE_1(qx, ix, iy)*imagi);
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
        const double coeff = COEFF_SPACE(iy+1, 0, ix+1) * charge_i_o2;
        for(int qx=0 ; qx<4 ; qx++){
            const double reali = xp*ap - CC_COEFF_PLANE_X1[qx]*yp * CC_COEFF_PLANE_X2[qx]*bp;
            const double imagi = xp * CC_COEFF_PLANE_X2[qx]*bp + CC_COEFF_PLANE_X1[qx]*yp*ap;
            tmp_energy[0] += coeff*(RRPLANE_2(qx, ix, iy)*reali - IRPLANE_2(qx, ix, iy)*imagi);
        }
    }
}


// finally loop over axes and quadrants
//RRS_INDEX(k,l,m,q)
for(int iz=0 ; iz<NM ; iz++ ){
    const double gp = TMP_RECIP_AXES[ZQR][iz];
    const double hp = TMP_RECIP_AXES[ZQI][iz];
    const double izGZ = (iz+1)*GZ;
    const double recip_len_z = izGZ*izGZ;

    for(int iy=0 ; iy<NL ; iy++){
        const double ap = TMP_RECIP_AXES[YQR][iy];
        const double bp = TMP_RECIP_AXES[YQI][iy];
        const double iyGY = (iy+1)*GY;
        const double recip_len_zy = recip_len_z + iyGY*iyGY;
        
        for(int ix=0 ; ix<NK ; ix++ ){
            const double ixGX = (ix+1)*GX;
            const double recip_len_zyx = recip_len_zy + ixGX*ixGX;
            
            if (recip_len_zyx < MAX_RECIP_SQ){
                const double xp = TMP_RECIP_AXES[XQR][ix];
                const double yp = TMP_RECIP_AXES[XQI][ix];
                const double* r_base_index = &RRS_INDEX(ix,iy,iz,0);
                const double* i_base_index = &IRS_INDEX(ix,iy,iz,0);
                const double coeff = COEFF_SPACE(ix+1, iy+1, iz+1) * charge_i_o2;
                for(int qx=0 ; qx<8 ; qx++){
                    const double ycp = yp * CC_MAP_X(qx);
                    const double bcp = bp * CC_MAP_Y(qx);
                    const double hcp = hp * CC_MAP_Z(qx);
                    const double xa_m_yb = xp*ap - ycp*bcp;
                    const double xb_p_ya = xp*bcp + ycp*ap;

                    tmp_energy[0] += coeff*((gp*xa_m_yb - hcp*xb_p_ya)*(*(r_base_index+qx)) - (xa_m_yb*hcp + xb_p_ya*gp)*(*(i_base_index+qx)));
                }
            } else {
                break;
            }
        }
    }
}

Energy[0] += tmp_energy[0];

// kernel end -----------------------------------------
