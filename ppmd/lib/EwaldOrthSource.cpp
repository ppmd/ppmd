


/*
args
~~~~
double * RecipSpace
const double * Positions
*/



// kernel start --------------------------------------

const int IndexRealStart = 12*NKAXIS;
const int IndexImagStart = IndexRealStart + 8*LEN_QUAD;
double* RRecipSpace = RecipSpace + IndexRealStart;
double* IRecipSpace = RecipSpace + IndexImagStart;

const double ri[4] = {-1.0*Positions.i[0]*GX, -1.0*Positions.i[1]*GX, -1.0*Positions.i[2]*GX, 0.0};
double re_exp[4];
double im_exp[4];

// could pad to 4 for avx call instead of an sse call
for(int ix=0 ; ix<3 ; ix++) { im_exp[ix] = sin(ri[ix]); re_exp[ix] = cos(ri[ix]); }


// populate first and second entries in reciprocal axis
//RE
TMP_RECIP_AXES[XQR][0] = 1.0;
TMP_RECIP_AXES[YQR][0] = 1.0;
TMP_RECIP_AXES[ZQR][0] = 1.0;
//IM
TMP_RECIP_AXES[XQI][0] = 0.0;
TMP_RECIP_AXES[YQI][0] = 0.0;
TMP_RECIP_AXES[ZQI][0] = 0.0;

//RE
TMP_RECIP_AXES[XQR][1] = re_exp[0];
TMP_RECIP_AXES[YQR][2] = re_exp[1];
TMP_RECIP_AXES[ZQR][1] = re_exp[2];
//IM
TMP_RECIP_AXES[XQI][1]  = im_exp[0];
TMP_RECIP_AXES[YQI][1]  = im_exp[1];
TMP_RECIP_AXES[ZQI][1]  = im_exp[2];


// multiply out x dir
const double re_p1x = TMP_RECIP_AXES[XQR][1];
const double im_p1x = TMP_RECIP_AXES[XQI][1];
for(int ix=2 ; ix<NK ; ix++) {
    COMP_AB(&re_p1x, &im_p1x, &TMP_RECIP_AXES[XQR][ix-1], &TMP_RECIP_AXES[XQI][ix-1], &TMP_RECIP_AXES[XQR][ix], &TMP_RECIP_AXES[XQI][ix]);
}
// multiply out y dir
const double re_p1y = TMP_RECIP_AYES[YQR][1];
const double im_p1y = TMP_RECIP_AYES[YQI][1];
for(int ix=2 ; ix<NL ; ix++) {
    COMP_AB(&re_p1x, &im_p1x, &TMP_RECIP_AYES[YQR][ix-1], &TMP_RECIP_AYES[YQI][ix-1], &TMP_RECIP_AYES[YQR][ix], &TMP_RECIP_AYES[YQI][ix]);
}
// multiply out z dir
const double re_p1z = TMP_RECIP_AZES[ZQR][1];
const double im_p1z = TMP_RECIP_AZES[ZQI][1];
for(int ix=2 ; ix<NM ; ix++) {
    COMP_AB(&re_p1x, &im_p1x, &TMP_RECIP_AZES[ZQR][ix-1], &TMP_RECIP_AZES[ZQI][ix-1], &TMP_RECIP_AZES[ZQR][ix], &TMP_RECIP_AZES[ZQI][ix]);
}


// now can multiply out and add to recip space
// start with the axes

// X
for( int ii=1 ; ii<NK ; ii++ ){
    RRAXIS(0, ii) += TMP_RECIP_AXIS[XQR][ii];
    IRAXIS(0, ii) += TMP_RECIP_AXIS[XQI][ii];
    RRAXIS(2, ii) += TMP_RECIP_AXIS[XQR][ii];
    IRAXIS(2, ii) -= TMP_RECIP_AXIS[XQI][ii];
}
// Y
for( int ii=1 ; ii<NL ; ii++ ){
    RRAXIS(1, ii) += TMP_RECIP_AXIS[YQR][ii];
    IRAXIS(1, ii) += TMP_RECIP_AXIS[YQI][ii];
    RRAXIS(3, ii) += TMP_RECIP_AXIS[YQR][ii];
    IRAXIS(3, ii) -= TMP_RECIP_AXIS[YQI][ii];
}
// Z
for( int ii=1 ; ii<NM ; ii++ ){
    RRAXIS(4, ii) += TMP_RECIP_AXIS[ZQR][ii];
    IRAXIS(4, ii) += TMP_RECIP_AXIS[ZQI][ii];
    RRAXIS(5, ii) += TMP_RECIP_AXIS[ZQR][ii];
    IRAXIS(5, ii) -= TMP_RECIP_AXIS[ZQI][ii];
}






















// kernel end -----------------------------------------
