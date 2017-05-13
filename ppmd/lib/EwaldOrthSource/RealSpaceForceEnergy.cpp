






const double R0 = P.j[0] - P.i[0];
const double R1 = P.j[1] - P.i[1];
const double R2 = P.j[2] - P.i[2];

const double r2 = R0*R0 + R1*R1 + R2*R2;

const double r = sqrt(r2);

const double r_m1 = 0.5/r;
const double qiqj_rm1 = Q.i[0] * Q.j[0] * r_m1;
const double sqrtalpha_r = SQRT_ALPHA*r;

u[0] += (r < REAL_CUTOFF) ? qiqj_rm1*erfc(sqrtalpha_r) : 0.0;


//F.i[0]+= (r2 < rc2) ? f_tmp*R0 : 0.0;
//F.i[1]+= (r2 < rc2) ? f_tmp*R1 : 0.0;
//F.i[2]+= (r2 < rc2) ? f_tmp*R2 : 0.0;
