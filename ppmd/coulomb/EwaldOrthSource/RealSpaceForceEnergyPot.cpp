const double R0 = P.j[0] - P.i[0];
const double R1 = P.j[1] - P.i[1];
const double R2 = P.j[2] - P.i[2];

const double r2 = R0*R0 + R1*R1 + R2*R2;
//const double rmrc = r2 - REAL_CUTOFF_SQ;
//const double mask = (   (*((uint64_t*)&(rmrc) ) & ((uint64_t)1 << 63)    )>> 63) & 1;
const double mask = (r2 < REAL_CUTOFF_SQ)? 1.0 : 0.0;

const double r = sqrt(r2);
const double r_m1 = 1.0/r;

const double qiqj_rm1 = Q.i[0] * Q.j[0] * r_m1 * mask;
const double sqrtalpha_r = SQRT_ALPHA*r;

// (qi*qj/rij) * erfc(sqrt(alpha)*rij)
const double term1 = qiqj_rm1*erfc(sqrtalpha_r);

u[0] += ENERGY_UNITO2*term1;
//term2 = term1*(1/rij) = (qi*qj/(rij**2)) * erfc(sqrt(alpha)*rij)
const double term2 = r_m1*term1;
//term3 = (qi*qj/rij)*(sqrt(alpha/pi) * -2)*(exp(-alpha*(rij**2))) - term2
const double term3 = r_m1*(qiqj_rm1 * M2_SQRT_ALPHAOPI * exp(MALPHA*r2) - term2);

F.i[0] += FORCE_UNIT*term3 * R0;
F.i[1] += FORCE_UNIT*term3 * R1;
F.i[2] += FORCE_UNIT*term3 * R2;
UPP.i[0] += ENERGY_UNIT * term1;

