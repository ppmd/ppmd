if (M.i[0] > 0){
    if (M.j[0] < 1){
    const double R0 = P.j[0] - P.i[0];
    const double R1 = P.j[1] - P.i[1];
    const double R2 = P.j[2] - P.i[2];

    const double r2 = R0*R0 + R1*R1 + R2*R2;
    //const double rmrc = r2 - REAL_CUTOFF_SQ;
    //const double mask = (   (*((uint64_t*)&(rmrc) ) & ((uint64_t)1 << 63)    )>> 63) & 1;




    const double mask = (r2 < REAL_CUTOFF_SQ)? 2.0 : 0.0;

    const double r = sqrt(r2);
    const double r_m1 = 1.0/r;

    const double qiqj_rm1 = Q.j[0] * r_m1 * mask;
    const double sqrtalpha_r = SQRT_ALPHA*r;

    // (qi*qj/rij) * erfc(sqrt(alpha)*rij)
    const double term1 = qiqj_rm1*erfc(sqrtalpha_r);

    u.i[0] += ENERGY_UNITO2*term1;


    //printf("i %f\t%f\t%f\t|\t%f\t%f\t%f|\t%f\n", P.i[0], P.i[1],P.i[2],P.j[0],P.j[1],P.j[2],ENERGY_UNITO2*term1);

    }
}


