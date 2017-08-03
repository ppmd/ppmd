


extern "C"
int OneRankPBC(
    INT _end,
    REAL *P,
    REAL *E,
    REAL *F
){

    INT _F = 0;

    for(INT _ix = 0; _ix < _end; _ix++){

        if (abs_md(P[3*_ix]) >= 0.5*E[0]){
            const REAL E0_2 = 0.5*E[0];
            const REAL x = P[3*_ix] + E0_2;

            if (x < 0){
                P[3*_ix] = (E[0] - fmod(abs_md(x) , E[0])) - E0_2;
                _F = 1;
            }
            else{
                P[3*_ix] = fmod( x , E[0] ) - E0_2;
                _F = 1;
            }
        }

        if (abs_md(P[3*_ix+1]) >= 0.5*E[1]){
            const REAL E1_2 = 0.5*E[1];
            const REAL x = P[3*_ix+1] + E1_2;

            if (x < 0){
                P[3*_ix+1] = (E[1] - fmod(abs_md(x) , E[1])) - E1_2;
                _F = 1;
            }
            else{
                P[3*_ix+1] = fmod( x , E[1] ) - E1_2;
                _F = 1;
            }
        }

        if (abs_md(P[3*_ix+2]) >= 0.5*E[2]){
            const REAL E2_2 = 0.5*E[2];
            const REAL x = P[3*_ix+2] + E2_2;

            if (x < 0){
                P[3*_ix+2] = (E[2] - fmod(abs_md(x) , E[2])) - E2_2;
                _F = 1;
            }
            else{
                P[3*_ix+2] = fmod( x , E[2] ) - E2_2;
                _F = 1;
            }
        }

    }

    F[0] = _F;

    return 0;
}


