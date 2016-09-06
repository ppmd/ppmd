
    int _F = 0;

    if (abs_md(P(0)) > 0.5*E0){
        const double E0_2 = 0.5*E0;
        const double x = P(0) + E0_2;


        if (x < 0){
            P(0) = (E0 - fmod(abs_md(x) , E0)) - E0_2;
            _F = 1;
        }
        else{
            P(0) = fmod( x , E0 ) - E0_2;
            _F = 1;
        }
    }

    if (abs_md(P(1)) > 0.5*E1){
        const double E1_2 = 0.5*E1;
        const double x = P(1) + E1_2;

        if (x < 0){
            P(1) = (E1 - fmod(abs_md(x) , E1)) - E1_2;
            _F = 1;
        }
        else{
            P(1) = fmod( x , E1 ) - E1_2;
            _F = 1;
        }
    }

    if (abs_md(P(2)) > 0.5*E2){
        const double E2_2 = 0.5*E2;
        const double x = P(2) + E2_2;

        if (x < 0){
            P(2) = (E2 - fmod(abs_md(x) , E2)) - E2_2;
            _F = 1;
        }
        else{
            P(2) = fmod( x , E2 ) - E2_2;
            _F = 1;
        }
    }

    if (_F > 0){
        //BCFLAG(0) = BCFLAG(0) || _F;
        BCFLAG(0) = _F;
    }


