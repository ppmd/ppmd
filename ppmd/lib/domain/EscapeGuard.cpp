







extern "C"
int EscapeGuard(
    INT _end, //N
    INT * RESTRICT EC, //Escape count
    INT * RESTRICT BL, //bin to lin map
    INT * RESTRICT ELL, //escape linked list
    REAL * RESTRICT B, // boundary
    REAL * RESTRICT P // positions
){

    int ELL_index = 26;

    for(int _ix = 0; _ix < _end; _ix++){
        int b = 0;

        //Check x direction
        if (P[3*_ix] < B[0]){
            b ^= 4;
        }else if (P[3*_ix] >= B[1]){
            b ^= 32;
        }

        //check y direction
        if (P[3*_ix+1] < B[2]){
            b ^= 2;
        }else if (P[3*_ix+1] >= B[3]){
            b ^= 16;
        }

        //check z direction
        if (P[3*_ix+2] < B[4]){
            b ^= 1;
        }else if (P[3*_ix+2] >= B[5]){
            b ^= 8;
        }

        //If b > 0 then particle has escaped through some boundary
        if (b>0){

            EC[BL[b]]++;        //lookup which direction then increment that direction escape count.

            ELL[ELL_index] = _ix;            //Add current local id to linked list.
            ELL[ELL_index+1] = ELL[BL[b]];   //Set previous index to be next element.
            ELL[BL[b]] = ELL_index;          //Set current index in ELL to be the last index.

            ELL_index += 2;
        }

    }

    return 0;
}










