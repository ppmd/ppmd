

extern "C"
int HaloCellLinkedList(
    const INT CC,
    const INT shift,
    const INT end,
    INT * RESTRICT q,
    const INT * RESTRICT LCI,
    const INT * RESTRICT CRC,
    INT * RESTRICT CCC,
    INT * RESTRICT CRL
){
    INT index = shift;
    for(INT ix = 0; ix < CC; ix++){

        //get number of particles
        const INT _tmp = CRC[ix];

        if (_tmp>0){

            //first index in cell region of cell list.
            q[end+LCI[ix]] = index;
            CCC[LCI[ix]] = _tmp;

            //start at first particle in halo cell, work forwards
            for(INT iy = 0; iy < _tmp-1; iy++){
                q[index+iy]=index+iy+1;
                CRL[index+iy]=LCI[ix];
            }
            q[index+_tmp-1] = -1;
            CRL[index+_tmp-1] = LCI[ix];
        }

        index += CRC[ix];
    }

    return 0;
}
