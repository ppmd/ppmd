
extern "C"
int CellLinkedList(
    const INT end_ix,
    const INT n,
    const REAL * RESTRICT B,  // Inner boundary on local domain (inc halo cells)
    const REAL * RESTRICT P,  // positions
    const REAL * RESTRICT CEL,// cell edge lengths
    const INT * RESTRICT CA,  // local domain cell array
    INT * RESTRICT q,         // cell list
    INT * RESTRICT CCC,       // contents count for each cell
    INT * RESTRICT CRL        // reverse cell lookup map
){

    const REAL _icel0 = 1.0/CEL[0];
    const REAL _icel1 = 1.0/CEL[1];
    const REAL _icel2 = 1.0/CEL[2];

    const REAL _b0 = B[0];
    const REAL _b2 = B[2];
    const REAL _b4 = B[4];

    for (INT ix=0; ix<end_ix; ix++) {

        INT C0 = 1 + (INT)((P[ix*3]     - _b0)*_icel0);
        INT C1 = 1 + (INT)((P[ix*3 + 1] - _b2)*_icel1);
        INT C2 = 1 + (INT)((P[ix*3 + 2] - _b4)*_icel2);

        if ((C0 < 1) || (C0 > (CA[0]-2))) {
            if ( (C0 > (CA[0]-2)) && (P[ix*3] <= B[1]  )) {
                C0 = CA[0]-2;

            } else {
                cout << "!! PARTICLE OUTSIDE DOMAIN IN CELL LIST REBUILD !! " << ix << " C0 " << C0 << endl;
                cout << "B[0] " << B[0] << " B[1] " << B[1] << " Px " << P[ix*3+0] << " " << (P[ix*3]-_b0)*_icel0 << endl;
                return -1;
            }
        }
        if ((C1 < 1) || (C1 > (CA[1]-2))) {
            if ( (C1 > (CA[1]-2)) && (P[ix*3+1] <= B[3]  )) {
                C1 = CA[1]-2;
            } else {
                cout << "!! PARTICLE OUTSIDE DOMAIN IN CELL LIST REBUILD !! " << ix << " C1 " << C1 << endl;
                cout << "B[2] " << B[2] << " B[3] " << B[3] << " Py " << P[ix*3+1]<< " " << (P[ix*3+1]-_b2)*_icel1 << endl;
                return -1;
            }
        }
        if ((C2 < 1) || (C2 > (CA[2]-2))) {
            if ( (C2 > (CA[2]-2)) && (P[ix*3+2] <= B[5]  )) {
                C2 = CA[2]-2;
            } else {
                cout << "!! PARTICLE OUTSIDE DOMAIN IN CELL LIST REBUILD !! " << ix << " C2 " << C2 << endl;
                cout << "B[4] " << B[4] << " B[5] " << B[5] << " Pz " << P[ix*3+2]<< " " << (P[ix*3 + 2]-_b4)*_icel2 << endl;
                return -1;
            }
        }

        const INT val = (C2*CA[1] + C1)*CA[0] + C0;
        CCC[val]++;
        CRL[ix] = val;

        q[ix] = q[n + val];
        q[n + val] = ix;

    }

    return 0;
}

