const double ipx = P.i[0] + {hex};
const double ipy = P.i[1] + {hey};
const double ipz = P.i[2] + {hez};

const int nsx = {nsx};
const int nsy = {nsy};
const int nsz = {nsz};

const double jpx = P.j[0] + {hex};
const double jpy = P.j[1] + {hey};
const double jpz = P.j[2] + {hez};

const int icx = FMM_CELL.i[0] % nsx;
const int icy = ((FMM_CELL.i[0] - icx) / nsx) % nsy;
const int icz = (FMM_CELL.i[0] - icx - icy*nsx) / (nsx*nsy);

int jcx = FMM_CELL.j[0] % nsx;
int jcy = ((FMM_CELL.j[0] - jcx) / nsx) % nsy;
int jcz = (FMM_CELL.j[0] - jcx - jcy*nsx) / (nsx*nsy);

if (P.j[0] >= {hex}) {{ jcx += nsx; }}
if (P.j[1] >= {hey}) {{ jcy += nsy; }}
if (P.j[2] >= {hez}) {{ jcz += nsz; }}

if (P.j[0] <= -1.0*{hex}) {{ jcx -= nsx; }}
if (P.j[1] <= -1.0*{hey}) {{ jcy -= nsy; }}
if (P.j[2] <= -1.0*{hez}) {{ jcz -= nsz; }}

int dx = icx - jcx;
int dy = icy - jcy;
int dz = icz - jcz;

int dr2 = dx*dx + dy*dy + dz*dz;

{FREE_SPACE}

const double mask = ((dr2 > 3) ? 0.0 : 1.0) * \
    maskx*masky*maskz;

const double rx = P.j[0] - P.i[0];
const double ry = P.j[1] - P.i[1];
const double rz = P.j[2] - P.i[2];

#define ABS(x) (((x)>0)? (x): -1*(x))

//if(ABS(Q.i[0]*Q.j[0])>0.00001){{
//    printf("\t\t i | %f %f %f \n", rx, ry, rz);
//}}

const double r2 = rx*rx + ry*ry + rz*rz;
const double r = sqrt(r2);

const double ir = 1./r;
const double term1 = mask * Q.i[0] * Q.j[0] * ir;
PHI[0] += 0.5 * {ENERGY_UNIT} * term1;

const double fcoeff = {FORCE_UNIT}*ir*ir*term1;

F.i[0] -= fcoeff * rx;
F.i[1] -= fcoeff * ry;
F.i[2] -= fcoeff * rz;
