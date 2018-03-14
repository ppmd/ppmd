


static inline REAL get_mat_element(
	const INT64 j,
	const INT64 mp,
	const INT64 m,
	const REAL * RESTRICT mat
){	


	if ((m > j) || (m<-1*j) || (mp > j) || (mp<-1*j)) { return 0.0; }
	
	const INT64 p = 2*j+1;
	const INT64 mx = j+m;
	const INT64 mpx = j+mp;
	
	return mat[mpx*p + mx];	
}


static inline REAL rec0(
	const INT64 j,
	const INT64 m,
	const INT64 mp,
	const REAL cb,
	const REAL sb,
	const REAL * RESTRICT jm1_mat
){
    const INT64 denom = ((j+mp)*(j-mp));

	const REAL sc = sb * cb;
	const REAL idenom = 1.0/denom;

	const REAL term1 = sc * sqrt((j+m)*(j+m-1)*idenom) * \
		get_mat_element(j-1, mp, m-1, jm1_mat);

	const REAL term2 = (cb*cb - sb*sb)*sqrt(
		((j-m)*(j+m))*idenom) * \
		get_mat_element(j-1, mp, m, jm1_mat);


	const REAL term3 = sc * sqrt(
		(j-m)*(j-m-1)*idenom) * \
		get_mat_element(j-1, mp, m+1, jm1_mat);

	return term1 + term2 - term3;
}



static inline REAL rec1(
	const INT64 j,
	const INT64 m,
	const INT64 mp,
	const REAL cb,
	const REAL sb,
	const REAL * RESTRICT jm1_mat
){
    const INT64 denom = ((j + mp)*(j + mp -1));

	const REAL idenom = 1.0/denom;

	const REAL term1 = cb * cb * sqrt(
		((j + m)*(j + m - 1))*idenom) * \
		get_mat_element(j-1, mp-1, m-1, jm1_mat);

	const REAL term2 = 2. * sb * cb * sqrt(
		((j+m)*(j-m)) * idenom) * \
		get_mat_element(j-1, mp-1, m, jm1_mat);

	const REAL term3 = sb * sb * sqrt(
		((j-m)*(j-m-1.))*idenom) * \
		get_mat_element(j-1, mp-1, m+1, jm1_mat);

	return term1 - term2 + term3;
}


static inline REAL rec2(
	const INT64 j,
	const INT64 m,
	const INT64 mp,
	const REAL cb,
	const REAL sb,
	const REAL * RESTRICT jm1_mat
){
    const INT64 denom = ((j-mp)*(j-mp-1));

	const REAL idenom = 1.0/denom;

	const REAL term1 = sb * sb * sqrt(
		((j+m)*(j+m-1))*idenom) * \
		get_mat_element(j-1, mp+1, m-1, jm1_mat);

	const REAL term2 = 2. * sb * cb * sqrt(
		((j+m)*(j-m))*idenom) * \
		get_mat_element(j-1, mp+1, m, jm1_mat);

	const REAL term3 = cb * cb * sqrt(
		((j-m)*(j-m-1)) * idenom) * \
		get_mat_element(j-1, mp+1, m+1, jm1_mat);

	return term1 + term2 + term3;
}


static inline REAL rec(
	const INT64 j,
	const INT64 m,
	const INT64 mp,
	const REAL cb,
	const REAL sb,
	const REAL * RESTRICT jm1_mat
){
	if (((j+mp)*(j - mp)) != 0){
		//return 0.0;
		return rec0(j, m, mp, cb, sb, jm1_mat);
	} else if (((j + mp)*(j + mp - 1)) != 0) {
		return rec1(j, m, mp, cb, sb, jm1_mat);
	} else {
		return rec2(j, m, mp, cb, sb, jm1_mat);
	}
}


extern "C"
int get_matrix_set(
    const INT64 maxj,
	const REAL beta,
    REAL * RESTRICT * RESTRICT mat_re
)
{
    if (maxj<1) { return 0;}
	const REAL cb = cos(0.5*beta);
	const REAL sb = sin(0.5*beta);
	
    const INT64 nthreads = omp_get_num_threads();
    const REAL inthreads = 1.0/nthreads;
	// zeroth entry is always 1
	mat_re[0][0] = 1.0;
	for (INT64 jx=1 ; jx<maxj ; jx++){
		const INT64 p = 2*jx+1;
        if (p > nthreads){
            const INT64 wp = p*inthreads;
#pragma omp parallel for default(none) shared(mat_re, jx) schedule(static, wp)
            for(INT64 mpx=-1*jx ; mpx<=jx ; mpx++){
                for(INT64 mx=-1*jx ; mx<=jx ; mx++){
                    mat_re[jx][(mpx+jx)*p + (mx+jx)] = rec(jx, mx, mpx, cb, sb, mat_re[jx-1]);
                }
            }
        }else {
            for(INT64 mpx=-1*jx ; mpx<=jx ; mpx++){
                for(INT64 mx=-1*jx ; mx<=jx ; mx++){
                    mat_re[jx][(mpx+jx)*p + (mx+jx)] = rec(jx, mx, mpx, cb, sb, mat_re[jx-1]);
                }
            }
        }

	}

    return 0;
}




