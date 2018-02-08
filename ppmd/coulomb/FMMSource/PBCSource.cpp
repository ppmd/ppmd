


static inline SPH cart_to_sph(const CART r){
    SPH rp;
    const REAL x2y2 = r.x*r.x + r.y*r.y;
    rp.radius = sqrt(x2y2 + r.z*r.z);
    rp.phi = atan2(r.y, r.x);
    rp.theta = atan2(sqrt(x2y2), r.z);
    return rp;
}
static inline void print_sph(const SPH r){
    cout << "radius: " << r.radius << "\tphi: " << r.phi << "\ttheta: " << r.theta << endl;
}
static inline void print_cart(const CART r){
    cout << "x: " << r.x << "\ty: " << r.y << "\tz: " << r.z << endl;
}

extern "C"
int test1(){
    CART r1 = {1.0, 2.0, 3.0};
    SPH sr1 = cart_to_sph(r1);

    //print_cart(r1);
    //print_sph(sr1);

    return 0;
}

extern "C"
int compute_g(
){
    
    return 0;
}








