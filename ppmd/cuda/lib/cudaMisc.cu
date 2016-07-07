
#include "cudaMisc.h"







int cudaExclusiveScanDouble(double * d_ptr, const int len){
    thrust_exclusive_scan<double>(d_ptr, len);
    return 0;
}


int cudaExclusiveScanInt(int * d_ptr, const int len){
    thrust_exclusive_scan<int>(d_ptr, len);
    return 0;
}





