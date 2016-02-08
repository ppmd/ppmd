#ifndef __COUNTING_TYPES__
#define __COUNTING_TYPES__

namespace double_counters{
    ulong a = 0;    // addition
    ulong s = 0;    // subtraction
    ulong m = 0;    // multiplication
    ulong d = 0;    // division
    ulong r = 0;    // reads
    ulong w = 0;    // writes
    std::size_t size = sizeof(double);
}


class double_prof {

    public:
        double v; // This is the only data storage 
                  // otherwise there will be a size missmatch.
        
        double_prof(){ this->v = 0; }
        double_prof(double v){ this->v = v; }

        //operators.
        double_prof operator+(const double_prof& other);
        double_prof operator+=(const double_prof& other);
        double_prof operator*(const double_prof& other);
        double_prof operator/(const double_prof& other);
        double_prof operator-(const double_prof& other);

};

double_prof double_prof::operator+(const double_prof& other){
    double_counters::a ++;
    return double_prof(this->v + other.v);
}

double_prof double_prof::operator+=(const double_prof& other){
    double_counters::a ++;
    return double_prof(this->v += other.v);
}

double_prof double_prof::operator*(const double_prof& other){
    double_counters::m ++;
    return double_prof(this->v * other.v);
}

double_prof double_prof::operator/(const double_prof& other){
    double_counters::d ++;
    return double_prof(this->v / other.v);
}

double_prof double_prof::operator-(const double_prof& other){
    double_counters::s ++;
    return double_prof(this->v - other.v);
}





namespace int_counters{
    ulong a = 0;    // addition
    ulong s = 0;    // subtraction
    ulong m = 0;    // multiplication
    ulong d = 0;    // division
    ulong r = 0;    // reads
    ulong w = 0;    // writes
    std::size_t size = sizeof(int);
}


class int_prof {

    public:
        int v; // This is the only data storage 
                  // otherwise there will be a size missmatch.
        
        int_prof(){ this->v = 0; }
        int_prof(int v){ this->v = v; }
        

        //operators.
        int_prof operator+(const int_prof& other);
        int_prof operator+=(const int_prof& other);
        int_prof operator*(const int_prof& other);
        int_prof operator/(const int_prof& other);
        int_prof operator-(const int_prof& other);

};

int_prof int_prof::operator+(const int_prof& other){
    int_counters::a ++;
    return int_prof(this->v + other.v);
}

int_prof int_prof::operator+=(const int_prof& other){
    int_counters::a ++;
    return int_prof(this->v += other.v);
}

int_prof int_prof::operator*(const int_prof& other){
    int_counters::m ++;
    return int_prof(this->v * other.v);
}

int_prof int_prof::operator/(const int_prof& other){
    int_counters::d ++;
    return int_prof(this->v / other.v);
}

int_prof int_prof::operator-(const int_prof& other){
    int_counters::s ++;
    return int_prof(this->v - other.v);
}



#endif
