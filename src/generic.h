#ifndef __GENERIC__
#define __GENERIC__


#define M_PI 3.14159265358979323846264338327
#define sign(x) (((x) > 0) - ((x) < 0))      
#define isign(x) (((x) < 0) - ((x) > 0))
#define abs_md(x) ((x) < 0 ? -1*(x) : (x))
#define LINIDX_2D(NX,iy,ix)   ((NX)*(iy) + (ix))

/*
#define LINIDX_ZYX(NX,NY,ix,iy,iz)    \
  (  ((NX)+2*(OL))*((NY)+2*(OL))*(iz) \
   + ((NX)+2*(OL))*((iy)-1+(OL))      \
   + ((ix)-1+(OL))                    \
  )
*/

#define LINIDX_ZYX(NX,NY,ix,iy,iz)    \
  (  ((NX)+2)*((NY)+2)*(iz) \
   + ((NX)+2)*(iy)      \
   + (ix)                    \
  )


#endif
