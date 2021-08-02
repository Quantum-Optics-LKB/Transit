#include <iostream>
#include <boost/numeric/odeint.hpp>
#include <pybind11/pybind11.h>

#define PRECISION double
using namespace boost::numeric::odeint;
using namespace std;

//Define a type for the density vector
typedef boost::array<complex<PRECISION>, 8> state_type;
typedef runge_kutta_cash_karp54< state_type > error_stepper_type;
typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;

const complex<PRECISION> I( 0.0 , 1.0 );

class deriv_notransit{
private:
  complex<PRECISION> v;
  complex<PRECISION> u0;
  complex<PRECISION> u1;
  complex<PRECISION> xinit;
  complex<PRECISION> yinit;
  complex<PRECISION> Gamma;
  complex<PRECISION> Omega13;
  complex<PRECISION> Omega23;
  complex<PRECISION> gamma21tilde;
  complex<PRECISION> gamma31tilde;
  complex<PRECISION> gamma32tilde;
  complex<PRECISION> waist;
  complex<PRECISION> r0;
public:
  //class constructor
  deriv_notransit(vector<complex<PRECISION>> p){
    v = p[0];
    u0 = p[1];
    u1 = p[2];
    xinit = p[3];
    yinit = p[4];
    Gamma = p[5];
    Omega13 = p[6];
    Omega23 = p[7];
    gamma21tilde = p[8];
    gamma31tilde = p[9];
    gamma32tilde = p[10];
    waist = p[11];
    r0 = p[12];
  }
  //Define RHS of the equation
  void rhs(const state_type &x, state_type &dx, const PRECISION t){
    complex<PRECISION>  r_sq = (xinit+u0*v*t - r0)*(xinit+u0*v*t - r0) +
           (yinit+u1*v*t - r0)*(yinit+u1*v*t - r0);
    complex<PRECISION> Om23 = Omega23 * exp(-r_sq/(2.0*waist*waist));
    complex<PRECISION> Om13 = Omega13 * exp(-r_sq/(2.0*waist*waist));
    dx[0] = (-Gamma/2.0)*x[0]-(Gamma/2.0)*x[1]+(I*conj(Om13)/2.0)*x[4]-(I*Om13/2.0)*x[5]+Gamma/2.0;
    dx[1] = (-Gamma/2.0)*x[0]-(Gamma/2.0)*x[1]+(I*conj(Om23)/2.0)*x[6]-(I*Om23/2.0)*x[7]+Gamma/2.0;
    dx[2] = -gamma21tilde*x[2]+(I*conj(Om23)/2.0)*x[4]-(I*Om13/2.0)*x[7];
    dx[3] = -conj(gamma21tilde)*x[3] - (I*Om23/2.0)*x[5] + (I*conj(Om13)/2.0)*x[6];
    dx[4] = I*Om13*x[0] + (I*Om13/2.0)*x[1] + (I*Om23/2.0)*x[2] - gamma31tilde*x[4]-I*Om13/2.0;
    dx[5] = -I*conj(Om13)*x[0]-I*(conj(Om13)/2.0)*x[1]-(I*conj(Om23)/2.0)*x[3]-conj(gamma31tilde)*x[5]+I*conj(Om13)/2.0;
    dx[6] = (I*Om23/2.0)*x[0]+I*Om23*x[1]+(I*Om13/2.0)*x[3]-gamma32tilde*x[6] - I*Om23/2.0;
    dx[7] = (-I*conj(Om23)/2.0)*x[0]-I*conj(Om23)*x[1]-(I*conj(Om13)/2.0)*x[2.0]-conj(gamma32tilde)*x[7]+I*conj(Om23)/2.0;
  }
};
void output_filler(vector<PRECISION>, vector<complex(PRECISION)>){
  
};
// For debugging
int main(){
  vector<complex<PRECISION>> p0 = {0.0+0.0*I, 0.0+0.0*I,0.0+0.0*I,0.0+0.0*I,0.0+0.0*I,0.0+0.0*I,0.0+0.0*I,0.0+0.0*I,0.0+0.0*I,0.0+0.0*I,0.0+0.0*I,1.0+0.0*I,0.0+0.0*I};
  deriv_notransit der(p0);
  // cout << "Gamma = " << der.Gamma << "\n";
  state_type x = {1.0+0.0*I, 1.0+0.0*I, 1.0+0.0*I, 1.0+0.0*I, 1.0+0.0*I, 1.0+0.0*I, 1.0+0.0*I, 1.0+0.0*I};
  PRECISION t = 0;
  der.rhs(x, x, t);
  cout << "RHS(x=0, t=0)  = " << x[0] << "\n";
  return 0;

}
