#include <iostream>
#include <boost/numeric/odeint.hpp>
#include <pybind11/pybind11.h>

#define PRECISION double
using namespace boost::numeric::odeint;
using namespace std;
namespace py = pybind11;



// Define custom vector type for complex vectors
//[my_vector
template< size_t MAX_N >
class my_vector
{
    typedef std::vector< std::complex<PRECISION> > vector;

public:
    typedef vector::iterator iterator;
    typedef vector::const_iterator const_iterator;

public:
    my_vector( const size_t N )
        : m_v( N )
    {
        m_v.reserve( MAX_N );
    }

    my_vector()
        : m_v()
    {
        m_v.reserve( MAX_N );
    }

// ... [ implement container interface ]
//]
    const std::complex<PRECISION> & operator[]( const size_t n ) const
    { return m_v[n]; }

    std::complex<PRECISION> & operator[]( const size_t n )
    { return m_v[n]; }

    iterator begin()
    { return m_v.begin(); }

    const_iterator begin() const
    { return m_v.begin(); }

    iterator end()
    { return m_v.end(); }

    const_iterator end() const
    { return m_v.end(); }

    size_t size() const
    { return m_v.size(); }

    void resize( const size_t n )
    { m_v.resize( n ); }

private:
    std::vector< std::complex<PRECISION> > m_v;

};

//[my_vector_resizeable
// define my_vector as resizeable

namespace boost { namespace numeric { namespace odeint {

template<size_t N>
struct is_resizeable< my_vector<N> >
{
    typedef boost::true_type type;
    static const bool value = type::value;
};

} } }

//Define a type for the density vector
typedef my_vector<8> state_type;
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
  void operator() (const state_type &x, state_type &dx, const PRECISION t){
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
    dx[7] = (-I*conj(Om23)/2.0)*x[0]-I*conj(Om23)*x[1]-(I*conj(Om13)/2.0)*x[2]-conj(gamma32tilde)*x[7]+I*conj(Om23)/2.0;
  }
};
struct push_back_state_and_time{
  vector<state_type>& m_states;
  vector<PRECISION>& m_times;

  push_back_state_and_time(vector<state_type> &states , vector<PRECISION> &times )
  : m_states( states ) , m_times( times ) { }

  void operator()(const state_type &x , PRECISION t)
  {
      m_states.push_back(x);
      m_times.push_back(t);
  }
};
tuple<vector<state_type>, vector<PRECISION>> integrate_bloch(
  vector<complex<PRECISION>> p, state_type x0,
              const PRECISION t1){
  /*
   * [main description]
   * @return [description]
   */
  // make sure resizing is ON
  BOOST_STATIC_ASSERT( is_resizeable<state_type>::value == true );
  // Instantiate deriv_notransit
  deriv_notransit rhs(p);
  // Compute number of steps / dt
  PRECISION dt = 1e-11;
  vector<state_type> xout;
  vector<PRECISION> tout;
  integrate_const(runge_kutta4 < state_type >() , rhs , x0 , 0.0 , t1 , dt, push_back_state_and_time(xout, tout) );
  return {xout, tout};
};

// For python compatibility
PYBIND11_MODULE(example, m) {
    m.doc() = "Integration routine using a RK4 solver to solve the MBE.\nThe syntax is integrate_bloch(p, x0, t1) where: \np is a list of params, x0 the initial condition and t1 the end time."; // optional module docstring
    m.def("integrate_bloch", &integrate_bloch, "A function which integrates MBE");
}


// For debugging
int main(){
  vector<complex<PRECISION>> p0 = { 1.64649793e+01+0.00000000e+00*I,  8.63247752e-01+0.00000000e+00*I,
 -5.04780465e-01+0.00000000e+00*I,  1.75781250e-04+0.00000000e+00*I,
  2.48046875e-03+0.00000000e+00*I,  3.81075189e+07+0.00000000e+00*I,
  1.35886167e+08+0.00000000e+00*I,  1.35886167e+08+0.00000000e+00*I,
  3.21033057e+05+4.29392884e+10*I,  4.65644660e+07+6.14952836e+10*I,
  4.65644660e+07+1.85559952e+10*I,  1.00000000e-03+0.00000000e+00*I,
  1.25000000e-03+0.00000000e+00*I};
  // deriv_notransit der(p0);
  // cout << "Gamma = " << der.Gamma << "\n";
  state_type x(8);
  x[0] = 3./8. + 0.0*I; x[1] = 5./8. +0.0*I; x[2] = 1.0+0.0*I; x[3] = 1.0+0.0*I;
  x[4] = 1.0+0.0*I; x[5] = 1.0+0.0*I; x[6] = 1.0+0.0*I; x[7] = 1.0+0.0*I;
  // PRECISION t = 0;
  // der(x, x, t);
  // cout << "RHS(x=0, t=0)  = " << x[0] << "\n";
  vector<state_type> xout;
  vector<PRECISION> tout;
  tuple<vector<state_type>, vector<PRECISION>> {xout, tout} = integrate_bloch(p0, x, 10.0e-6);
  return 0;

}
