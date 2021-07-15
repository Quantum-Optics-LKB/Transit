#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <string>

#include "Lorenz_8D_SystemDefinition.cuh"
#include "SingleSystem_PerThread_Interface.cuh"

#define PI 3.14159265358979323846

using namespace std;

// Solver Configuration
#define SOLVER RKCK45        // RK4, RKCK45
#define PRECISION float  // float, double
const int NT   = 92160; // NumberOfThreads
const int SD   = 8;     // SystemDimension
const int NCP  = 1;     // NumberOfControlParameters
const int NSP  = 0;     // NumberOfSharedParameters
const int NISP = 0;     // NumberOfIntegerSharedParameters
const int NE   = 0;     // NumberOfEvents
const int NA   = 0;     // NumberOfAccessories
const int NIA  = 0;     // NumberOfIntegerAccessories
const int NDO  = 0;     // NumberOfPointsOfDenseOutput

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, int);
void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, int);
int RunWithOutput(bool out);


int main()
{
	int NumberOfProblems = NT;
	int BlockSize        = 64;

	ListCUDADevices();

	int MajorRevision  = 7;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);

	PrintPropertiesOfSpecificDevice(SelectedDevice);


	int NumberOfParameters_R = NumberOfProblems;
	PRECISION R_RangeLower = 0.0;
    PRECISION R_RangeUpper = 21.0;
		vector<PRECISION> Parameters_R_Values(NumberOfParameters_R,0);
		Linspace(Parameters_R_Values, R_RangeLower, R_RangeUpper, NumberOfParameters_R);


	ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION> ScanLorenz(SelectedDevice);

	ScanLorenz.SolverOption(ThreadsPerBlock, BlockSize);
	ScanLorenz.SolverOption(InitialTimeStep, 1.0e-6);

  // ScanLorenz.SolverOption(MaximumTimeStep, 1.0e3);
	ScanLorenz.SolverOption(MinimumTimeStep, 1.0e-12);
	// ScanLorenz.SolverOption(TimeStepGrowLimit, 10.0);
	// ScanLorenz.SolverOption(TimeStepShrinkLimit, 0.2);

  // for (int i=0; i<SD; i++){
	// ScanLorenz.SolverOption(RelativeTolerance, i, 1.0e-9);
	// ScanLorenz.SolverOption(AbsoluteTolerance, i, 1.0e-9);
  // }

	clock_t SimulationStart;
	clock_t SimulationEnd;

	FillSolverObject(ScanLorenz, Parameters_R_Values, NT);

	SimulationStart = clock();

	ScanLorenz.SynchroniseFromHostToDevice(All);

	ScanLorenz.Solve();

	ScanLorenz.SynchroniseFromDeviceToHost(All);
	ScanLorenz.InsertSynchronisationPoint();
	ScanLorenz.SynchroniseSolver();

	SimulationEnd = clock();
		cout << "Total simulation time:           " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl;
		cout << "Simulation time / 1000 RKCK45 step: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC / 10 << "ms" << endl;
		cout << "Ensemble size:                   " << NT << endl << endl;

	// SaveData(ScanLorenz, NT);

	cout << "Test finished!" << endl;
}

int RunWithOutput(bool out)
{
	int NumberOfProblems = NT;
	int BlockSize        = 64;

	ListCUDADevices();

	int MajorRevision  = 7;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);

	PrintPropertiesOfSpecificDevice(SelectedDevice);


	int NumberOfParameters_R = NumberOfProblems;
	PRECISION R_RangeLower = 0.0;
    PRECISION R_RangeUpper = 21.0;
		vector<PRECISION> Parameters_R_Values(NumberOfParameters_R,0);
		Linspace(Parameters_R_Values, R_RangeLower, R_RangeUpper, NumberOfParameters_R);


	ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION> ScanLorenz(SelectedDevice);

	ScanLorenz.SolverOption(ThreadsPerBlock, BlockSize);
	ScanLorenz.SolverOption(InitialTimeStep, 1.0e-6);

  // ScanLorenz.SolverOption(MaximumTimeStep, 1.0e3);
	ScanLorenz.SolverOption(MinimumTimeStep, 1.0e-12);
	// ScanLorenz.SolverOption(TimeStepGrowLimit, 10.0);
	// ScanLorenz.SolverOption(TimeStepShrinkLimit, 0.2);

  // for (int i=0; i<SD; i++){
	// ScanLorenz.SolverOption(RelativeTolerance, i, 1.0e-9);
	// ScanLorenz.SolverOption(AbsoluteTolerance, i, 1.0e-9);
  // }

	clock_t SimulationStart;
	clock_t SimulationEnd;

	FillSolverObject(ScanLorenz, Parameters_R_Values, NT);

	SimulationStart = clock();

	ScanLorenz.SynchroniseFromHostToDevice(All);

	ScanLorenz.Solve();

	ScanLorenz.SynchroniseFromDeviceToHost(All);
	ScanLorenz.InsertSynchronisationPoint();
	ScanLorenz.SynchroniseSolver();

	SimulationEnd = clock();
		cout << "Total simulation time:           " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl;
		cout << "Simulation time / 1000 RKCK45 step: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC / 10 << "ms" << endl;
		cout << "Ensemble size:                   " << NT << endl << endl;

	// SaveData(ScanLorenz, NT);

	cout << "Test finished!" << endl;
	if (out){
		return 0;
	}
	return 1;
}
// AUXILIARY FUNCTION -----------------------------------------------------------------------------

void Linspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
    PRECISION Increment;

	x[0]   = B;

	if ( N>1 )
	{
		x[N-1] = E;
		Increment = (E-B)/(N-1);

		for (int i=1; i<N-1; i++)
		{
			x[i] = B + i*Increment;
		}
	}
}

void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& R_Values, int NumberOfThreads)
{
	int ProblemNumber = 0;
	for (int k=0; k<NumberOfThreads; k++)
	{
		Solver.SetHost(ProblemNumber, TimeDomain,  0, 0 );
		Solver.SetHost(ProblemNumber, TimeDomain,  1, 0.001*10000.0 );

		Solver.SetHost(ProblemNumber, ActualState, 0, 10.0 );
		Solver.SetHost(ProblemNumber, ActualState, 1, 10.0 );
		Solver.SetHost(ProblemNumber, ActualState, 2, 10.0 );
    Solver.SetHost(ProblemNumber, ActualState, 3, 0.0 );
		Solver.SetHost(ProblemNumber, ActualState, 4, 0.0 );
		Solver.SetHost(ProblemNumber, ActualState, 5, 0.0 );
    Solver.SetHost(ProblemNumber, ActualState, 6, 0.0 );
		Solver.SetHost(ProblemNumber, ActualState, 7, 0.0 );

		Solver.SetHost(ProblemNumber, ControlParameters, 0, R_Values[k] );

		ProblemNumber++;
	}
}

void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, int NumberOfThreads)
{
	ofstream DataFile;
	DataFile.open ( "Lorenz.txt" );

	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);

	for (int tid=0; tid<NumberOfThreads; tid++)
	{
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ControlParameters, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 1) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 2) << ',';
    DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 3) << ',';
    DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 4) << ',';
    DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 5) << ',';
    DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 6) << ',';
    DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 7);
		DataFile << '\n';
	}

	DataFile.close();
}
