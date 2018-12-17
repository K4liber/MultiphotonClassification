#include "TVector3.h"

Double_t r_norm(Double_t mean, double sigmaE);
Double_t sigmaE(Double_t E, Double_t coeff);
Double_t smearEnergy(Double_t energy);
Double_t smearTime(Double_t time);
Double_t smearZ(Double_t z);
void energySmearTest(int nProbes);
void zSmearTest(int nProbes);
void timeSmearTest(int nProbes);

// WK My version
TVector3 recoEmissionPoint(double x1, double y1, double z1, double time1,
                            double x2, double y2, double z2, double time2);