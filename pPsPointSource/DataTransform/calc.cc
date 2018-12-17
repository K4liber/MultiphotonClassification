#include "calc.hh"
#include <TRandom3.h>
#include <random>
#include "TVector3.h"
#include <stdlib.h>
#include "TCanvas.h"
#include "TH1F.h"

std::random_device rd;
std::mt19937 gen(rd());

Double_t r_norm(Double_t mean, double sigmaE)
{
  std::normal_distribution<Double_t> d(mean, sigmaE);
  return d(gen);
}

Double_t sigmaE(Double_t E, Double_t coeff = 0.0444)
{
  return coeff / TMath::Sqrt(E) * E;
}

Double_t smearEnergy(Double_t energy)
{
  return r_norm(energy, 1000. * sigmaE((energy) * 1. / 1000.));
}

Double_t smearTime(Double_t time)
{
  return r_norm(time, 0.15); // Sigma = 150ps
}

Double_t smearZ(Double_t z)
{
  return r_norm(z, 10); // Sigma = 1cm
}

// WK My version
TVector3 recoEmissionPoint(double x1, double y1, double z1, double time1,
                            double x2, double y2, double z2, double time2) {
  TVector3 hit1;
  TVector3 hit2;
  double t1 = 0;
  double t2 = 0;
  if (time1 < time2) {
    hit1.SetXYZ(x1, y1, z1);
    hit2.SetXYZ(x2, y2, z2);
    t1 = time1;
    t2 = time2;
  } else {
    hit2.SetXYZ(x1, y1, z1);
    hit1.SetXYZ(x2, y2, z2);
    t2 = time1;
    t1 = time2;
  }
  TVector3 vCenter = (hit1 + hit2);
  vCenter = 0.5 * vCenter;
  TVector3 vVect = hit2 - hit1;
  TVector3 vVersor = vVect.Unit();
  double tDiff = (t1 - t2) / 2;
  double speedOfLight = 300; /// mm/ns
  double shift = tDiff * speedOfLight;
  TVector3 vect;
  vect.SetXYZ(vCenter.X() + shift * std::abs(vVersor.X()),
              vCenter.Y() + shift * std::abs(vVersor.Y()),
              vCenter.Z() + shift * std::abs(vVersor.Z()));
  return vect;
}

void energySmearTest(int nProbes) {
    TH1F *energies = new TH1F("Energy smear test", "Energy smear test", 40, 70, 130);
    for (int i=0;i<nProbes;i++) {
      energies->Fill(smearEnergy(100));
    }
    TCanvas c1("c", "c", 2000, 2000);
    energies->SetLineColor(kBlack);
    energies->Draw();
    c1.SaveAs("energySmearTest.png");
}

void zSmearTest(int nProbes) {
    TH1F *energies = new TH1F("Z smear test", "Z smear test", 40, 70, 130);
    for (int i=0;i<nProbes;i++) {
      energies->Fill(smearZ(100));
    }
    TCanvas c1("c", "c", 2000, 2000);
    energies->SetLineColor(kBlack);
    energies->Draw();
    c1.SaveAs("ZSmearTest.png");
}

void timeSmearTest(int nProbes) {
    TH1F *energies = new TH1F("Time smear test", "Time smear test", 40, 9999999.3, 10000000.7);
    for (int i=0;i<nProbes;i++) {
      energies->Fill(smearTime(10000000));
    }
    TCanvas c1("c", "c", 2000, 2000);
    energies->SetLineColor(kBlack);
    energies->Draw();
    c1.SaveAs("timeSmearTest.png");
}