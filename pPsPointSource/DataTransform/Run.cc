#include "GlobalActorReader.hh"
#include "ClassExtractor.hh"
#include "Event.h"
#include "ClassExtractor.hh"
#include "calc.hh"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2.h"
#include "TVector3.h"
#include <TLorentzVector.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <TRandom3.h>
#include <random>
#include <TTreeReader.h>

using namespace std;

int energyCut = 100; // keV
bool createStats = false;
bool smear = true;

struct gammaTrack {
  int eventID;
  int trackID;
  Double_t energy;
  Double_t x;
  Double_t y;
  Double_t z;
  Double_t time;
  Double_t globalTime;
  string volume;
  bool pPs;
};

vector<gammaTrack> data;
TH1F *timeDiff = new TH1F("Pairs time differences", "Pairs time differences",
                          200, 0, 10000000);
TH1F *htimeDiff2 =
    new TH1F("htimeDiff2", "Pairs time differences", 1000, -5, 5);
TH1F *hTimeDiffPPsSameLayer =
    new TH1F("hTimeDiffPPsSameLayer", "hTimeDiffPPsSameLayer", 1000, -1, 1);
TH2 *hRecoEmissionPointTrue =
    new TH2D("hRecoEmissionPointTrue", "hRecoEmissionPointTrue", 1000, -5, 5,
             1000, -5, 5);
TH2 *hRecoEmissionPointAll = new TH2D(
    "hRecoEmissionPointAll", "hRecoEmissionPointAll", 1000, -5, 5, 1000, -5, 5);
TH2 *pos = new TH2D("XY", "XY", 1200, -600, 600, 1200, -600, 600);

void addEntryToEvent(const GlobalActorReader &gar, Event *outEvent) {
  assert(outEvent);
  outEvent->fEventID = gar.GetEventID();

  TrackInteraction trkStep;
  trkStep.fHitPosition = gar.GetProcessPosition();
  trkStep.fVolumeCenter = gar.GetScintilatorPosition();
  trkStep.fLocalTime = gar.GetLocalTime();
  trkStep.fGlobalTime = gar.GetGlobalTime();
  trkStep.fEnergyDeposition = gar.GetEnergyLossDuringProcess();
  trkStep.fEnergyBeforeProcess = gar.GetEnergyBeforeProcess();
  trkStep.fVolumeName = gar.GetVolumeName();

  int currentTrackID = gar.GetTrackID();
  if (!outEvent->fTracks.empty()) {
    auto &lastTrack = outEvent->fTracks.back();
    if (lastTrack.fTrackID == currentTrackID) {
      lastTrack.fTrackInteractions.push_back(trkStep);
    } else {
      Track trk;
      trk.fEmissionEnergy = gar.GetEmissionEnergyFromSource();
      trk.fTrackID = currentTrackID;
      trk.fTrackInteractions.push_back(trkStep);
      outEvent->fTracks.push_back(trk);
    }
  } else {
    Track trk;
    trk.fEmissionEnergy = gar.GetEmissionEnergyFromSource();
    trk.fTrackID = currentTrackID;
    trk.fTrackInteractions.push_back(trkStep);
    outEvent->fTracks.push_back(trk);
  }
}

void clearEvent(Event *outEvent) {
  assert(outEvent);
  outEvent->fEventID = -1;
  outEvent->fTracks.clear();
}

void transformToEventTree(const std::string &inFileName,
                          const std::string &outFileName) {
  TFile fileOut(outFileName.c_str(), "RECREATE");
  TTree *tree = new TTree("Tree", "Tree");
  Event *event = nullptr;
  tree->Branch("Event", &event, 16000, 99);
  try {
    event = new Event;
    GlobalActorReader gar;
    if (gar.LoadFile(inFileName.c_str())) {
      bool isNewEvent = false;
      bool isFirstEvent = false;
      auto previousID = event->fEventID;
      auto currentID = previousID;
      while (gar.Read()) {
        currentID = gar.GetEventID();
        isFirstEvent = (previousID < 0) && (currentID > 0);
        isNewEvent = currentID != previousID;

        if (isFirstEvent) {
        addEntryToEvent(gar, event);
        } else {
          if (isNewEvent) {
            tree->Fill();
            clearEvent(event);
          }
          addEntryToEvent(gar, event);
        }
        previousID = currentID;
      }
      if (event->fEventID > 0) {
        tree->Fill();
        clearEvent(event);
      }
    } else {
      std::cerr << "Loading file failed." << std::endl;
    }
  } catch (const std::logic_error &e) {
    std::cerr << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Udefined exception" << std::endl;
  }
  fileOut.cd();
  assert(tree);
  fileOut.Write();
}

void saveEntry(gammaTrack &gt1, gammaTrack &gt2, bool isPPsEvent,
               ofstream &outputFile) {
  outputFile << gt1.eventID << "," << gt2.eventID << "," << gt1.trackID << ","
             << gt2.trackID << "," << gt1.x << "," << gt1.y
             << "," << gt1.z << "," << gt2.x << "," << gt2.y << "," << gt2.z
             << "," << gt1.energy << "," << gt2.energy << ","
             << gt1.globalTime - gt2.globalTime << "," << gt1.time << "," << gt2.time << ","
             << gt1.volume << "," << gt2.volume << "," << int(isPPsEvent)
             << endl;
}

void readEntry(const GlobalActorReader &gar) {
  // Energy cut - eletronic noise
  Double_t energy = gar.GetEnergyLossDuringProcess();
  if (smear) energy = smearEnergy(energy);
  if (energy > energyCut) {
    TVector3 hitPosition = gar.GetProcessPosition();
    Double_t x = hitPosition.X();
    Double_t y = hitPosition.Y();
    Double_t z = hitPosition.Z();
    Double_t globalTime = gar.GetGlobalTime();
    if (smear) {
      auto scintPosition = gar.GetScintilatorPosition();
      x = scintPosition.X();
      y = scintPosition.Y();
      z = smearZ(z);
      globalTime = smearTime(globalTime);
    }

    gammaTrack newOne;
    newOne.eventID = gar.GetEventID();
    newOne.trackID = gar.GetTrackID();
    newOne.x = x;
    newOne.y = y;
    newOne.z = z;
    newOne.time = gar.GetLocalTime();
    newOne.globalTime = globalTime;
    newOne.energy = energy;
    newOne.pPs = (gar.GetEnergyBeforeProcess() == 511);
    newOne.volume = gar.GetVolumeName();
    data.push_back(newOne);
  }
}

void createCSVFile() {
  ofstream outputFile;
  outputFile.open("data.csv");
  bool isPPsEvent = false;
  int dataSize = data.size();
  int loops = dataSize*dataSize/2 - dataSize;
  int counter = 0;

  for (vector<gammaTrack>::iterator it = data.begin(); it != data.end(); ++it) {
    if (createStats) {
      pos->Fill(it->x, it->y);
    }
    for (vector<gammaTrack>::iterator it2 = next(it); it2 != data.end(); ++it2) {
      // Process log
      cout << "\r" << (counter*100)/loops << "%";
      counter++;
      // Time cut
      const double kTimeCut = 0.1;
      bool areIntimeCut = (abs(it->globalTime - it2->globalTime) <= kTimeCut);
      bool areSameEvents = (it->eventID == it2->eventID);
      bool areDifferentTracks = (it->trackID != it2->trackID);
      if (!areIntimeCut) continue;

      if (createStats) {
        auto recoEPoint = recoEmissionPoint(it->x, it->y, it->z, it->time, it2->x,
                                          it2->y, it2->z, it2->time);
        timeDiff->Fill(std::abs(it->time - it2->time));
        htimeDiff2->Fill(it->time - it2->time);
        hRecoEmissionPointAll->Fill(recoEPoint.X(), recoEPoint.Y());
      }
      if (areSameEvents && areDifferentTracks && it->pPs &&
          it2->pPs) {
        isPPsEvent = true;
        if (createStats) {
          hTimeDiffPPsSameLayer->Fill(it->time - it2->time);
          pos->Fill((it->x * it2->time + it2->x * it->time) /
                        (it->time + it2->time), // x
                    (it->y * it2->time + it2->y * it->time) /
                        (it->time + it2->time) // y
                    );
          auto recoEPoint = recoEmissionPoint(it->x, it->y, it->z, it->time, it2->x,
                                          it2->y, it2->z, it2->time);
          hRecoEmissionPointTrue->Fill(recoEPoint.X(), recoEPoint.Y());
        }
      }

      saveEntry(*it, *it2, isPPsEvent, outputFile);
      // Symmetrical case
      saveEntry(*it2, *it, isPPsEvent, outputFile);
      isPPsEvent = false;
    }
  }
  outputFile.close();
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    cerr << "Invalid number of variables." << endl;
  } else {
    // Handling parameters
    string file_name(argv[1]);
    energyCut = atoi(argv[2]);
    std::stringstream ss(argv[3]);
    if(!(ss >> std::boolalpha >> smear)) {
        cerr << "As smear type true or false." << endl;
        return 1;
    }

    string out_file_name = "out.root";
    transformToEventTree(file_name, out_file_name);
    ClassExtractor* classExtractor = new ClassExtractor(out_file_name, smear, energyCut);
    classExtractor->extractTwoPhotonsEvents("extractedData.csv");
    /*
    try {
      GlobalActorReader gar;

      int counter = 0;
      if (gar.LoadFile(file_name)) {
        while (gar.Read()) {
          if (counter > 10000)
            break;
          readEntry(gar);
          counter++;
        }
      } else {
        cerr << "Loading file failed." << endl;
      }
      //createCSVFile();
      if (createStats) {
        // Time differences
        TCanvas c1("c", "c", 2000, 2000);
        timeDiff->SetLineColor(kBlack);
        timeDiff->Draw();
        c1.SaveAs("timeDiff.png");
        // Positions
        TCanvas c2("c", "c", 2000, 2000);
        pos->SetLineColor(kBlack);
        pos->Draw();
        c2.SaveAs("posXY.png");
        // Time differences part 2
        TCanvas c3("c", "c", 2000, 2000);
        c3.Divide(2, 1);
        c3.cd(1);
        htimeDiff2->SetLineColor(kBlack);
        htimeDiff2->Draw();
        c3.cd(2);
        hTimeDiffPPsSameLayer->Draw();
        c3.SaveAs("timeDiff.png");
        c3.SaveAs("timeDiff.root");
        TCanvas c4("c", "c", 2000, 2000);
        c4.Divide(2, 1);
        c4.cd(1);
        hRecoEmissionPointAll->Draw();
        c4.cd(2);
        hRecoEmissionPointTrue->Draw();
        c4.SaveAs("RecoXY.png");
        c4.SaveAs("RecoXY.root");
      }
    } catch (const logic_error &e) {
      cerr << e.what() << endl;
    } catch (...) {
      cerr << "Udefined exception" << endl;
    }
    */
  }
  return 0;
}
