/**
 *  @copyright Copyright 2018 The J-PET Framework Authors. All rights reserved.
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may find a copy of the License in the LICENCE file.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  @file ClassExtractor.cxx
 */

#include "./ClassExtractor.hh"

ClassExtractor::ClassExtractor(std::string &in) {
    this->inFileName = in;
}

/*
### extractTwoPhotonsEvents ###
We are taking into consideration different signl sources:
*** class 0 - pPs - pPs
*** class 1 - prompt - pPs                                      
*** class 2 - prompt - pPs (distracted in phantom)              
*** class 3 - prompt - pPs
*/
void ClassExtractor::extractTwoPhotonsEvents(std::string outFileName, bool smear) {
  std::ofstream outputFile;
  outputFile.open(outFileName);
  TFile file(this->inFileName.c_str(), "READ");
  TTreeReader reader("Tree", &file);
  TTreeReaderValue<Event> event(reader, "Event");
  double cut = 100;
  double numberOfEvents = 0;
  while (reader.Next()) {
    std::cout << "\r" << "Number of evaluated events: " << numberOfEvents++;
    if (event->fTracks.size() == 2) {
      auto &iterTrack1 = event->fTracks[0].fTrackInteractions.front();
      auto &iterTrack2 = event->fTracks[1].fTrackInteractions.front();
      if (iterTrack1.fEnergyDeposition > cut && iterTrack2.fEnergyDeposition > cut &&
        iterTrack1.fEnergyBeforeProcess == 511 && iterTrack2.fEnergyBeforeProcess == 511) {
        
        if (smear) {
            outputFile << std::setprecision(5) << iterTrack1.fVolumeCenter.X() << "," 
                << iterTrack1.fVolumeCenter.Y() << ","
                << smearZ(iterTrack1.fHitPosition.Z()) << "," 
                << iterTrack2.fVolumeCenter.X() << "," 
                << iterTrack2.fVolumeCenter.Y() << ","
                << smearZ(iterTrack2.fHitPosition.Z()) << "," 
                << smearEnergy(iterTrack1.fEnergyDeposition) << ","
                << smearEnergy(iterTrack2.fEnergyDeposition) << "," 
                << std::setprecision(12) << iterTrack1.fGlobalTime << "," 
                << iterTrack2.fGlobalTime << ","
                << iterTrack1.fVolumeName << "," 
                << iterTrack1.fVolumeName << "," 
                << 0 // This stands for pPs - pPs event
                << std::endl;
        } else {
            outputFile << std::setprecision(5) << iterTrack1.fHitPosition.X() << "," 
                << iterTrack1.fHitPosition.Y() << ","
                << iterTrack1.fHitPosition.Z() << "," 
                << iterTrack2.fHitPosition.X() << "," 
                << iterTrack2.fHitPosition.Y() << ","
                << iterTrack2.fHitPosition.Z() << "," 
                << iterTrack1.fEnergyDeposition << ","
                << iterTrack2.fEnergyDeposition << "," 
                << std::setprecision(15) << iterTrack1.fGlobalTime << "," 
                << iterTrack2.fGlobalTime << ","
                << iterTrack1.fVolumeName << "," 
                << iterTrack1.fVolumeName << "," 
                << 0 // This stands for pPs - pPs event
                << std::endl;
        }
      }
    }
  }
}

void ClassExtractor::setInFileName(std::string &in) {
    this->inFileName = in;
}