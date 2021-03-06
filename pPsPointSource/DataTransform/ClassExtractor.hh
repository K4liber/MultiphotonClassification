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
 *  @file ClassExtractor.h
 */

#ifndef CLASSEXTRACTOR_H 
#define CLASSEXTRACTOR_H 
#include <TObject.h>
#include <string>
#include <vector>
#include <TVector3.h>
#include "TTree.h"
#include "TFile.h"
#include <TTreeReader.h>
#include "calc.hh"
#include "Event.h"

class ClassExtractor{
    public:
        ClassExtractor(std::string &inFileName, bool smear, float cut);
        void extractTwoPhotonsEvents(std::string outFileName);
        void setInFileName(std::string &inFileName);
    private:
        std::string inFileName;
        bool smear;
        float cut;
        void saveToFile(std::ofstream& outputFile, TrackInteraction& i1, TrackInteraction& i2, int cl);
};

#endif /*  !CLASSEXTRACTOR_H */