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
        ClassExtractor(std::string &inFileName);
        void extractTwoPhotonsEvents(std::string outFileName, bool smear);
    private:
        std::string inFileName;
        void setInFileName(std::string &inFileName);
};

#endif /*  !CLASSEXTRACTOR_H */