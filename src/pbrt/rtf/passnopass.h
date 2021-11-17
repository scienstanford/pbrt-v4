// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_PASSNOPASS_H
#define PBRT_PASSNOPASS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/ray.h>
#include <pbrt/samplers.h>
#include <pbrt/util/image.h>
#include <pbrt/util/scattering.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
namespace pbrt {

     class PassNoPass{
        public:
         virtual bool isValidRay(Ray rayOnInputPlane){return false;};
         virtual void print(){
            std::cout << "passnopass base" << "\n";
         };
     };



    class PassNoPassEllipse: public PassNoPass{
        public:
        PassNoPassEllipse(pstd::vector<Float> positions,pstd::vector<Float> radiiX,pstd::vector<Float> radiiY,pstd::vector<Float> centersX,pstd::vector<Float> centersY,Float circlePlaneZ): positions(positions),radiiX(radiiX),radiiY(radiiY),centersX(centersX),centersY(centersY),circlePlaneZ(circlePlaneZ){ 
        };
         pstd::vector<Float> positions;
        pstd::vector<Float> radiiX;
        pstd::vector<Float> radiiY;
        pstd::vector<Float> centersX;
        pstd::vector<Float> centersY;
        Float circlePlaneZ;



        void print(){
           std::cout << "passnopass ellipse" << "\n";
        };
        

        bool isValidRay(Ray rayOnInputPlane){
            return false;
           
        }
     
   };



}

#endif  // PBRT_PASSNOPASS_H