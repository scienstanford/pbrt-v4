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
         virtual bool isValidRay(const Ray &rotatedRayOnInputplane) const {return false;} ;
         virtual void print(){};
         virtual Float distanceInputToIntersectPlane()=0;
         virtual Float getOnAxisRadiusEstimate()=0;

   

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


         private:
         int findClosestPosition(Float offAxisDistanceOnInputPlane) const{
            
            for (int i = 0; i < positions.size(); i++) {
               if(positions[i]>offAxisDistanceOnInputPlane){
                  // Make more accurate by choosing closest neighbour, will be slower
                  // But should be no problem when accurately sampled
                  return i;
               }       
            }
            return -1;
         }

         bool pointInEllipse(int ellipseIndex,int x, int y) const{
               Float radiusXSquared = radiiX[ellipseIndex]*radiiX[ellipseIndex];
               Float radiusYSquared = radiiY[ellipseIndex]*radiiY[ellipseIndex];
               Float distX = (x-centersX[ellipseIndex])*(x-centersX[ellipseIndex]);;
               Float distY = (y-centersY[ellipseIndex])*(y-centersY[ellipseIndex]);

               bool pass = (distX/radiusXSquared + distY/radiusYSquared) <= 1;
            //if(!pass){ std::cout << "nopass";}
            return pass;

         }
         public:

        void print(){
           std::cout << "passnopass ellipse" << "\n";
        };

         Float getOnAxisRadiusEstimate(){
            return radiiX[0];
         }
        bool isValidRay(const Ray &rotatedRayOnInputplane) const {
            Vector3f dir = (rotatedRayOnInputplane.d);

            // The off axis distance on the input plane determines how much the circles
            // change
            Float offaxisDistanceInputplane = rotatedRayOnInputplane.o.y;  // Off axis distance IN input plane

            // To calculate the intersections with all circles we project the ray to the
            // circle plane
            Float alpha = circlePlaneZ / dir.z; 
            Point3f pointOnCirclePlane = rotatedRayOnInputplane.o + alpha * dir;
            

            // Choose closest ellipse
            int ellipseIndex = findClosestPosition(offaxisDistanceInputplane);

            // Test whether the point projected onto intersection plane lies within the ellipse
            if(ellipseIndex>=0){
               return pointInEllipse(ellipseIndex,pointOnCirclePlane.x,pointOnCirclePlane.y);
            }else{ 
               return false;
            }
        }

        Float distanceInputToIntersectPlane() { return circlePlaneZ; }
    };

    }  // namespace pbrt

#endif  // PBRT_PASSNOPASS_H