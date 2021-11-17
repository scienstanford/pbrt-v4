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
               int radiusXSquared = radiiX[ellipseIndex]*radiiX[ellipseIndex];
               int radiusYSquared = radiiY[ellipseIndex]*radiiY[ellipseIndex];
               int distX = x*x;
               int distY = (y-centersY[ellipseIndex])*(y-centersY[ellipseIndex]);


            return (distX/radiusXSquared + distY/radiusYSquared) <= 1;

         }
         public:

        void print(){
           std::cout << "passnopass ellipse" << "\n";
        };

        bool isValidRay(const Ray &rotatedRayOnInputplane) const {
            Vector3f dir = (rotatedRayOnInputplane.d);

            // The off axis distance on the input plane determines how much the circles
            // change
            Float offaxisDistanceInputplane = rotatedRayOnInputplane.o.y;  // Off axis distance IN input plane

            // To calculate the intersections with all circles we project the ray to the
            // circle plane
            Float alpha = circlePlaneZ / dir.z; 
            Point3f pointOnCirclePlane = rotatedRayOnInputplane.o + alpha * dir;
            // std::cout << rotatedRayOnInputplane.o.z << "\n";
            Float pointOnIntersectionPlaneXsquared = 0;  // only compute once
            // Because it is rotated, this x coordinate should be zero. 
            

            // Choose closest ellipse
            int ellipseIndex = findClosestPosition(offaxisDistanceInputplane);

            // Test whether the point projected onto intersection plane lies within the ellipse
            if(ellipseIndex>=0){
               return pointInEllipse(ellipseIndex,pointOnCirclePlane.x,pointOnCirclePlane.y);
            }else{ return false;}
        }

        Float distanceInputToIntersectPlane() { return circlePlaneZ; }
    };

    }  // namespace pbrt

#endif  // PBRT_PASSNOPASS_H