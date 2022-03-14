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

        // Returns boolean value whether the ray should pass or not
         virtual bool isValidRay(const Ray &rotatedRayOnInputplane) const {return false;} ;
         // Print a string version to cout
         virtual void print(){};
         
         // I believe this should be refactored out at some point
         // Other could does not need to know that the passnopass function is using intersections at a certain plane
         // One might for example attempt different kinds of implementations
         virtual Float distanceInputToIntersectPlane()=0;

         // This return the size of the pupil seen on-axis   and at a distance equal to
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
         // This returns the vector index for the first ellipse that is greater or qual than the givendistance
         int findClosestPosition(Float offAxisDistanceOnInputPlane) const{
              for(int i=0; i<positions.size();i++){
               if(positions[i]>=offAxisDistanceOnInputPlane){
                  return i;
               }       
            }
            // This means that the offaxis distance was beyond the maximum positions, assume vignetted
            return -1;
         }

   


         Float interpolationFactorNeighbours(int ellipseIndexRight,Float offAxisDistanceOnInputPlane) const{
            Float positionRight= positions[ellipseIndexRight]; // Upperlimit of interval
            Float positionLeft; // Underlimit of interval
            if(ellipseIndexRight >0){
                  positionLeft = positions[ellipseIndexRight-1];// Underlimit of interval
           }else{
               return 0; // If there is no positionRight (outside of vector)
           }
           
           // Linear interpolation factor from 0 to 1  . At zero  at left position and for one at right position.
            Float alpha = (offAxisDistanceOnInputPlane-positionLeft)/(positionRight-positionLeft);

            return alpha;

         }

         
         // Checks whether a point (x,y) lies within ellipse with given radii and centers
         bool pointInEllipse(Float radiusX, Float radiusY, Float centerX, Float centerY, Float x, Float y) const{
               
               Float radiusXSquared = radiusX*radiusX;
               Float radiusYSquared = radiusY*radiusY;
               Float distX = (x-centerX)*(x-centerX);
               Float distY = (y-centerY)*(y-centerY);

               bool pass = (distX/radiusXSquared + distY/radiusYSquared) <= 1;

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
               Float radiusX,radiusY,centerX,centerY;
               if(ellipseIndex==0) { // There is no position left from this to interpolate towards
                  radiusX = radiiX[0];radiusY = radiiY[0];
                  centerX = centersX[0];centerY = centersY[0];
               }else{
                  // Interpolate radius and position of ellipse from nearest neighbours
                  Float interpFactor = interpolationFactorNeighbours(ellipseIndex,offaxisDistanceInputplane);
                  radiusX=Lerp(interpFactor,radiiX[ellipseIndex-1],radiiX[ellipseIndex]);
                  radiusY=Lerp(interpFactor,radiiY[ellipseIndex-1],radiiY[ellipseIndex]);
                  centerX=Lerp(interpFactor,centersX[ellipseIndex-1],centersX[ellipseIndex]);
                  centerY=Lerp(interpFactor,centersY[ellipseIndex-1],centersY[ellipseIndex]);
               }
               
               return pointInEllipse(radiusX,radiusY,centerX,centerY,pointOnCirclePlane.x,pointOnCirclePlane.y);
            }else{ 
               return false;
            }
        }

        Float distanceInputToIntersectPlane() { return circlePlaneZ; }
    };

  



 class PassNoPassCircleIntersection: public PassNoPass{
       
        public:
        PassNoPassCircleIntersection(pstd::vector<Float> radii,pstd::vector<Float> sensitivities,Float rayPassPlaneDistanceFromInput): radii(radii),sensitivities(sensitivities),rayPassPlaneDistanceFromInput(rayPassPlaneDistanceFromInput){ 
        };
        pstd::vector<Float> radii;
        pstd::vector<Float> sensitivities;
        Float rayPassPlaneDistanceFromInput;


         private:
    

        
         public:

        void print(){
           std::cout << "passnopass circle" << "\n";
        };

         Float getOnAxisRadiusEstimate(){
            return radii[0];
         }
        bool isValidRay(const Ray &rotatedRayOnInputplane) const {
          
            Vector3f dir = (rotatedRayOnInputplane.d);

           // The off axis distance on the input plane determines how much the circles change
            Float offaxisDistanceInputplane=rotatedRayOnInputplane.o.y; // Off axis distance IN input plane

            // To calculate the intersections with all circles we project the ray to the circle plane
            Float alpha = rayPassPlaneDistanceFromInput / dir.z;  //  not needed
            Point3f pointOnCirclePlane = rotatedRayOnInputplane.o + alpha * dir; 
            //std::cout << rotatedRayOnInputplane.o.z << "\n";
          Float pointOnCirclePlaneXsquared = pointOnCirclePlane.x * pointOnCirclePlane.x; // only compute once
          // Because it is rotated, this x coordinate should be zero. 
          
     
          // The ray intersection with the input plane should be within all circles
          // If not, by construction the ray is vignetted (should not be traced)
            for (int i = 0; i < radii.size(); i++) {
            // Calculate the length
             Float radius;Float distanceFromCenterY;
             distanceFromCenterY = pointOnCirclePlane.y-offaxisDistanceInputplane*sensitivities[i];
              
              // Test whether the point projected onto circle plane lies within all circles
             // Returns false when it finds the first circle which did not intersect with the ray
             if (std::sqrt(pointOnCirclePlaneXsquared+ distanceFromCenterY*distanceFromCenterY) >= radii[i]) {
                   return false;
             }
    } 
    return true;
           
        }

        Float distanceInputToIntersectPlane() { return rayPassPlaneDistanceFromInput; }
    };

    }  // namespace pbrt
    
#endif  // PBRT_PASSNOPASS_H