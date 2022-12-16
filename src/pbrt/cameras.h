// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CAMERAS_H
#define PBRT_CAMERAS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/ray.h>
#include <pbrt/samplers.h>
#include <pbrt/util/image.h>
#include <pbrt/util/scattering.h>

#include <gsl/gsl_randist.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace pbrt {

// CameraTransform Definition
class CameraTransform {
  public:
    // CameraTransform Public Methods
    CameraTransform() = default;
    explicit CameraTransform(const AnimatedTransform &worldFromCamera);

    PBRT_CPU_GPU
    Point3f RenderFromCamera(Point3f p, Float time) const {
        return renderFromCamera(p, time);
    }
    PBRT_CPU_GPU
    Point3f CameraFromRender(Point3f p, Float time) const {
        return renderFromCamera.ApplyInverse(p, time);
    }
    PBRT_CPU_GPU
    Point3f RenderFromWorld(Point3f p) const { return worldFromRender.ApplyInverse(p); }

    PBRT_CPU_GPU
    Transform RenderFromWorld() const { return Inverse(worldFromRender); }
    PBRT_CPU_GPU
    Transform CameraFromRender(Float time) const {
        return Inverse(renderFromCamera.Interpolate(time));
    }
    PBRT_CPU_GPU
    Transform CameraFromWorld(Float time) const {
        return Inverse(worldFromRender * renderFromCamera.Interpolate(time));
    }

    PBRT_CPU_GPU
    bool CameraFromRenderHasScale() const { return renderFromCamera.HasScale(); }

    PBRT_CPU_GPU
    Vector3f RenderFromCamera(Vector3f v, Float time) const {
        return renderFromCamera(v, time);
    }

    PBRT_CPU_GPU
    Normal3f RenderFromCamera(Normal3f n, Float time) const {
        return renderFromCamera(n, time);
    }

    PBRT_CPU_GPU
    Ray RenderFromCamera(const Ray &r) const { return renderFromCamera(r); }

    PBRT_CPU_GPU
    RayDifferential RenderFromCamera(const RayDifferential &r) const {
        return renderFromCamera(r);
    }

    PBRT_CPU_GPU
    Vector3f CameraFromRender(Vector3f v, Float time) const {
        return renderFromCamera.ApplyInverse(v, time);
    }

    PBRT_CPU_GPU
    Normal3f CameraFromRender(Normal3f v, Float time) const {
        return renderFromCamera.ApplyInverse(v, time);
    }

    PBRT_CPU_GPU
    const AnimatedTransform &RenderFromCamera() const { return renderFromCamera; }

    PBRT_CPU_GPU
    const Transform &WorldFromRender() const { return worldFromRender; }

    std::string ToString() const;

  private:
    // CameraTransform Private Members
    AnimatedTransform renderFromCamera;
    Transform worldFromRender;
};

// CameraWiSample Definition
struct CameraWiSample {
    // CameraWiSample Public Methods
    CameraWiSample() = default;
    PBRT_CPU_GPU
    CameraWiSample(const SampledSpectrum &Wi, const Vector3f &wi, Float pdf,
                   Point2f pRaster, const Interaction &pRef, const Interaction &pLens)
        : Wi(Wi), wi(wi), pdf(pdf), pRaster(pRaster), pRef(pRef), pLens(pLens) {}

    SampledSpectrum Wi;
    Vector3f wi;
    Float pdf;
    Point2f pRaster;
    Interaction pRef, pLens;
};

// CameraRay Definition
struct CameraRay {
    Ray ray;
    SampledSpectrum weight = SampledSpectrum(1);
};

// CameraRayDifferential Definition
struct CameraRayDifferential {
    RayDifferential ray;
    SampledSpectrum weight = SampledSpectrum(1);
};

// CameraBaseParameters Definition
struct CameraBaseParameters {
    CameraTransform cameraTransform;
    Float shutterOpen = 0, shutterClose = 1;
    Film film;
    Medium medium;
    CameraBaseParameters() = default;
    CameraBaseParameters(const CameraTransform &cameraTransform, Film film, Medium medium,
                         const ParameterDictionary &parameters, const FileLoc *loc);
};

// CameraBase Definition
class CameraBase {
  public:
    // CameraBase Public Methods
    PBRT_CPU_GPU
    Film GetFilm() const { return film; }
    PBRT_CPU_GPU
    const CameraTransform &GetCameraTransform() const { return cameraTransform; }

    PBRT_CPU_GPU
    Float SampleTime(Float u) const { return Lerp(u, shutterOpen, shutterClose); }

    void InitMetadata(ImageMetadata *metadata) const;
    std::string ToString() const;

    PBRT_CPU_GPU
    void Approximate_dp_dxy(Point3f p, Normal3f n, Float time, int samplesPerPixel,
                            Vector3f *dpdx, Vector3f *dpdy) const {
        // Compute tangent plane equation for ray differential intersections
        Point3f pCamera = CameraFromRender(p, time);
        Transform DownZFromCamera =
            RotateFromTo(Normalize(Vector3f(pCamera)), Vector3f(0, 0, 1));
        Point3f pDownZ = DownZFromCamera(pCamera);
        Normal3f nDownZ = DownZFromCamera(CameraFromRender(n, time));
        Float d = nDownZ.z * pDownZ.z;

        // Find intersection points for approximated camera differential rays
        Ray xRay(Point3f(0, 0, 0) + minPosDifferentialX,
                 Vector3f(0, 0, 1) + minDirDifferentialX);
        Float tx = -(Dot(nDownZ, Vector3f(xRay.o)) - d) / Dot(nDownZ, xRay.d);
        Ray yRay(Point3f(0, 0, 0) + minPosDifferentialY,
                 Vector3f(0, 0, 1) + minDirDifferentialY);
        Float ty = -(Dot(nDownZ, Vector3f(yRay.o)) - d) / Dot(nDownZ, yRay.d);
        Point3f px = xRay(tx), py = yRay(ty);

        // Estimate $\dpdx$ and $\dpdy$ in tangent plane at intersection point
        Float sppScale =
            GetOptions().disablePixelJitter
                ? 1
                : std::max<Float>(.125, 1 / std::sqrt((Float)samplesPerPixel));
        *dpdx =
            sppScale * RenderFromCamera(DownZFromCamera.ApplyInverse(px - pDownZ), time);
        *dpdy =
            sppScale * RenderFromCamera(DownZFromCamera.ApplyInverse(py - pDownZ), time);
    }

  protected:
    // CameraBase Protected Members
    CameraTransform cameraTransform;
    Float shutterOpen, shutterClose;
    Film film;
    Medium medium;
    Vector3f minPosDifferentialX, minPosDifferentialY;
    Vector3f minDirDifferentialX, minDirDifferentialY;

    // CameraBase Protected Methods
    CameraBase() = default;
    CameraBase(CameraBaseParameters p);

    PBRT_CPU_GPU
    static pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        Camera camera, CameraSample sample, SampledWavelengths &lambda);

    PBRT_CPU_GPU
    Ray RenderFromCamera(const Ray &r) const {
        return cameraTransform.RenderFromCamera(r);
    }

    PBRT_CPU_GPU
    RayDifferential RenderFromCamera(const RayDifferential &r) const {
        return cameraTransform.RenderFromCamera(r);
    }

    PBRT_CPU_GPU
    Vector3f RenderFromCamera(Vector3f v, Float time) const {
        return cameraTransform.RenderFromCamera(v, time);
    }

    PBRT_CPU_GPU
    Normal3f RenderFromCamera(Normal3f v, Float time) const {
        return cameraTransform.RenderFromCamera(v, time);
    }

    PBRT_CPU_GPU
    Point3f RenderFromCamera(Point3f p, Float time) const {
        return cameraTransform.RenderFromCamera(p, time);
    }

    PBRT_CPU_GPU
    Vector3f CameraFromRender(Vector3f v, Float time) const {
        return cameraTransform.CameraFromRender(v, time);
    }

    PBRT_CPU_GPU
    Normal3f CameraFromRender(Normal3f v, Float time) const {
        return cameraTransform.CameraFromRender(v, time);
    }

    PBRT_CPU_GPU
    Point3f CameraFromRender(Point3f p, Float time) const {
        return cameraTransform.CameraFromRender(p, time);
    }

    void FindMinimumDifferentials(Camera camera);
};

// ProjectiveCamera Definition
class ProjectiveCamera : public CameraBase {
  public:
    // ProjectiveCamera Public Methods
    ProjectiveCamera() = default;
    void InitMetadata(ImageMetadata *metadata) const;

    std::string BaseToString() const;

    ProjectiveCamera(CameraBaseParameters baseParameters,
                     const Transform &screenFromCamera, Bounds2f screenWindow,
                     Float lensRadius, Float focalDistance)
        : CameraBase(baseParameters),
          screenFromCamera(screenFromCamera),
          lensRadius(lensRadius),
          focalDistance(focalDistance) {
        // Compute projective camera transformations
        // Compute projective camera screen transformations
        Transform NDCFromScreen =
            Scale(1 / (screenWindow.pMax.x - screenWindow.pMin.x),
                  1 / (screenWindow.pMax.y - screenWindow.pMin.y), 1) *
            Translate(Vector3f(-screenWindow.pMin.x, -screenWindow.pMax.y, 0));
        Transform rasterFromNDC =
            Scale(film.FullResolution().x, -film.FullResolution().y, 1);
        rasterFromScreen = rasterFromNDC * NDCFromScreen;
        screenFromRaster = Inverse(rasterFromScreen);

        cameraFromRaster = Inverse(screenFromCamera) * screenFromRaster;
    }

    // ProjectiveCamera Protected Members
    Transform screenFromCamera, cameraFromRaster;
    Transform rasterFromScreen, screenFromRaster;
    Float lensRadius, focalDistance;
};

// OrthographicCamera Definition
class OrthographicCamera : public ProjectiveCamera {
  public:
    // OrthographicCamera Public Methods
    OrthographicCamera(CameraBaseParameters baseParameters, Bounds2f screenWindow,
                       Float lensRadius, Float focalDist)
        : ProjectiveCamera(baseParameters, Orthographic(0, 1), screenWindow, lensRadius,
                           focalDist) {
        // Compute differential changes in origin for orthographic camera rays
        dxCamera = cameraFromRaster(Vector3f(1, 0, 0));
        dyCamera = cameraFromRaster(Vector3f(0, 1, 0));

        // Compute minimum differentials for orthographic camera
        minDirDifferentialX = minDirDifferentialY = Vector3f(0, 0, 0);
        minPosDifferentialX = dxCamera;
        minPosDifferentialY = dyCamera;
    }

    PBRT_CPU_GPU
    pstd::optional<CameraRay> GenerateRay(CameraSample sample,
                                          SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        CameraSample sample, SampledWavelengths &lambda) const;

    static OrthographicCamera *Create(const ParameterDictionary &parameters,
                                      const CameraTransform &cameraTransform, Film film,
                                      Medium medium, const FileLoc *loc,
                                      Allocator alloc = {});

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const {
        LOG_FATAL("We() unimplemented for OrthographicCamera");
        return {};
    }

    PBRT_CPU_GPU
    void PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("PDF_We() unimplemented for OrthographicCamera");
    }

    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> SampleWi(const Interaction &ref, Point2f u,
                                            SampledWavelengths &lambda) const {
        LOG_FATAL("SampleWi() unimplemented for OrthographicCamera");
        return {};
    }

    std::string ToString() const;

  private:
    // OrthographicCamera Private Members
    Vector3f dxCamera, dyCamera;
};

// PerspectiveCamera Definition
class PerspectiveCamera : public ProjectiveCamera {
  public:
    // distortionpolynomials definition; --added by zhenyi
    struct distortionPolynomials {
        std::vector<float> wavelength;
        std::vector<std::vector<float>> polynomials;
    };
    // PerspectiveCamera Public Methods
    PerspectiveCamera(CameraBaseParameters baseParameters, Float fov,
                      Bounds2f screenWindow, Float lensRadius, Float focalDistance,
                      PerspectiveCamera::distortionPolynomials distortionPolynomials)
        : ProjectiveCamera(baseParameters, Perspective(fov, 1e-2f, 1000.f), screenWindow,
                           lensRadius, focalDistance) {
        // Compute differential changes in origin for perspective camera rays
        dxCamera =
            cameraFromRaster(Point3f(1, 0, 0)) - cameraFromRaster(Point3f(0, 0, 0));
        dyCamera =
            cameraFromRaster(Point3f(0, 1, 0)) - cameraFromRaster(Point3f(0, 0, 0));

        // Compute _cosTotalWidth_ for perspective camera
        Point2f radius = Point2f(film.GetFilter().Radius());
        Point3f pCorner(-radius.x, -radius.y, 0.f);
        Vector3f wCornerCamera = Normalize(Vector3f(cameraFromRaster(pCorner)));
        cosTotalWidth = wCornerCamera.z;
        DCHECK_LT(.9999 * cosTotalWidth, std::cos(Radians(fov / 2)));

        // Compute image plane area at $z=1$ for _PerspectiveCamera_
        Point2i res = film.FullResolution();
        Point3f pMin = cameraFromRaster(Point3f(0, 0, 0));
        Point3f pMax = cameraFromRaster(Point3f(res.x, res.y, 0));
        pMin /= pMin.z;
        pMax /= pMax.z;
        A = std::abs((pMax.x - pMin.x) * (pMax.y - pMin.y));
        distPolys = distortionPolynomials;
        // Compute minimum differentials for _PerspectiveCamera_
        FindMinimumDifferentials(this);
    }

    PerspectiveCamera() = default;

    static PerspectiveCamera *Create(const ParameterDictionary &parameters,
                                     const CameraTransform &cameraTransform, Film film,
                                     Medium medium, const FileLoc *loc,
                                     Allocator alloc = {});

    PBRT_CPU_GPU
    pstd::optional<CameraRay> GenerateRay(CameraSample sample,
                                          SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        CameraSample sample, SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const;
    PBRT_CPU_GPU
    void PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const;
    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> SampleWi(const Interaction &ref, Point2f u,
                                            SampledWavelengths &lambda) const;

    std::string ToString() const;

  private:
    // PerspectiveCamera Private Members
    Vector3f dxCamera, dyCamera;
    Float cosTotalWidth;
    Float A;
    PerspectiveCamera::distortionPolynomials distPolys;
};

// SphericalCamera Definition
class SphericalCamera : public CameraBase {
  public:
    // SphericalCamera::Mapping Definition
    enum Mapping { EquiRectangular, EqualArea };

    // SphericalCamera Public Methods
    SphericalCamera(CameraBaseParameters baseParameters, Mapping mapping)
        : CameraBase(baseParameters), mapping(mapping) {
        // Compute minimum differentials for _SphericalCamera_
        FindMinimumDifferentials(this);
    }

    static SphericalCamera *Create(const ParameterDictionary &parameters,
                                   const CameraTransform &cameraTransform, Film film,
                                   Medium medium, const FileLoc *loc,
                                   Allocator alloc = {});

    PBRT_CPU_GPU
    pstd::optional<CameraRay> GenerateRay(CameraSample sample,
                                          SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        CameraSample sample, SampledWavelengths &lambda) const {
        return CameraBase::GenerateRayDifferential(this, sample, lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const {
        LOG_FATAL("We() unimplemented for SphericalCamera");
        return {};
    }

    PBRT_CPU_GPU
    void PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("PDF_We() unimplemented for SphericalCamera");
    }

    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> SampleWi(const Interaction &ref, Point2f u,
                                            SampledWavelengths &lambda) const {
        LOG_FATAL("SampleWi() unimplemented for SphericalCamera");
        return {};
    }

    std::string ToString() const;

  private:
    // SphericalCamera Private Members
    Mapping mapping;
};

// ExitPupilSample Definition
struct ExitPupilSample {
    Point3f pPupil;
    Float pdf;
};

// RealisticCamera Definition
class RealisticCamera : public CameraBase {
  public:
    // RealisticCamera Public Methods
    RealisticCamera(CameraBaseParameters baseParameters,
                    std::vector<Float> &lensParameters, Float focusDistance,
                    Float apertureDiameter, Image apertureImage, Allocator alloc);

    static RealisticCamera *Create(const ParameterDictionary &parameters,
                                   const CameraTransform &cameraTransform, Film film,
                                   Medium medium, const FileLoc *loc,
                                   Allocator alloc = {});

    PBRT_CPU_GPU
    pstd::optional<CameraRay> GenerateRay(CameraSample sample,
                                          SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        CameraSample sample, SampledWavelengths &lambda) const {
        return CameraBase::GenerateRayDifferential(this, sample, lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const {
        LOG_FATAL("We() unimplemented for RealisticCamera");
        return {};
    }

    PBRT_CPU_GPU
    void PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("PDF_We() unimplemented for RealisticCamera");
    }

    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> SampleWi(const Interaction &ref, Point2f u,
                                            SampledWavelengths &lambda) const {
        LOG_FATAL("SampleWi() unimplemented for RealisticCamera");
        return {};
    }

    std::string ToString() const;

  private:
    // RealisticCamera Private Declarations
    struct LensElementInterface {
        Float curvatureRadius;
        Float thickness;
        Float eta;
        Float apertureRadius;
        std::string ToString() const;
    };

    // RealisticCamera Private Methods
    PBRT_CPU_GPU
    Float LensRearZ() const { return elementInterfaces.back().thickness; }

    PBRT_CPU_GPU
    Float LensFrontZ() const {
        Float zSum = 0;
        for (const LensElementInterface &element : elementInterfaces)
            zSum += element.thickness;
        return zSum;
    }

    PBRT_CPU_GPU
    Float RearElementRadius() const { return elementInterfaces.back().apertureRadius; }

    PBRT_CPU_GPU
    Float TraceLensesFromFilm(const Ray &rCamera, Ray *rOut) const;

    PBRT_CPU_GPU
    static bool IntersectSphericalElement(Float radius, Float zCenter, const Ray &ray,
                                          Float *t, Normal3f *n) {
        // Compute _t0_ and _t1_ for ray--element intersection
        Point3f o = ray.o - Vector3f(0, 0, zCenter);
        Float A = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
        Float B = 2 * (ray.d.x * o.x + ray.d.y * o.y + ray.d.z * o.z);
        Float C = o.x * o.x + o.y * o.y + o.z * o.z - radius * radius;
        Float t0, t1;
        if (!Quadratic(A, B, C, &t0, &t1))
            return false;

        // Select intersection $t$ based on ray direction and element curvature
        bool useCloserT = (ray.d.z > 0) ^ (radius < 0);
        *t = useCloserT ? std::min(t0, t1) : std::max(t0, t1);
        if (*t < 0)
            return false;

        // Compute surface normal of element at ray intersection point
        *n = Normal3f(Vector3f(o + *t * ray.d));
        *n = FaceForward(Normalize(*n), -ray.d);

        return true;
    }

    PBRT_CPU_GPU
    Float TraceLensesFromScene(const Ray &rCamera, Ray *rOut) const;

    void DrawLensSystem() const;
    void DrawRayPathFromFilm(const Ray &r, bool arrow, bool toOpticalIntercept) const;
    void DrawRayPathFromScene(const Ray &r, bool arrow, bool toOpticalIntercept) const;

    static void ComputeCardinalPoints(Ray rIn, Ray rOut, Float *p, Float *f);
    void ComputeThickLensApproximation(Float pz[2], Float f[2]) const;
    Float FocusThickLens(Float focusDistance);
    Bounds2f BoundExitPupil(Float filmX0, Float filmX1) const;
    void RenderExitPupil(Float sx, Float sy, const char *filename) const;

    PBRT_CPU_GPU
    pstd::optional<ExitPupilSample> SampleExitPupil(Point2f pFilm, Point2f uLens) const;

    void TestExitPupilBounds() const;

    // RealisticCamera Private Members
    Bounds2f physicalExtent;
    pstd::vector<LensElementInterface> elementInterfaces;
    Image apertureImage;
    pstd::vector<Bounds2f> exitPupilBounds;
};

// HumanEyeCamera Definition
class HumanEyeCamera : public CameraBase {
  public:
    // HumanEyeCamera Public Declarations
    struct LensElementEye {
        Float radiusX;
        Float radiusY;
        Float thickness;
        Float mediumIndex;
        Float semiDiameter;
        Float conicConstantX;
        Float conicConstantY;
    };
    pstd::vector<LensElementEye> lensEls;

    // HumanEyeCamera Public Methods
    HumanEyeCamera(CameraBaseParameters baseParameters,
                   pstd::vector<LensElementEye> &eyeInterfacesData, Float pupilDiameter,
                   Float retinaDistance, Float retinaRadius, Float retinaSemiDiam,
                   pstd::vector<Spectrum> iorSpectra,
                   pstd::vector<Point3f> surfaceLookupTable,
                     bool diffractionEnabled,
                   Allocator alloc);
    static HumanEyeCamera *Create(const ParameterDictionary &parameters,
                                  const CameraTransform &cameraTransform, Film film,
                                  Medium medium, const FileLoc *loc,
                                  Allocator alloc = {});
    PBRT_CPU_GPU
    pstd::optional<CameraRay> GenerateRay(CameraSample sample,
                                          SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        CameraSample sample, SampledWavelengths &lambda) const {
        return CameraBase::GenerateRayDifferential(this, sample, lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const {
        LOG_FATAL("We() unimplemented for HumanEyeCamera");
        return {};
    }

    PBRT_CPU_GPU
    void PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("PDF_We() unimplemented for HumanEyeCamera");
    }

    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> SampleWi(const Interaction &ref, Point2f u,
                                            SampledWavelengths &lambda) const {
        LOG_FATAL("SampleWi() unimplemented for HumanEyeCamera");
        return {};
    }

    PBRT_CPU_GPU
    bool useLookupTable() const {
         return lookupTable.size()>0;
         
    }

    PBRT_CPU_GPU
    // TG: This function implements the legacy realisticEye code. 
    // A point on the film is mapped to a spherical surface.
    Point3f mapToSphere(const Point2f pFilm) const {
        // Determine the size of the sensor in real world units (i.e. convert from pixels
        // to millimeters).
        //printf("Pfilm %f,%f:",pFilm.x,pFilm.y);
        Point2i filmRes = film.FullResolution();

        // To calculate the "film diagonal", we use the retina semi-diameter. The film
        // diagonal is the diagonal of the rectangular image rendered out by PBRT, in real
        // units. Since we restrict samples to a circular image, we can calculate the film
        // diagonal to be the same as a square that circumscribes the circular image.
        Float aspectRatio = (Float)filmRes.x / (Float)filmRes.y;
        Float width = retinaDiag / std::sqrt((1.f + 1.f / (aspectRatio * aspectRatio)));
        Float height = width / aspectRatio;

        Point3f startingPoint;

        startingPoint.x = -((pFilm.x) - filmRes.x / 2.f - .25) / (filmRes.y / 2.f);
        startingPoint.y = ((pFilm.y) - filmRes.y / 2.f - .25) / (filmRes.y / 2.f);

        // Convert starting point units to millimeters
        startingPoint.x = startingPoint.x * width / 2.f;
        startingPoint.y = startingPoint.y * height / 2.f;
        startingPoint.z = -retinaDistance;

        // Project sampled points onto the curved retina
        if (retinaRadius != 0) {
            // Right now the code only lets you curve the sensor toward the scene and not
            // the other way around. See diagram:
            /*

                The distance between the zero point on the z-axis (i.e. the lens element
            closest to the sensor) and the dotted line will be equal to the
            "retinaDistance." The retina curvature is defined by the "retinaRadius" and
            it's height in the y and x direction is defined by the "retinaSemiDiam."


                                        :
                                        |  :
                                        | :
                            | | |         |:
            scene <------ | | | <----   |:
                            | | |         |:
                        Lens System      | :
                                        |  :
                                        :
                                    retina
            <---- +z

                */

            // Limit sample points to a circle within the retina semi-diameter
            if ((startingPoint.x * startingPoint.x + startingPoint.y * startingPoint.y) >
                (retinaSemiDiam * retinaSemiDiam)) {
                return {};
            }

            // Calculate the distance of a disc that fits inside the curvature of the
            // retina.
            Float zDiscDistance = -1 * std::sqrt(retinaRadius * retinaRadius -
                                                 retinaSemiDiam * retinaSemiDiam);

            // If we are within this radius, project each point out onto a sphere. There
            // may be some issues here with even sampling, since this is a direct
            // projection...
            Float el = atan(startingPoint.x / zDiscDistance);
            Float az = atan(startingPoint.y / zDiscDistance);

            // Convert spherical coordinates to cartesian coordinates (note: we switch up
            // the x,y,z axis to match our conventions)
            Float xc, yc, zc, rcoselev;
            xc = -1 * retinaRadius * sin(el);  // TODO: Confirm this flip?
            rcoselev = retinaRadius * cos(el);
            zc = -1 * (rcoselev * cos(az));  // The -1 is to account for the curvature
                                             // described above in the diagram
            yc = -1 * rcoselev * sin(az);    // TODO: Confirm this flip?

            zc = zc + -1 * retinaDistance +
                 retinaRadius;  // Move the z coordinate out to correct retina distance

            startingPoint = Point3f(xc, yc, zc);
        }

        
        return startingPoint;
    }

    // TG: Casting a Float to integer requires another function on GPU and CPU
    // note that Float is a template class which has a different meaning on CPU and GPU.
    // On GPU Float is a double.
    // Rounding Down
    PBRT_CPU_GPU inline int Float2int_rd(Float arg) const {
#ifdef PBRT_IS_GPU_CODE

        return ::__double2int_rd(arg)
#else
        return (int)(std::floor(arg));
#endif
    }

    PBRT_CPU_GPU
    // TG: This function takes a point on the film, finds its corresponding index in the 2D lookup table and 
    // simply returns the point given in the lookup table. The actual position of the film is meaningless since we map
    // it to an arbitrary point given by the lookup table. 
    // Note that of a prime number of data points are given, a rectangular grid can never represent the right number of pixels
    // It is perfectly valid to define a film that has only one row, since we use it as a datastructure rather as a physical film
    Point3f mapLookupTable(const Point2f pFilm) const {

        // We need to find the pixel index to know where to evaluate the lookupTable.
        // pFIlm actually is already the filmindex. It is a floating point number to allow for jitter within the pixel
        // But if you round it down (floor), you will get the pixel index starting at zero.
        // I implemented a function that should work on both GPU and CPU
        Point2i filmIndex=Point2i(Float2int_rd(pFilm.x),Float2int_rd(pFilm.y));
        
        // It is not known in advance whether the caller made the film a column or row vector
        // just assume that the largest index must be the right dimension
        int linearIndex = filmIndex.x;
        if(filmIndex.y>filmIndex.y){
            int linearIndex = filmIndex.y;
        }

        // DO not evaluae the lookuptable if the index is larger than its size - for whatever reason
        if((filmIndex.x < lookupTable.XSize()) && (filmIndex.y < lookupTable.YSize())){
            Point3f startingPoint = lookupTable[filmIndex];
        
        if(filmIndex.x < lookupTable.size()){
        Point3f startingPoint = lookupTable[filmIndex];


        
 



            return startingPoint;
        }else{
           // REturn empty value if index not within domain of lookupTable;
          return {};
         }
        
    }

    // std::string ToString() const; // not necessary for now --Zhenyi

  private:
    // HumanEyeCamera Private Methods
    // const bool simpleWeighting;
    // const bool noWeighting;

    // Lookup table
    pstd::vector<Point3f> lookupTable;
    Bounds2f physicalExtent;


    // Lens information

    Float effectiveFocalLength;

    // Specific parameters for the human eye
    Float pupilDiameter;
    Float retinaDistance;
    Float retinaRadius;
    Float retinaSemiDiam;
    Float retinaDiag;      // This will take the place of "film->diag"
    Float frontThickness;  // The distance from the back of the lens to the front of the
                           // eye.
    pstd::vector<Spectrum> iorSpectra;

    // Flags for conventions
    bool diffractionEnabled;
    Float lensScaling;

    // Private methods for tracing through lens
    // PBRT_CPU_GPU
    bool IntersectLensElAspheric(const Ray &r, Float *tHit, LensElementEye currElement,
                                 Float zShift, Vector3f *n) const;

    // PBRT_CPU_GPU
    void applySnellsLaw(Float n1, Float n2, Float lensRadius, Vector3f &normalVec,
                        Ray *ray) const;

    // PBRT_CPU_GPU
    Float lookUpIOR(int mediumIndex, const Ray &ray) const;

    void diffractHURB(Point3f intersect, Float apertureRadius, const Float wavelength,
                      const Vector3f oldDirection, Vector3f *newDirection) const;

    // Handy method to explicity solve for the z(x,y) at a given point (x,y), for the
    // biconic SAG
    Float BiconicZ(Float x, Float y, LensElementEye currElement) const;

    // ZLY: To check whether this is needed or not
    // // GSL seed(?) for random number generation
    gsl_rng *r;
};

// OmniCamera Definition
class OmniCamera : public CameraBase {
  public:
    // OmniCamera Public Declarations
    struct LensElementInterface {
        LensElementInterface() {}
        LensElementInterface(Float cRadius, Float aRadius, Float thickness, Float ior,
                             SampledSpectrum iorspectral)
            : curvatureRadius(cRadius, cRadius),
              apertureRadius(aRadius, aRadius),
              conicConstant((Float)0.0, (Float)0.0),
              transform(Transform()),
              thickness(thickness),
              eta(ior),
              etaspectral(iorspectral) {}
        Vector2f curvatureRadius;
        Vector2f apertureRadius;
        Vector2f conicConstant;
        Transform transform;
        Float thickness;
        Float eta;
        SampledSpectrum etaspectral;
        std::vector<Float> asphericCoefficients;
        Float zMin;
        Float zMax;
        std::string ToString() const;
    };

    struct MicrolensData {
        pstd::vector<LensElementInterface> elementInterfaces;
        float offsetFromSensor;
        pstd::vector<Vector2f> offsets;
        Vector2i dimensions;
        // Non-physical term
        int simulationRadius;
    };

    // TG: Variables not in struct as experiment for GPU compatible code
    pstd::vector<LensElementInterface> microlensElementInterfaces;
    float microlensOffsetFromSensor;
    pstd::vector<Vector2f> microlensOffsets;
    Vector2i microlensDimensions;
    // Non-physical term
    int microlensSimulationRadius;

    // OmniCamera Public Methods
    OmniCamera(CameraBaseParameters baseParameters,
               pstd::vector<OmniCamera::LensElementInterface> &lensInterfaceData,
               Float focusDistance, Float filmDistance, bool caFlag,
               bool diffractionEnabled,
               pstd::vector<OmniCamera::LensElementInterface> &microlensData,
               Vector2i microlensDims, pstd::vector<Vector2f> &microlensOffsets,
               Float microlensSensorOffset, int microlensSimulationRadius,
               Float apertureDiameter, Image apertureImage, Allocator alloc);

    static OmniCamera *Create(const ParameterDictionary &parameters,
                              const CameraTransform &cameraTransform, Film film,
                              Medium medium, const FileLoc *loc, Allocator alloc = {});

    PBRT_CPU_GPU
    pstd::optional<CameraRay> GenerateRay(CameraSample sample,
                                          SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        CameraSample sample, SampledWavelengths &lambda) const {
        return CameraBase::GenerateRayDifferential(this, sample, lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const {
        LOG_FATAL("We() unimplemented for OmniCamera");
        return {};
    }

    PBRT_CPU_GPU
    void PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("PDF_We() unimplemented for OmniCamera");
    }

    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> SampleWi(const Interaction &ref, Point2f u,
                                            SampledWavelengths &lambda) const {
        LOG_FATAL("SampleWi() unimplemented for OmniCamera");
        return {};
    }

    std::string ToString() const;

  private:
    // OmniCamera Private Declarations

    enum IntersectResult { MISS, CULLED_BY_APERTURE, HIT };

    // OmniCamera Private Methods
    struct MicrolensElement {
        Point2f center;
        ConvexQuadf centeredBounds;
        Point2f corner1;
        Point2f corner2;
        Point2f corner3;
        Point2f corner4;
        Point2f index;
        // Transform ComputeCameraToMicrolens() const;
    };

    PBRT_CPU_GPU
    Float LensRearZ() const { return elementInterfaces.back().thickness; }

    PBRT_CPU_GPU
    Float LensFrontZ() const {
        Float zSum = 0;
        for (const LensElementInterface &element : elementInterfaces)
            zSum += element.thickness;
        return zSum;
    }

    PBRT_CPU_GPU
    Float RearElementRadius() const { return elementInterfaces.back().apertureRadius.x; }

    PBRT_CPU_GPU
    Float TToBackLens(const Ray &ray,
                      const pstd::vector<LensElementInterface> &interfaces,
                      const Transform LensFromCamera, const ConvexQuadf &bounds) const;

    PBRT_CPU_GPU
    Float TraceLensesFromFilm(const Ray &rCamera,
                              const pstd::vector<LensElementInterface> &interfaces,
                              Ray *rOut, const Transform LensFromCamera,
                              const ConvexQuadf &bounds) const;
    // Float TraceLensesFromFilm(const Ray &rCamera, Ray *rOut) const;

    PBRT_CPU_GPU
    static bool IntersectSphericalElement(Float radius, Float zCenter, const Ray &ray,
                                          Float *t, Normal3f *n) {
        // Compute _t0_ and _t1_ for ray--element intersection
        Point3f o = ray.o - Vector3f(0, 0, zCenter);
        Float A = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
        Float B = 2 * (ray.d.x * o.x + ray.d.y * o.y + ray.d.z * o.z);
        Float C = o.x * o.x + o.y * o.y + o.z * o.z - radius * radius;
        Float t0, t1;
        if (!Quadratic(A, B, C, &t0, &t1))
            return false;

        // Select intersection $t$ based on ray direction and element curvature
        bool useCloserT = (ray.d.z > 0) ^ (radius < 0);
        *t = useCloserT ? std::min(t0, t1) : std::max(t0, t1);
        if (*t < 0)
            return false;

        // Compute surface normal of element at ray intersection point
        *n = Normal3f(Vector3f(o + *t * ray.d));
        *n = FaceForward(Normalize(*n), -ray.d);

        return true;
    }

    // PBRT_CPU_GPU
    void diffractHURB(Ray &rLens, const LensElementInterface &element,
                      const Float t) const;

    // GSL seed(?) for random number generation
    gsl_rng *r;

    PBRT_CPU_GPU
    Float TraceLensesFromScene(const Ray &rCamera, Ray *rOut) const;

    void DrawLensSystem() const;
    void DrawRayPathFromFilm(const Ray &r, bool arrow, bool toOpticalIntercept) const;
    void DrawRayPathFromScene(const Ray &r, bool arrow, bool toOpticalIntercept) const;

    static void ComputeCardinalPoints(Ray rIn, Ray rOut, Float *p, Float *f);
    void ComputeThickLensApproximation(Float pz[2], Float f[2]) const;
    Float FocusBinarySearch(Float focusDistance);
    Float FocusThickLens(Float focusDistance);
    Float FocusDistance(Float filmDist);
    Bounds2f BoundExitPupil(Float filmX0, Float filmX1) const;
    void RenderExitPupil(Float sx, Float sy, const char *filename) const;

    PBRT_CPU_GPU
    IntersectResult TraceElement(const LensElementInterface &element, const Ray &rLens,
                                 const Float &elementZ, Float &t, Normal3f &n,
                                 bool &isStop, const ConvexQuadf &bounds) const;

    PBRT_CPU_GPU
    pstd::optional<ExitPupilSample> SampleExitPupil(Point2f pFilm, Point2f uLens) const;

    void TestExitPupilBounds() const;

    // static Vector2f mapMul(Vector2f v0, Vector2f v1) {
    // return Vector2f(v0.x*v1.x, v0.y*v1.y);
    // };
    // static Vector2f mapDiv(Vector2f v0, Vector2f v1) {
    //     return Vector2f(v0.x/v1.x, v0.y/v1.y);
    // }
    // static Vector2f mapDiv(Vector2f v0, Vector2i v1) {
    //     return Vector2f(v0.x / (pbrt::Float)v1.x, v0.y / (pbrt::Float)v1.y);
    // }
    // static Point2f mapDiv(Point2f v0, Vector2f v1) {
    //     return Point2f(v0.x / v1.x, v0.y / v1.y);
    // }

    PBRT_CPU_GPU
    Point2f MicrolensIndex(const Point2f p) const;

    PBRT_CPU_GPU
    Point2f MicrolensCenterFromIndex(const Point2f idx) const;

    PBRT_CPU_GPU
    MicrolensElement MicrolensElementFromIndex(const Point2f idx) const;

    PBRT_CPU_GPU
    MicrolensElement ComputeMicrolensElement(const Ray filmRay) const;

    PBRT_CPU_GPU
    Float TraceFullLensSystemFromFilm(const Ray &rIn, Ray *rOut) const;

    PBRT_CPU_GPU
    pstd::optional<ExitPupilSample> SampleMicrolensPupil(Point2f pFilm,
                                                         Point2f uLens) const;

    // bool HasMicrolens() const;

    // OmniCamera Private Members
    Bounds2f physicalExtent;
    pstd::vector<LensElementInterface> elementInterfaces;
    Image apertureImage;
    pstd::vector<Bounds2f> exitPupilBounds;
    const bool caFlag;
    const bool diffractionEnabled;
    MicrolensData microlens;
};

inline pstd::optional<CameraRay> Camera::GenerateRay(CameraSample sample,
                                                     SampledWavelengths &lambda) const {
    auto generate = [&](auto ptr) { return ptr->GenerateRay(sample, lambda); };
    return Dispatch(generate);
}

inline Film Camera::GetFilm() const {
    auto getfilm = [&](auto ptr) { return ptr->GetFilm(); };
    return Dispatch(getfilm);
}

inline Float Camera::SampleTime(Float u) const {
    auto sample = [&](auto ptr) { return ptr->SampleTime(u); };
    return Dispatch(sample);
}

inline const CameraTransform &Camera::GetCameraTransform() const {
    auto gtc = [&](auto ptr) -> const CameraTransform & {
        return ptr->GetCameraTransform();
    };
    return Dispatch(gtc);
}

inline void Camera::Approximate_dp_dxy(Point3f p, Normal3f n, Float time,
                                       int samplesPerPixel, Vector3f *dpdx,
                                       Vector3f *dpdy) const {
    if constexpr (AllInheritFrom<CameraBase>(Camera::Types())) {
        return ((const CameraBase *)ptr())
            ->Approximate_dp_dxy(p, n, time, samplesPerPixel, dpdx, dpdy);
    } else {
        auto approx = [&](auto ptr) {
            return ptr->Approximate_dp_dxy(p, n, time, samplesPerPixel, dpdx, dpdy);
        };
        return Dispatch(approx);
    }
}

}  // namespace pbrt

#endif  // PBRT_CAMERAS_H
