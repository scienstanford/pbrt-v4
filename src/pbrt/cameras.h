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

#include <pbrt/rtf/passnopass.h>

#include <algorithm>
#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <gsl/gsl_randist.h>

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


class LightfieldCameraBase : public CameraBase{
public:
   
    //LightfieldCameraBase() = default;
    LightfieldCameraBase(CameraBaseParameters p);

 
    PBRT_CPU_GPU
    pstd::optional<std::pair<CameraRay,CameraRay>> GenerateRayIO(CameraSample sample,
                                         SampledWavelengths &lambda) const {
                                             std::cout << "GenerateRayIO Virtual : This should not run" <<"\n"; 
                                             return{};
                                             };

    PBRT_CPU_GPU
    //LightfieldFilmWrapper* GetLightfieldFilm() const { return (LightfieldFilmWrapper*)(&film); }

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
                      Bounds2f screenWindow, Float lensRadius, 
                      Float focalDistance, PerspectiveCamera::distortionPolynomials distortionPolynomials)
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

// OmniCamera Definition
class OmniCamera : public LightfieldCameraBase {
  public:
    // OmniCamera Public Declarations
    struct LensElementInterface {
        LensElementInterface() {}
        LensElementInterface(Float cRadius, Float aRadius,
            Float thickness, Float ior, SampledSpectrum iorspectral) :
            curvatureRadius(cRadius,cRadius),
            apertureRadius(aRadius ,aRadius),
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
        pstd::vector<Float> offsets;
        Vector2i dimensions;
        // Non-physical term
        int simulationRadius;
    };

    // OmniCamera Public Methods
    OmniCamera(CameraBaseParameters baseParameters,
                    pstd::vector<OmniCamera::LensElementInterface> &lensInterfaceData,
                    Float focusDistance, Float filmDistance,
                    bool caFlag, bool diffractionEnabled,
                    pstd::vector<OmniCamera::LensElementInterface>& microlensData,
                    Vector2i microlensDims, pstd::vector<Float> & microlensOffsets, Float microlensSensorOffset,
                    int microlensSimulationRadius,
                    Float apertureDiameter, Image apertureImage, Allocator alloc);

    static OmniCamera *Create(const ParameterDictionary &parameters,
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
    pstd::optional<std::pair<CameraRay,CameraRay>> GenerateRayIO(CameraSample sample,
                                          SampledWavelengths &lambda) const;

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

    enum IntersectResult {MISS,CULLED_BY_APERTURE,HIT};

    // OmniCamera Private Methods
    struct MicrolensElement {
        Point2f center;
        ConvexQuadf centeredBounds;
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
    Float TToBackLens(const Ray &ray, const pstd::vector<LensElementInterface>& interfaces,
        const Transform LensFromCamera, const ConvexQuadf& bounds) const;    
   
    PBRT_CPU_GPU
    Float TraceLensesFromFilm(const Ray &rCamera, const pstd::vector<LensElementInterface>& interfaces, Ray *rOut,
        const Transform LensFromCamera, const ConvexQuadf& bounds) const;
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
    void diffractHURB(Ray &rLens, const LensElementInterface &element, const Float t) const;

      // GSL seed(?) for random number generation
    gsl_rng * r;

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
    IntersectResult TraceElement(const LensElementInterface &element, const Ray& rLens, const Float& elementZ,
         Float& t, Normal3f& n, bool& isStop, const ConvexQuadf& bounds) const;

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
    Float TraceFullLensSystemFromFilm(const Ray & rIn, Ray * rOut) const;

    PBRT_CPU_GPU
    pstd::optional<ExitPupilSample> SampleMicrolensPupil(Point2f pFilm, Point2f uLens) const;


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

// RTFCamera Definition
class RTFCamera : public LightfieldCameraBase {
  public:
    // RTFCamera Public Declarations
    struct LensPolynomialTerm {
        LensPolynomialTerm() {}
        LensPolynomialTerm(std::string n, pstd::vector<Float> tr,
                        pstd::vector<Float> tu, pstd::vector<Float> tv,
                        pstd::vector<Float> coeff) :
                        name(n), termr(tr), termu(tu),
                        termv(tv), coeff(coeff) {}
        std::string name;
        pstd::vector<Float> termr;
        pstd::vector<Float> termu;
        pstd::vector<Float> termv;
        pstd::vector<Float> coeff;
    };

    struct RTFPolynomialOutputs {
            RTFPolynomialOutputs() {}
            RTFPolynomialOutputs(pstd::vector<LensPolynomialTerm> position,pstd::vector<LensPolynomialTerm> direction ): position(position), direction(direction){}
            pstd::vector<LensPolynomialTerm> position;
            pstd::vector<LensPolynomialTerm> direction;
    };

    struct RTFVignettingTerms {
        RTFVignettingTerms() {}
        RTFVignettingTerms(Float circlePlaneZ,int exitpupilIndex, pstd::vector<Float> circleRadii,  pstd::vector<Float> circleSensitivities):
        circlePlaneZ(circlePlaneZ), exitpupilIndex(exitpupilIndex), circleRadii(circleRadii),circleSensitivities(circleSensitivities) {}
        Float circlePlaneZ;
        int exitpupilIndex; // Index that indicates main exit pupil
        pstd::vector<Float> circleRadii;
        pstd::vector<Float> circleSensitivities;

        // To accomidate nonlinear transformation of the circle corresponding to diaphragm
        // Both vectors represent polynomial coefficients in ascending degree
        // [a0 a1 a2 a3 a4 ...]  -->  a0 + a1*x + a2*x^2 +....

        // RADIUS (unit independent polynomial)
        // A polynomial that, when multiplied with the on-axis radius, gives the off-axis radius
        /// newradius = radius_onaxis*poly(offaxis_distance)
        // By construction circleRadiusPoly[0]=1
        pstd::vector<Float> circleRadiusPoly;

        // SENSITIVITY (JSON file:unit for for milimeters, converted to coefficients in meters when loaded in)
        // A polynomial that, when multiplied with the off axi distance, gives the off-axis displacement of the circle
        /// offset = poly(offaxis_distance)*offaxis_distance
        // By construction circleRadiusPoly[0]=0
        pstd::vector<Float> circleSensitivityPoly;
    };

    // RTFCamera Public Methods
    RTFCamera(CameraBaseParameters baseParameters,
                std::string bbmode,
                Float filmDistance, bool caFlag, Float apertureDiameter,
                Float planeOffsetInput, Float planeOffsetOutput, Float lensThickness,
                pstd::vector<pstd::vector< RTFCamera::LensPolynomialTerm>> polynomialMaps,
                pstd::vector<std::shared_ptr<PassNoPass>> passNoPassPerWavelength,
                pstd::vector<Float> polyWavelengths_nm,
                Allocator alloc);

    static RTFCamera *Create(const ParameterDictionary &parameters,
                                   const CameraTransform &cameraTransform, Film film,
                                   Medium medium, const FileLoc *loc,
                                   Allocator alloc = {});

    // PBRT_CPU_GPU
    pstd::optional<CameraRay> GenerateRay(CameraSample sample,
                                          SampledWavelengths &lambda) const;
    

    PBRT_CPU_GPU
    pstd::optional<std::pair<CameraRay,CameraRay>> GenerateRayIO(CameraSample sample,
                                          SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        CameraSample sample, SampledWavelengths &lambda) const {
        return CameraBase::GenerateRayDifferential(this, sample, lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const {
        LOG_FATAL("We() unimplemented for RTFCamera");
        return {};
    }

    PBRT_CPU_GPU
    void PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("PDF_We() unimplemented for RTFCamera");
    }

    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> SampleWi(const Interaction &ref, Point2f u,
                                            SampledWavelengths &lambda) const {
        LOG_FATAL("SampleWi() unimplemented for RTFCamera");
        return {};
    }

    std::string ToString() const;

  private:
    // RTFCamera Private Declarations
    enum IntersectResult {MISS,CULLED_BY_APERTURE,HIT};
    const bool caFlag;
    const Float filmDistance;
    const Float planeOffsetInput;
    const Float planeOffsetOutput;
    const Float lensThickness;
    pstd::vector<Bounds2f> exitPupilBounds;

    // RTFCamera Private Methods
    // PBRT_CPU_GPU
    // Float LensRearZ() const { return elementInterfaces.back().thickness; }

    // PBRT_CPU_GPU
    // Float LensFrontZ() const {
    //     Float zSum = 0;
    //     for (const LensElementInterface &element : elementInterfaces)
    //         zSum += element.thickness;
    //     return zSum;
    // }

    // PBRT_CPU_GPU
    // static bool IntersectSphericalElement(Float radius, Float zCenter, const Ray &ray,
    //                                       Float *t, Normal3f *n) {
    //     // Compute _t0_ and _t1_ for ray--element intersection
    //     Point3f o = ray.o - Vector3f(0, 0, zCenter);
    //     Float A = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
    //     Float B = 2 * (ray.d.x * o.x + ray.d.y * o.y + ray.d.z * o.z);
    //     Float C = o.x * o.x + o.y * o.y + o.z * o.z - radius * radius;
    //     Float t0, t1;
    //     if (!Quadratic(A, B, C, &t0, &t1))
    //         return false;

    //     // Select intersection $t$ based on ray direction and element curvature
    //     bool useCloserT = (ray.d.z > 0) ^ (radius < 0);
    //     *t = useCloserT ? std::min(t0, t1) : std::max(t0, t1);
    //     if (*t < 0)
    //         return false;

    //     // Compute surface normal of element at ray intersection point
    //     *n = Normal3f(Vector3f(o + *t * ray.d));
    //     *n = FaceForward(Normalize(*n), -ray.d);

    //     return true;
    // }

    // PBRT_CPU_GPU
    // Float TraceLensesFromScene(const Ray &rCamera, Ray *rOut) const;


    int pupilIndex;
    Ray ApplyPolynomial(Float rho, Vector3f dir, pstd::vector< RTFCamera::LensPolynomialTerm>  &polynomialMap) const;
    inline Ray RotateRays(const Ray &thisRay, Float deg) const;
    Float PolynomialCal(Float rho, Float dx, Float dy, LensPolynomialTerm &polyTerm) const;
    inline Vector2f Pos2RadiusRotation(const Point3f pos) const;
    bool IsValidRayCircles(const Ray &rotatedRay, RTFVignettingTerms &vignetting) const;
    bool TraceLensesFromFilm(const Ray &ray, Ray *rOut,int wlIndex) const;
    inline Float distanceCirclePlaneFromFilm() const;
    // PBRT_CPU_GPU
    // pstd::optional<ExitPupilSample> SampleExitPupil(Point2f pFilm, Point2f uLens) const;
    // pstd::optional<ExitPupilSample> SampleMicrolensPupil(Point2f pFilm, Point2f uLens) const;
    Point3f SampleExitPupil(const Point2f &pFilm, const Point2f &lensSample,
                            Float *sampleBoundsArea) const;
    Point3f SampleExitPupilVignetting(const Point2f &pFilm, const Point2f &lensSample,RTFVignettingTerms &vignettingTerms) const;
    Point3f SampleMainCircle(const Point2f &pFilm, const Point2f &lensSample,RTFVignettingTerms &vignettingTerms, Float *sampleBoundsArea) const;

    pstd::vector< RTFCamera::LensPolynomialTerm> poly;

    // Wavelength dependent RTF : vectorized. Each element corresponds to a given wavelength
    pstd::vector<Float> polyWavelengths_nm; // wavelengths read from file
    pstd::vector<pstd::vector< RTFCamera::LensPolynomialTerm>> polynomialMaps; // Each element has corresponding wavelength
    pstd::vector<RTFCamera::RTFVignettingTerms> vignettingTerms;
    pstd::vector<std::shared_ptr<PassNoPass>> passNoPassPerWavelength;

    std::vector<Float> circleRadii;
    std::vector<Float> circleSensitivities;
    std::string bbmode;

    Float getPupilPosition(RTFVignettingTerms vignetting, int circleIndex) const;
    Float getPupilRadius(RTFVignettingTerms vignetting, int circleIndex) const;

    void TestExitPupilBounds() const;

    // RTFCamera Private Members
    Bounds2f physicalExtent;
    std::vector<Bounds2f> exitPupilBoundsRTF;
    Bounds2f BoundExitPupilRTF(Float filmX0, Float filmX1) const;
};


inline pstd::optional<CameraRay> Camera::GenerateRay(CameraSample sample,
                                                     SampledWavelengths &lambda) const {
    auto generate = [&](auto ptr) { return ptr->GenerateRay(sample, lambda); };
    return Dispatch(generate);
}

inline pstd::optional<std::pair<CameraRay,CameraRay>> LightfieldCamera::GenerateRayIO(CameraSample sample,
                                                     SampledWavelengths &lambda) const {
// THomas: I want this to be LightFieldCameraBase. Which implies setting generateRayIO virtual in the base class
// But then I get NAN errors my renderings which I cannot explain
// So for now i will break the polymorphism . 
// Only woskf or RTF Camera
    auto generate = [&](auto ptr) { return ((OmniCamera*)ptr)->GenerateRayIO(sample, lambda); };
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
