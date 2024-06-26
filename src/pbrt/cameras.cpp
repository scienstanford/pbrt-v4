// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/cameras.h>

#include <pbrt/base/medium.h>
#include <pbrt/bsdf.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>
#include <ext/json.hpp> 
#include <fstream> 
#include <iostream>


// RTF
#include <pbrt/rtf/passnopass.h>

#include <algorithm>
#include <math.h>
#include <cmath>
#include <gsl/gsl_roots.h> // For solving biconic surface intersections.
#include <gsl/gsl_errno.h>




using json = nlohmann::json;
namespace pbrt {

// CameraTransform Method Definitions
CameraTransform::CameraTransform(const AnimatedTransform &worldFromCamera) {
    switch (Options->renderingSpace) {
    case RenderingCoordinateSystem::Camera: {
        // Compute _worldFromRender_ for camera-space rendering
        Float tMid = (worldFromCamera.startTime + worldFromCamera.endTime) / 2;
        worldFromRender = worldFromCamera.Interpolate(tMid);
        break;
    }
    case RenderingCoordinateSystem::CameraWorld: {
        // Compute _worldFromRender_ for camera-world space rendering
        Float tMid = (worldFromCamera.startTime + worldFromCamera.endTime) / 2;
        Point3f pCamera = worldFromCamera(Point3f(0, 0, 0), tMid);
        worldFromRender = Translate(Vector3f(pCamera));
        break;
    }
    case RenderingCoordinateSystem::World: {
        // Compute _worldFromRender_ for world-space rendering
        worldFromRender = Transform();
        break;
    }
    default:
        LOG_FATAL("Unhandled rendering coordinate space");
    }
    LOG_VERBOSE("World-space position: %s", worldFromRender(Point3f(0, 0, 0)));
    // Compute _renderFromCamera_ transformation
    Transform renderFromWorld = Inverse(worldFromRender);
    Transform rfc[2] = {renderFromWorld * worldFromCamera.startTransform,
                        renderFromWorld * worldFromCamera.endTransform};
    renderFromCamera = AnimatedTransform(rfc[0], worldFromCamera.startTime, rfc[1],
                                         worldFromCamera.endTime);
}

std::string CameraTransform::ToString() const {
    return StringPrintf("[ CameraTransform renderFromCamera: %s worldFromRender: %s ]",
                        renderFromCamera, worldFromRender);
}

// Camera Method Definitions
pstd::optional<CameraRayDifferential> Camera::GenerateRayDifferential(
    CameraSample sample, SampledWavelengths &lambda) const {
    auto gen = [&](auto ptr) { return ptr->GenerateRayDifferential(sample, lambda); };
    return Dispatch(gen);
}

SampledSpectrum Camera::We(const Ray &ray, SampledWavelengths &lambda,
                           Point2f *pRaster2) const {
    auto we = [&](auto ptr) { return ptr->We(ray, lambda, pRaster2); };
    return Dispatch(we);
}

void Camera::PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    auto pdf = [&](auto ptr) { return ptr->PDF_We(ray, pdfPos, pdfDir); };
    return Dispatch(pdf);
}

pstd::optional<CameraWiSample> Camera::SampleWi(const Interaction &ref, Point2f u,
                                                SampledWavelengths &lambda) const {
    auto sample = [&](auto ptr) { return ptr->SampleWi(ref, u, lambda); };
    return Dispatch(sample);
}

void Camera::InitMetadata(ImageMetadata *metadata) const {
    auto init = [&](auto ptr) { return ptr->InitMetadata(metadata); };
    return DispatchCPU(init);
}

std::string Camera::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

// CameraBase Method Definitions
CameraBase::CameraBase(CameraBaseParameters p)
    : cameraTransform(p.cameraTransform),
      shutterOpen(p.shutterOpen),
      shutterClose(p.shutterClose),
      film(p.film),
      medium(p.medium) {
    if (cameraTransform.CameraFromRenderHasScale())
        Warning("Scaling detected in rendering space to camera space transformation!\n"
                "The system has numerous assumptions, implicit and explicit,\n"
                "that this transform will have no scale factors in it.\n"
                "Proceed at your own risk; your image may have errors or\n"
                "the system may crash as a result of this.");
}

pstd::optional<CameraRayDifferential> CameraBase::GenerateRayDifferential(
    Camera camera, CameraSample sample, SampledWavelengths &lambda) {
    // Generate regular camera ray _cr_ for ray differential
    pstd::optional<CameraRay> cr = camera.GenerateRay(sample, lambda);
    if (!cr)
        return {};
    RayDifferential rd(cr->ray);

    // Find camera ray after shifting one pixel in the $x$ direction
    pstd::optional<CameraRay> rx;
    for (Float eps : {.05f, -.05f}) {
        CameraSample sshift = sample;
        sshift.pFilm.x += eps;
        // Try to generate ray with _sshift_ and compute $x$ differential
        if (rx = camera.GenerateRay(sshift, lambda); rx) {
            rd.rxOrigin = rd.o + (rx->ray.o - rd.o) / eps;
            rd.rxDirection = rd.d + (rx->ray.d - rd.d) / eps;
            break;
        }
    }

    // Find camera ray after shifting one pixel in the $y$ direction
    pstd::optional<CameraRay> ry;
    for (Float eps : {.05f, -.05f}) {
        CameraSample sshift = sample;
        sshift.pFilm.y += eps;
        if (ry = camera.GenerateRay(sshift, lambda); ry) {
            rd.ryOrigin = rd.o + (ry->ray.o - rd.o) / eps;
            rd.ryDirection = rd.d + (ry->ray.d - rd.d) / eps;
            break;
        }
    }

    // Return approximate ray differential and weight
    rd.hasDifferentials = rx && ry;
    return CameraRayDifferential{rd, cr->weight};
}

void CameraBase::FindMinimumDifferentials(Camera camera) {
    minPosDifferentialX = minPosDifferentialY = minDirDifferentialX =
        minDirDifferentialY = Vector3f(Infinity, Infinity, Infinity);

    CameraSample sample;
    sample.pLens = Point2f(0.5, 0.5);
    sample.time = 0.5;
    SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.5);

    int n = 512;
    for (int i = 0; i < n; ++i) {
        sample.pFilm.x = Float(i) / (n - 1) * film.FullResolution().x;
        sample.pFilm.y = Float(i) / (n - 1) * film.FullResolution().y;

        pstd::optional<CameraRayDifferential> crd =
            camera.GenerateRayDifferential(sample, lambda);
        if (!crd)
            continue;

        RayDifferential &ray = crd->ray;
        Vector3f dox = CameraFromRender(ray.rxOrigin - ray.o, ray.time);
        if (Length(dox) < Length(minPosDifferentialX))
            minPosDifferentialX = dox;
        Vector3f doy = CameraFromRender(ray.ryOrigin - ray.o, ray.time);
        if (Length(doy) < Length(minPosDifferentialY))
            minPosDifferentialY = doy;

        ray.d = Normalize(ray.d);
        ray.rxDirection = Normalize(ray.rxDirection);
        ray.ryDirection = Normalize(ray.ryDirection);

        Frame f = Frame::FromZ(ray.d);
        Vector3f df = f.ToLocal(ray.d);  // should be (0, 0, 1);
        Vector3f dxf = Normalize(f.ToLocal(ray.rxDirection));
        Vector3f dyf = Normalize(f.ToLocal(ray.ryDirection));

        if (Length(dxf - df) < Length(minDirDifferentialX))
            minDirDifferentialX = dxf - df;
        if (Length(dyf - df) < Length(minDirDifferentialY))
            minDirDifferentialY = dyf - df;
    }

    LOG_VERBOSE("Camera min pos differentials: %s, %s", minPosDifferentialX,
                minPosDifferentialY);
    LOG_VERBOSE("Camera min dir differentials: %s, %s", minDirDifferentialX,
                minDirDifferentialY);
}

void CameraBase::InitMetadata(ImageMetadata *metadata) const {
    metadata->cameraFromWorld = cameraTransform.CameraFromWorld(shutterOpen).GetMatrix();
}

std::string CameraBase::ToString() const {
    return StringPrintf("cameraTransform: %s shutterOpen: %f shutterClose: %f film: %s "
                        "medium: %s minPosDifferentialX: %s minPosDifferentialY: %s "
                        "minDirDifferentialX: %s minDirDifferentialY: %s ",
                        cameraTransform, shutterOpen, shutterClose, film,
                        medium ? medium.ToString().c_str() : "(nullptr)",
                        minPosDifferentialX, minPosDifferentialY, minDirDifferentialX,
                        minDirDifferentialY);
}

std::string CameraSample::ToString() const {
    return StringPrintf("[ CameraSample pFilm: %s pLens: %s time: %f filterWeight: %f ]",
                        pFilm, pLens, time, filterWeight);
}

// ProjectiveCamera Method Definitions
void ProjectiveCamera::InitMetadata(ImageMetadata *metadata) const {
    metadata->cameraFromWorld = cameraTransform.CameraFromWorld(shutterOpen).GetMatrix();

    // TODO: double check this
    Transform NDCFromWorld = Translate(Vector3f(0.5, 0.5, 0.5)) * Scale(0.5, 0.5, 0.5) *
                             screenFromCamera * *metadata->cameraFromWorld;
    metadata->NDCFromWorld = NDCFromWorld.GetMatrix();

    CameraBase::InitMetadata(metadata);
}

std::string ProjectiveCamera::BaseToString() const {
    return CameraBase::ToString() +
           StringPrintf("screenFromCamera: %s cameraFromRaster: %s "
                        "rasterFromScreen: %s screenFromRaster: %s "
                        "lensRadius: %f focalDistance: %f",
                        screenFromCamera, cameraFromRaster, rasterFromScreen,
                        screenFromRaster, lensRadius, focalDistance);
}

Camera Camera::Create(const std::string &name, const ParameterDictionary &parameters,
                      Medium medium, const CameraTransform &cameraTransform, Film film,
                      const FileLoc *loc, Allocator alloc) {
    Camera camera;
    if (name == "perspective")
        camera = PerspectiveCamera::Create(parameters, cameraTransform, film, medium, loc,
                                           alloc);
    else if (name == "orthographic")
        camera = OrthographicCamera::Create(parameters, cameraTransform, film, medium,
                                            loc, alloc);
    else if (name == "realistic")
        camera = RealisticCamera::Create(parameters, cameraTransform, film, medium, loc,
                                         alloc);
    else if (name == "omni")

        camera = OmniCamera::Create(parameters, cameraTransform, film, medium, loc,
                                         alloc);
    else if (name == "raytransfer")
        camera = RTFCamera::Create(parameters, cameraTransform, film, medium, loc,
                                         alloc);
    else if (name == "humaneye" || name == "realisticEye")
        camera =
            HumanEyeCamera::Create(parameters, cameraTransform, film, medium, loc, alloc);

    else if (name == "spherical")
        camera = SphericalCamera::Create(parameters, cameraTransform, film, medium, loc,
                                         alloc);
    else
        ErrorExit(loc, "%s: camera type unknown.", name);

    if (!camera)
        ErrorExit(loc, "%s: unable to create camera.", name);

    parameters.ReportUnused();
    return camera;
}

// CameraBaseParameters Method Definitions
CameraBaseParameters::CameraBaseParameters(const CameraTransform &cameraTransform,
                                           Film film, Medium medium,
                                           const ParameterDictionary &parameters,
                                           const FileLoc *loc)
    : cameraTransform(cameraTransform), film(film), medium(medium) {
    shutterOpen = parameters.GetOneFloat("shutteropen", 0.f);
    shutterClose = parameters.GetOneFloat("shutterclose", 1.f);
    if (shutterClose < shutterOpen) {
        Warning(loc, "Shutter close time %f < shutter open %f.  Swapping them.",
                shutterClose, shutterOpen);
        pstd::swap(shutterClose, shutterOpen);
    }
    // if Options->multipleFrames
    // globalFrameNumber++;
}

// OrthographicCamera Method Definitions
pstd::optional<CameraRay> OrthographicCamera::GenerateRay(
    CameraSample sample, SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);

    Ray ray(pCamera, Vector3f(0, 0, 1), SampleTime(sample.time), medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }

    return CameraRay{RenderFromCamera(ray)};
}

pstd::optional<CameraRayDifferential> OrthographicCamera::GenerateRayDifferential(
    CameraSample sample, SampledWavelengths &lambda) const {
    // Compute main orthographic viewing ray
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);

    RayDifferential ray(pCamera, Vector3f(0, 0, 1), SampleTime(sample.time), medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }

    // Compute ray differentials for _OrthographicCamera_
    if (lensRadius > 0) {
        // Compute _OrthographicCamera_ ray differentials accounting for lens
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = pCamera + dxCamera + (ft * Vector3f(0, 0, 1));
        ray.rxOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.rxDirection = Normalize(pFocus - ray.rxOrigin);

        pFocus = pCamera + dyCamera + (ft * Vector3f(0, 0, 1));
        ray.ryOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.ryDirection = Normalize(pFocus - ray.ryOrigin);

    } else {
        ray.rxOrigin = ray.o + dxCamera;
        ray.ryOrigin = ray.o + dyCamera;
        ray.rxDirection = ray.ryDirection = ray.d;
    }

    ray.hasDifferentials = true;
    return CameraRayDifferential{RenderFromCamera(ray)};
}

std::string OrthographicCamera::ToString() const {
    return StringPrintf("[ OrthographicCamera %s dxCamera: %s dyCamera: %s ]",
                        BaseToString(), dxCamera, dyCamera);
}

OrthographicCamera *OrthographicCamera::Create(const ParameterDictionary &parameters,
                                               const CameraTransform &cameraTransform,
                                               Film film, Medium medium,
                                               const FileLoc *loc, Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    Float lensradius = parameters.GetOneFloat("lensradius", 0.f);
    Float focaldistance = parameters.GetOneFloat("focaldistance", 1e6f);
    Float frame =
        parameters.GetOneFloat("frameaspectratio", Float(film.FullResolution().x) /
                                                       Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = parameters.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (Options->fullscreen) {
                Warning("\"screenwindow\" is ignored in fullscreen mode");
        } else {
            if (sw.size() == 4) {
                screen.pMin.x = sw[0];
                screen.pMax.x = sw[1];
                screen.pMin.y = sw[2];
                screen.pMax.y = sw[3];
            } else {
                Error("\"screenwindow\" should have four values");
            }
        }
    }
    return alloc.new_object<OrthographicCamera>(cameraBaseParameters, screen, lensradius,
                                                focaldistance);
}

// PerspectiveCamera Method Definitions
pstd::optional<CameraRay> PerspectiveCamera::GenerateRay(
    CameraSample sample, SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);

    Ray ray(Point3f(0, 0, 0), Normalize(Vector3f(pCamera)), SampleTime(sample.time),
            medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }

    return CameraRay{RenderFromCamera(ray)};
}

pstd::optional<CameraRayDifferential> PerspectiveCamera::GenerateRayDifferential(
    CameraSample sample, SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);

    // tmp; no wavelength index now; modify after implement spectral path
    // load polynomials and calculate distortion
    int wavelength_index = 0;
    float kc[10] = {};
    // add distortion
    if (distPolys.polynomials.size() != 0) {
        std::vector<float> kc_size = distPolys.polynomials[0];
        for (int ii = 0; ii < kc_size.size(); ii++) {
            kc[ii] = distPolys.polynomials[wavelength_index][ii];
        }
        // add distorion  --zhenyi
        Float x_d = pCamera.x / pCamera.z;
        Float y_d = pCamera.y / pCamera.z;
        Float r_d = std::sqrt(x_d * x_d + y_d * y_d);
        Float r2 = r_d * r_d;
        Float r4 = r2 * r2;
        Float correction_factor = 1;

        correction_factor = kc[2] * r2 + kc[1] * r_d + kc[0];
        correction_factor += kc[5] * r4 * r_d + kc[4] * r4 + kc[3] * r2 * r_d;
        correction_factor += kc[8] * r4 * r4 + kc[7] * r4 * r2 * r_d + kc[6] * r4 * r2;
        correction_factor += kc[9] * r4 * r4 * r_d;
        correction_factor /= r_d;
        pCamera.x *= std::abs(correction_factor);
        pCamera.y *= std::abs(correction_factor);
    }

    Vector3f dir = Normalize(Vector3f(pCamera.x, pCamera.y, pCamera.z));
    RayDifferential ray(Point3f(0, 0, 0), dir, SampleTime(sample.time), medium);
    // Modify ray for depth of field
    if (lensRadius > 0) {
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute point on plane of focus
        Float ft = focalDistance / ray.d.z;
        Point3f pFocus = ray(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = Normalize(pFocus - ray.o);
    }

    // Compute offset rays for _PerspectiveCamera_ ray differentials
    if (lensRadius > 0) {
        // Compute _PerspectiveCamera_ ray differentials accounting for lens
        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

        // Compute $x$ ray differential for _PerspectiveCamera_ with lens
        Vector3f dx = Normalize(Vector3f(pCamera + dxCamera));
        Float ft = focalDistance / dx.z;
        Point3f pFocus = Point3f(0, 0, 0) + (ft * dx);
        ray.rxOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.rxDirection = Normalize(pFocus - ray.rxOrigin);

        // Compute $y$ ray differential for _PerspectiveCamera_ with lens
        Vector3f dy = Normalize(Vector3f(pCamera + dyCamera));
        ft = focalDistance / dy.z;
        pFocus = Point3f(0, 0, 0) + (ft * dy);
        ray.ryOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.ryDirection = Normalize(pFocus - ray.ryOrigin);

    } else {
        ray.rxOrigin = ray.ryOrigin = ray.o;
        ray.rxDirection = Normalize(Vector3f(pCamera) + dxCamera);
        ray.ryDirection = Normalize(Vector3f(pCamera) + dyCamera);
    }

    ray.hasDifferentials = true;
    return CameraRayDifferential{RenderFromCamera(ray)};
}

std::string PerspectiveCamera::ToString() const {
    return StringPrintf("[ PerspectiveCamera %s dxCamera: %s dyCamera: %s A: "
                        "%f cosTotalWidth: %f ]",
                        BaseToString(), dxCamera, dyCamera, A, cosTotalWidth);
}

PerspectiveCamera *PerspectiveCamera::Create(const ParameterDictionary &parameters,
                                             const CameraTransform &cameraTransform,
                                             Film film, Medium medium, const FileLoc *loc,
                                             Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    Float lensradius = parameters.GetOneFloat("lensradius", 0.f);
    Float focaldistance = parameters.GetOneFloat("focaldistance", 1e6);
    Float frame =
        parameters.GetOneFloat("frameaspectratio", Float(film.FullResolution().x) /
                                                       Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = parameters.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (Options->fullscreen) {
                Warning("\"screenwindow\" is ignored in fullscreen mode");
        } else {
            if (sw.size() == 4) {
                screen.pMin.x = sw[0];
                screen.pMax.x = sw[1];
                screen.pMin.y = sw[2];
                screen.pMax.y = sw[3];
            } else {
                Error(loc, "\"screenwindow\" should have four values");
            }
        }
    }
    Float fov = parameters.GetOneFloat("fov", 90.);
    // read distortion json file -- zhenyi
    std::string distortionFile = parameters.GetOneString("distortionfile", "");
    PerspectiveCamera::distortionPolynomials distortionPolynomials;
    if (distortionFile != "") {
        std::ifstream i(distortionFile);
        json j;
        i >> j;
        for (auto &elem : j["kc"])
            distortionPolynomials.wavelength.push_back(elem["wavelength"]);
        for (auto &elem : j["kc"])
            distortionPolynomials.polynomials.push_back(elem["polynomials"]);
    }

    return alloc.new_object<PerspectiveCamera>(cameraBaseParameters, fov, screen,
                                               lensradius, focaldistance,
                                               distortionPolynomials);
}

SampledSpectrum PerspectiveCamera::We(const Ray &ray, SampledWavelengths &lambda,
                                      Point2f *pRasterOut) const {
    // Check if ray is forward-facing with respect to the camera
    Float cosTheta = Dot(ray.d, RenderFromCamera(Vector3f(0, 0, 1), ray.time));
    if (cosTheta <= cosTotalWidth)
        return SampledSpectrum(0.);

    // Map ray $(\p{}, \w{})$ onto the raster grid
    Point3f pFocus = ray((lensRadius > 0 ? focalDistance : 1) / cosTheta);
    Point3f pCamera = CameraFromRender(pFocus, ray.time);
    Point3f pRaster = cameraFromRaster.ApplyInverse(pCamera);

    // Return raster position if requested
    if (pRasterOut)
        *pRasterOut = Point2f(pRaster.x, pRaster.y);

    // Return zero importance for out of bounds points
    Bounds2f sampleBounds = film.SampleBounds();
    if (!Inside(Point2f(pRaster.x, pRaster.y), sampleBounds))
        return SampledSpectrum(0.);

    // Compute lens area of perspective camera
    Float lensArea = lensRadius != 0 ? (Pi * Sqr(lensRadius)) : 1;

    // Return importance for point on image plane
    return SampledSpectrum(1 / (A * lensArea * Pow<4>(cosTheta)));
}

void PerspectiveCamera::PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    // Return zero PDF values if ray direction is not front-facing
    Float cosTheta = Dot(ray.d, RenderFromCamera(Vector3f(0, 0, 1), ray.time));
    if (cosTheta <= cosTotalWidth) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    // Map ray $(\p{}, \w{})$ onto the raster grid
    Point3f pFocus = ray((lensRadius > 0 ? focalDistance : 1) / cosTheta);
    Point3f pCamera = CameraFromRender(pFocus, ray.time);
    Point3f pRaster = cameraFromRaster.ApplyInverse(pCamera);

    // Return zero probability for out of bounds points
    Bounds2f sampleBounds = film.SampleBounds();
    if (!Inside(Point2f(pRaster.x, pRaster.y), sampleBounds)) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    // Compute lens area  and return perspective camera probabilities
    Float lensArea = lensRadius != 0 ? (Pi * Sqr(lensRadius)) : 1;
    *pdfPos = 1 / lensArea;
    *pdfDir = 1 / (A * Pow<3>(cosTheta));
}

pstd::optional<CameraWiSample> PerspectiveCamera::SampleWi(
    const Interaction &ref, Point2f u, SampledWavelengths &lambda) const {
    // Uniformly sample a lens interaction _lensIntr_
    Point2f pLens = lensRadius * SampleUniformDiskConcentric(u);
    Point3f pLensRender = RenderFromCamera(Point3f(pLens.x, pLens.y, 0), ref.time);
    Normal3f n = Normal3f(RenderFromCamera(Vector3f(0, 0, 1), ref.time));
    Interaction lensIntr(pLensRender, n, ref.time, medium);

    // Find incident direction to camera _wi_ at _ref_
    Vector3f wi = lensIntr.p() - ref.p();
    Float dist = Length(wi);
    wi /= dist;

    // Compute PDF for importance arriving at _ref_
    Float lensArea = lensRadius != 0 ? (Pi * Sqr(lensRadius)) : 1;
    Float pdf = Sqr(dist) / (AbsDot(lensIntr.n, wi) * lensArea);

    // Compute importance and return _CameraWiSample_
    Point2f pRaster;
    SampledSpectrum Wi = We(lensIntr.SpawnRay(-wi), lambda, &pRaster);
    if (!Wi)
        return {};
    return CameraWiSample(Wi, wi, pdf, pRaster, ref, lensIntr);
}

// SphericalCamera Method Definitions
pstd::optional<CameraRay> SphericalCamera::GenerateRay(CameraSample sample,
                                                       SampledWavelengths &lambda) const {
    // Compute spherical camera ray direction
    Point2f uv(sample.pFilm.x / film.FullResolution().x,
               sample.pFilm.y / film.FullResolution().y);
    Vector3f dir;
    if (mapping == EquiRectangular) {
        // Compute ray direction using equirectangular mapping
        Float theta = Pi * uv[1], phi = 2 * Pi * uv[0];
        dir = SphericalDirection(std::sin(theta), std::cos(theta), phi);

    } else {
        // Compute ray direction using equal area mapping
        uv = WrapEqualAreaSquare(uv);
        dir = EqualAreaSquareToSphere(uv);
    }
    pstd::swap(dir.y, dir.z);

    Ray ray(Point3f(0, 0, 0), dir, SampleTime(sample.time), medium);
    return CameraRay{RenderFromCamera(ray)};
}

SphericalCamera *SphericalCamera::Create(const ParameterDictionary &parameters,
                                         const CameraTransform &cameraTransform,
                                         Film film, Medium medium, const FileLoc *loc,
                                         Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    Float lensradius = parameters.GetOneFloat("lensradius", 0.f);
    Float focaldistance = parameters.GetOneFloat("focaldistance", 1e30f);
    Float frame =
        parameters.GetOneFloat("frameaspectratio", Float(film.FullResolution().x) /
                                                       Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = parameters.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (Options->fullscreen) {
                Warning("\"screenwindow\" is ignored in fullscreen mode");
        } else {
            if (sw.size() == 4) {
                screen.pMin.x = sw[0];
                screen.pMax.x = sw[1];
                screen.pMin.y = sw[2];
                screen.pMax.y = sw[3];
            } else {
                Error(loc, "\"screenwindow\" should have four values");
            }
        }
    }
    (void)lensradius;     // don't need this
    (void)focaldistance;  // don't need this

    std::string m = parameters.GetOneString("mapping", "equalarea");
    Mapping mapping;
    if (m == "equalarea")
        mapping = EqualArea;
    else if (m == "equirectangular")
        mapping = EquiRectangular;
    else
        ErrorExit(loc,
                  "%s: unknown mapping for spherical camera. (Must be "
                  "\"equalarea\" or \"equirectangular\".)",
                  m);

    return alloc.new_object<SphericalCamera>(cameraBaseParameters, mapping);
}

std::string SphericalCamera::ToString() const {
    return StringPrintf("[ SphericalCamera %s mapping: %s ]", CameraBase::ToString(),
                        mapping == EquiRectangular ? "EquiRectangular" : "EqualArea");
}

// RealisticCamera Method Definitions
RealisticCamera::RealisticCamera(CameraBaseParameters baseParameters,
                                 std::vector<Float> &lensParameters, Float focusDistance,
                                 Float setApertureDiameter, Image apertureImage,
                                 Allocator alloc)
    : CameraBase(baseParameters),
      elementInterfaces(alloc),
      exitPupilBounds(alloc),
      apertureImage(std::move(apertureImage)) {
    // Compute film's physical extent
    Float aspect = (Float)film.FullResolution().y / (Float)film.FullResolution().x;
    Float diagonal = film.Diagonal();
    Float x = std::sqrt(Sqr(diagonal) / (1 + Sqr(aspect)));
    Float y = aspect * x;
    physicalExtent = Bounds2f(Point2f(-x / 2, -y / 2), Point2f(x / 2, y / 2));

    // Initialize _elementInterfaces_ for camera
    for (size_t i = 0; i < lensParameters.size(); i += 4) {
        // Extract lens element configuration from _lensParameters_
        Float curvatureRadius = lensParameters[i] / 1000;
        Float thickness = lensParameters[i + 1] / 1000;
        Float eta = lensParameters[i + 2];
        Float apertureDiameter = lensParameters[i + 3] / 1000;

        if (curvatureRadius == 0) {
            // Set aperture stop diameter
            setApertureDiameter /= 1000;
            if (setApertureDiameter > apertureDiameter)
                Warning("Aperture diameter %f is greater than maximum possible %f. "
                        "Clamping it.",
                        setApertureDiameter, apertureDiameter);
            else
                apertureDiameter = setApertureDiameter;
        }
        // Add element interface to end of _elementInterfaces_
        elementInterfaces.push_back(
            {curvatureRadius, thickness, eta, apertureDiameter / 2});
    }

    // Compute lens--film distance for given focus distance
    elementInterfaces.back().thickness = FocusThickLens(focusDistance);

    // Compute exit pupil bounds at sampled points on the film
    int nSamples = 64;
    exitPupilBounds.resize(nSamples);
    ParallelFor(0, nSamples, [&](int i) {
        Float r0 = (Float)i / nSamples * film.Diagonal() / 2;
        Float r1 = (Float)(i + 1) / nSamples * film.Diagonal() / 2;
        exitPupilBounds[i] = BoundExitPupil(r0, r1);
    });

    // Compute minimum differentials for _RealisticCamera_
    FindMinimumDifferentials(this);
}

Float RealisticCamera::TraceLensesFromFilm(const Ray &rCamera, Ray *rOut) const {
    Float elementZ = 0, weight = 1;
    // Transform _rCamera_ from camera to lens system space
    Ray rLens(Point3f(rCamera.o.x, rCamera.o.y, -rCamera.o.z),
              Vector3f(rCamera.d.x, rCamera.d.y, -rCamera.d.z), rCamera.time);

    for (int i = elementInterfaces.size() - 1; i >= 0; --i) {
        const LensElementInterface &element = elementInterfaces[i];
        // Update ray from film accounting for interaction with _element_
        elementZ -= element.thickness;
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        bool isStop = (element.curvatureRadius == 0);
        if (isStop) {
            // Compute _t_ at plane of aperture stop
            t = (elementZ - rLens.o.z) / rLens.d.z;
            if (t < 0)
                return 0;

        } else {
            // Intersect ray with element to compute _t_ and _n_
            Float radius = element.curvatureRadius;
            Float zCenter = elementZ + element.curvatureRadius;
            if (!IntersectSphericalElement(radius, zCenter, rLens, &t, &n))
                return 0;
        }
        DCHECK_GE(t, 0);

        // Test intersection point against element aperture
        Point3f pHit = rLens(t);
        if (isStop && apertureImage) {
            // Check intersection point against _apertureImage_
            Point2f uv((pHit.x / element.apertureRadius + 1) / 2,
                       (pHit.y / element.apertureRadius + 1) / 2);
            weight = apertureImage.BilerpChannel(uv, 0, WrapMode::Black);
            if (weight == 0)
                return 0;

        } else {
            // Check intersection point against spherical aperture
            if (Sqr(pHit.x) + Sqr(pHit.y) > Sqr(element.apertureRadius))
                return 0;
        }
        rLens.o = pHit;

        // Update ray path for element interface interaction
        if (!isStop) {
            Vector3f w;
            Float eta_i = element.eta;
            Float eta_t = (i > 0 && elementInterfaces[i - 1].eta != 0)
                              ? elementInterfaces[i - 1].eta
                              : 1;
            if (!Refract(Normalize(-rLens.d), n, eta_t / eta_i, nullptr, &w))
                return 0;
            rLens.d = w;
        }
    }
    // Transform lens system space ray back to camera space
    if (rOut)
        *rOut = Ray(Point3f(rLens.o.x, rLens.o.y, -rLens.o.z),
                    Vector3f(rLens.d.x, rLens.d.y, -rLens.d.z), rLens.time);

    return weight;
}

void RealisticCamera::ComputeCardinalPoints(Ray rIn, Ray rOut, Float *pz, Float *fz) {
    Float tf = -rOut.o.x / rOut.d.x;
    *fz = -rOut(tf).z;
    Float tp = (rIn.o.x - rOut.o.x) / rOut.d.x;
    *pz = -rOut(tp).z;
}

void RealisticCamera::ComputeThickLensApproximation(Float pz[2], Float fz[2]) const {
    // Find height $x$ from optical axis for parallel rays
    Float x = .001f * film.Diagonal();

    // Compute cardinal points for film side of lens system
    Ray rScene(Point3f(x, 0, LensFrontZ() + 1), Vector3f(0, 0, -1));
    Ray rFilm;
    if (!TraceLensesFromScene(rScene, &rFilm))
        ErrorExit("Unable to trace ray from scene to film for thick lens "
                  "approximation. Is aperture stop extremely small?");
    ComputeCardinalPoints(rScene, rFilm, &pz[0], &fz[0]);

    // Compute cardinal points for scene side of lens system
    rFilm = Ray(Point3f(x, 0, LensRearZ() - 1), Vector3f(0, 0, 1));
    if (TraceLensesFromFilm(rFilm, &rScene) == 0)
        ErrorExit("Unable to trace ray from film to scene for thick lens "
                  "approximation. Is aperture stop extremely small?");
    ComputeCardinalPoints(rFilm, rScene, &pz[1], &fz[1]);
}

Float RealisticCamera::FocusThickLens(Float focusDistance) {
    Float pz[2], fz[2];
    ComputeThickLensApproximation(pz, fz);
    LOG_VERBOSE("Cardinal points: p' = %f f' = %f, p = %f f = %f.\n", pz[0], fz[0], pz[1],
                fz[1]);
    LOG_VERBOSE("Effective focal length %f\n", fz[0] - pz[0]);
    // Compute translation of lens, _delta_, to focus at _focusDistance_
    Float f = fz[0] - pz[0];
    Float z = -focusDistance;
    Float c = (pz[1] - z - pz[0]) * (pz[1] - z - 4 * f - pz[0]);
    if (c <= 0)
        ErrorExit("Coefficient must be positive. It looks focusDistance %f "
                  " is too short for a given lenses configuration",
                  focusDistance);
    Float delta = (pz[1] - z + pz[0] - std::sqrt(c)) / 2;

    return elementInterfaces.back().thickness + delta;
}

Bounds2f RealisticCamera::BoundExitPupil(Float filmX0, Float filmX1) const {
    Bounds2f pupilBounds;
    // Sample a collection of points on the rear lens to find exit pupil
    const int nSamples = 1024 * 1024;
    // Compute bounding box of projection of rear element on sampling plane
    Float rearRadius = RearElementRadius();
    Bounds2f projRearBounds(Point2f(-1.5f * rearRadius, -1.5f * rearRadius),
                            Point2f(1.5f * rearRadius, 1.5f * rearRadius));

    for (int i = 0; i < nSamples; ++i) {
        // Find location of sample points on $x$ segment and rear lens element
        Point3f pFilm(Lerp((i + 0.5f) / nSamples, filmX0, filmX1), 0, 0);
        Float u[2] = {RadicalInverse(0, i), RadicalInverse(1, i)};
        Point3f pRear(Lerp(u[0], projRearBounds.pMin.x, projRearBounds.pMax.x),
                      Lerp(u[1], projRearBounds.pMin.y, projRearBounds.pMax.y),
                      LensRearZ());

        // Expand pupil bounds if ray makes it through the lens system
        if (!Inside(Point2f(pRear.x, pRear.y), pupilBounds) &&
            TraceLensesFromFilm(Ray(pFilm, pRear - pFilm), nullptr))
            pupilBounds = Union(pupilBounds, Point2f(pRear.x, pRear.y));
    }

    // Return degenerate bounds if no rays made it through the lens system
    if (pupilBounds.IsDegenerate()) {
        LOG_VERBOSE("Unable to find exit pupil in x = [%f,%f] on film.", filmX0, filmX1);
        return pupilBounds;
    }

    // Expand bounds to account for sample spacing
    pupilBounds =
        Expand(pupilBounds, 2 * Length(projRearBounds.Diagonal()) / std::sqrt(nSamples));

    return pupilBounds;
}

pstd::optional<ExitPupilSample> RealisticCamera::SampleExitPupil(Point2f pFilm,
                                                                 Point2f uLens) const {
    // Find exit pupil bound for sample distance from film center
    Float rFilm = std::sqrt(Sqr(pFilm.x) + Sqr(pFilm.y));
    int rIndex = rFilm / (film.Diagonal() / 2) * exitPupilBounds.size();
    rIndex = std::min<int>(exitPupilBounds.size() - 1, rIndex);
    Bounds2f pupilBounds = exitPupilBounds[rIndex];
    if (pupilBounds.IsDegenerate())
        return {};

    // Generate sample point inside exit pupil bound
    Point2f pLens = pupilBounds.Lerp(uLens);
    Float pdf = 1 / pupilBounds.Area();

    // Return sample point rotated by angle of _pFilm_ with $+x$ axis
    Float sinTheta = (rFilm != 0) ? pFilm.y / rFilm : 0;
    Float cosTheta = (rFilm != 0) ? pFilm.x / rFilm : 1;
    Point3f pPupil(cosTheta * pLens.x - sinTheta * pLens.y,
                   sinTheta * pLens.x + cosTheta * pLens.y, LensRearZ());
    return ExitPupilSample{pPupil, pdf};
}

pstd::optional<CameraRay> RealisticCamera::GenerateRay(CameraSample sample,
                                                       SampledWavelengths &lambda) const {
    // Find point on film, _pFilm_, corresponding to _sample.pFilm_
    Point2f s(sample.pFilm.x / film.FullResolution().x,
              sample.pFilm.y / film.FullResolution().y);
    Point2f pFilm2 = physicalExtent.Lerp(s);
    Point3f pFilm(-pFilm2.x, pFilm2.y, 0);

    // Trace ray from _pFilm_ through lens system
    pstd::optional<ExitPupilSample> eps =
        SampleExitPupil(Point2f(pFilm.x, pFilm.y), sample.pLens);
    if (!eps)
        return {};
    Ray rFilm(pFilm, eps->pPupil - pFilm);
    Ray ray;
    Float weight = TraceLensesFromFilm(rFilm, &ray);
    if (weight == 0)
        return {};

    // Finish initialization of _RealisticCamera_ ray
    ray.time = SampleTime(sample.time);
    ray.medium = medium;
    ray = RenderFromCamera(ray);
    ray.d = Normalize(ray.d);

    // Compute weighting for _RealisticCamera_ ray
    Float cosTheta = Normalize(rFilm.d).z;
    weight *= Pow<4>(cosTheta) / (eps->pdf * Sqr(LensRearZ()));
    return CameraRay{ray, SampledSpectrum(weight)};
}

STAT_PERCENT("Camera/Rays vignetted by lens system", vignettedRays, totalRays);

std::string RealisticCamera::LensElementInterface::ToString() const {
    return StringPrintf("[ LensElementInterface curvatureRadius: %f thickness: %f "
                        "eta: %f apertureRadius: %f ]",
                        curvatureRadius, thickness, eta, apertureRadius);
}

Float RealisticCamera::TraceLensesFromScene(const Ray &rCamera, Ray *rOut) const {
    Float elementZ = -LensFrontZ();
    // Transform _rCamera_ from camera to lens system space
    const Transform LensFromCamera = Scale(1, 1, -1);
    Ray rLens = LensFromCamera(rCamera);
    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        bool isStop = (element.curvatureRadius == 0);
        if (isStop) {
            t = (elementZ - rLens.o.z) / rLens.d.z;
            if (t < 0)
                return 0;
        } else {
            Float radius = element.curvatureRadius;
            Float zCenter = elementZ + element.curvatureRadius;
            if (!IntersectSphericalElement(radius, zCenter, rLens, &t, &n))
                return 0;
        }

        // Test intersection point against element aperture
        // Don't worry about the aperture image here.
        Point3f pHit = rLens(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        if (r2 > element.apertureRadius * element.apertureRadius)
            return 0;
        rLens.o = pHit;

        // Update ray path for from-scene element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = (i == 0 || elementInterfaces[i - 1].eta == 0)
                              ? 1
                              : elementInterfaces[i - 1].eta;
            Float eta_t = (elementInterfaces[i].eta != 0) ? elementInterfaces[i].eta : 1;
            if (!Refract(Normalize(-rLens.d), n, eta_t / eta_i, nullptr, &wt))
                return 0;
            rLens.d = wt;
        }
        elementZ += element.thickness;
    }
    // Transform _rLens_ from lens system space back to camera space
    if (rOut)
        *rOut = Ray(Point3f(rLens.o.x, rLens.o.y, -rLens.o.z),
                    Vector3f(rLens.d.x, rLens.d.y, -rLens.d.z), rLens.time);
    return 1;
}

void RealisticCamera::DrawLensSystem() const {
    Float sumz = -LensFrontZ();
    Float z = sumz;
    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        Float r = element.curvatureRadius;
        if (r == 0) {
            // stop
            printf("{Thick, Line[{{%f, %f}, {%f, %f}}], ", z, element.apertureRadius, z,
                   2 * element.apertureRadius);
            printf("Line[{{%f, %f}, {%f, %f}}]}, ", z, -element.apertureRadius, z,
                   -2 * element.apertureRadius);
        } else {
            Float theta = std::abs(SafeASin(element.apertureRadius / r));
            if (r > 0) {
                // convex as seen from front of lens
                Float t0 = Pi - theta;
                Float t1 = Pi + theta;
                printf("Circle[{%f, 0}, %f, {%f, %f}], ", z + r, r, t0, t1);
            } else {
                // concave as seen from front of lens
                Float t0 = -theta;
                Float t1 = theta;
                printf("Circle[{%f, 0}, %f, {%f, %f}], ", z + r, -r, t0, t1);
            }
            if (element.eta != 0 && element.eta != 1) {
                // connect top/bottom to next element
                CHECK_LT(i + 1, elementInterfaces.size());
                Float nextApertureRadius = elementInterfaces[i + 1].apertureRadius;
                Float h = std::max(element.apertureRadius, nextApertureRadius);
                Float hlow = std::min(element.apertureRadius, nextApertureRadius);

                Float zp0, zp1;
                if (r > 0) {
                    zp0 = z + element.curvatureRadius -
                          element.apertureRadius / std::tan(theta);
                } else {
                    zp0 = z + element.curvatureRadius +
                          element.apertureRadius / std::tan(theta);
                }

                Float nextCurvatureRadius = elementInterfaces[i + 1].curvatureRadius;
                Float nextTheta =
                    std::abs(SafeASin(nextApertureRadius / nextCurvatureRadius));
                if (nextCurvatureRadius > 0) {
                    zp1 = z + element.thickness + nextCurvatureRadius -
                          nextApertureRadius / std::tan(nextTheta);
                } else {
                    zp1 = z + element.thickness + nextCurvatureRadius +
                          nextApertureRadius / std::tan(nextTheta);
                }

                // Connect tops
                printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, h, zp1, h);
                printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, -h, zp1, -h);

                // vertical lines when needed to close up the element profile
                if (element.apertureRadius < nextApertureRadius) {
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, h, zp0, hlow);
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, -h, zp0, -hlow);
                } else if (element.apertureRadius > nextApertureRadius) {
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp1, h, zp1, hlow);
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp1, -h, zp1, -hlow);
                }
            }
        }
        z += element.thickness;
    }

    // 24mm height for 35mm film
    printf("Line[{{0, -.012}, {0, .012}}], ");
    // optical axis
    printf("Line[{{0, 0}, {%f, 0}}] ", 1.2f * sumz);
}

void RealisticCamera::DrawRayPathFromFilm(const Ray &r, bool arrow,
                                          bool toOpticalIntercept) const {
    Float elementZ = 0;
    // Transform _ray_ from camera to lens system space
    static const Transform LensFromCamera = Scale(1, 1, -1);
    Ray ray = LensFromCamera(r);
    printf("{ ");
    if (TraceLensesFromFilm(r, nullptr) == 0) {
        printf("Dashed, RGBColor[.8, .5, .5]");
    } else
        printf("RGBColor[.5, .5, .8]");

    for (int i = elementInterfaces.size() - 1; i >= 0; --i) {
        const LensElementInterface &element = elementInterfaces[i];
        elementZ -= element.thickness;
        bool isStop = (element.curvatureRadius == 0);
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        if (isStop)
            t = -(ray.o.z - elementZ) / ray.d.z;
        else {
            Float radius = element.curvatureRadius;
            Float zCenter = elementZ + element.curvatureRadius;
            if (!IntersectSphericalElement(radius, zCenter, ray, &t, &n))
                goto done;
        }
        CHECK_GE(t, 0);

        printf(", Line[{{%f, %f}, {%f, %f}}]", ray.o.z, ray.o.x, ray(t).z, ray(t).x);

        // Test intersection point against element aperture
        Point3f pHit = ray(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        Float apertureRadius2 = element.apertureRadius * element.apertureRadius;
        if (r2 > apertureRadius2)
            goto done;
        ray.o = pHit;

        // Update ray path for element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = element.eta;
            Float eta_t = (i > 0 && elementInterfaces[i - 1].eta != 0)
                              ? elementInterfaces[i - 1].eta
                              : 1;
            if (!Refract(Normalize(-ray.d), n, eta_t / eta_i, nullptr, &wt))
                goto done;
            ray.d = wt;
        }
    }

    ray.d = Normalize(ray.d);
    {
        Float ta = std::abs(elementZ / 4);
        if (toOpticalIntercept) {
            ta = -ray.o.x / ray.d.x;
            printf(", Point[{%f, %f}]", ray(ta).z, ray(ta).x);
        }
        printf(", %s[{{%f, %f}, {%f, %f}}]", arrow ? "Arrow" : "Line", ray.o.z, ray.o.x,
               ray(ta).z, ray(ta).x);

        // overdraw the optical axis if needed...
        if (toOpticalIntercept)
            printf(", Line[{{%f, 0}, {%f, 0}}]", ray.o.z, ray(ta).z * 1.05f);
    }

done:
    printf("}");
}

void RealisticCamera::DrawRayPathFromScene(const Ray &r, bool arrow,
                                           bool toOpticalIntercept) const {
    Float elementZ = LensFrontZ() * -1;

    // Transform _ray_ from camera to lens system space
    static const Transform LensFromCamera = Scale(1, 1, -1);
    Ray ray = LensFromCamera(r);
    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        bool isStop = (element.curvatureRadius == 0);
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        if (isStop)
            t = -(ray.o.z - elementZ) / ray.d.z;
        else {
            Float radius = element.curvatureRadius;
            Float zCenter = elementZ + element.curvatureRadius;
            if (!IntersectSphericalElement(radius, zCenter, ray, &t, &n))
                return;
        }
        CHECK_GE(t, 0.f);

        printf("Line[{{%f, %f}, {%f, %f}}],", ray.o.z, ray.o.x, ray(t).z, ray(t).x);

        // Test intersection point against element aperture
        Point3f pHit = ray(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        Float apertureRadius2 = element.apertureRadius * element.apertureRadius;
        if (r2 > apertureRadius2)
            return;
        ray.o = pHit;

        // Update ray path for from-scene element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = (i == 0 || elementInterfaces[i - 1].eta == 0.f)
                              ? 1.f
                              : elementInterfaces[i - 1].eta;
            Float eta_t =
                (elementInterfaces[i].eta != 0.f) ? elementInterfaces[i].eta : 1.f;
            if (!Refract(Normalize(-ray.d), n, eta_t / eta_i, nullptr, &wt))
                return;
            ray.d = wt;
        }
        elementZ += element.thickness;
    }

    // go to the film plane by default
    {
        Float ta = -ray.o.z / ray.d.z;
        if (toOpticalIntercept) {
            ta = -ray.o.x / ray.d.x;
            printf("Point[{%f, %f}], ", ray(ta).z, ray(ta).x);
        }
        printf("%s[{{%f, %f}, {%f, %f}}]", arrow ? "Arrow" : "Line", ray.o.z, ray.o.x,
               ray(ta).z, ray(ta).x);
    }
}

void RealisticCamera::RenderExitPupil(Float sx, Float sy, const char *filename) const {
    Point3f pFilm(sx, sy, 0);

    const int nSamples = 2048;
    Image image(PixelFormat::Float, {nSamples, nSamples}, {"Y"});

    for (int y = 0; y < nSamples; ++y) {
        Float fy = (Float)y / (Float)(nSamples - 1);
        Float ly = Lerp(fy, -RearElementRadius(), RearElementRadius());
        for (int x = 0; x < nSamples; ++x) {
            Float fx = (Float)x / (Float)(nSamples - 1);
            Float lx = Lerp(fx, -RearElementRadius(), RearElementRadius());

            Point3f pRear(lx, ly, LensRearZ());

            if (lx * lx + ly * ly > RearElementRadius() * RearElementRadius())
                image.SetChannel({x, y}, 0, 1.);
            else if (TraceLensesFromFilm(Ray(pFilm, pRear - pFilm), nullptr))
                image.SetChannel({x, y}, 0, 0.5);
            else
                image.SetChannel({x, y}, 0, 0.);
        }
    }

    image.Write(filename);
}

void RealisticCamera::TestExitPupilBounds() const {
    Float filmDiagonal = film.Diagonal();

    static RNG rng;

    Float u = rng.Uniform<Float>();
    Point3f pFilm(u * filmDiagonal / 2, 0, 0);

    Float r = pFilm.x / (filmDiagonal / 2);
    int pupilIndex = std::min<int>(exitPupilBounds.size() - 1,
                                   pstd::floor(r * (exitPupilBounds.size() - 1)));
    Bounds2f pupilBounds = exitPupilBounds[pupilIndex];
    if (pupilIndex + 1 < (int)exitPupilBounds.size())
        pupilBounds = Union(pupilBounds, exitPupilBounds[pupilIndex + 1]);

    // Now, randomly pick points on the aperture and see if any are outside
    // of pupil bounds...
    for (int i = 0; i < 1000; ++i) {
        Point2f u2{rng.Uniform<Float>(), rng.Uniform<Float>()};
        Point2f pd = SampleUniformDiskConcentric(u2);
        pd *= RearElementRadius();

        Ray testRay(pFilm, Point3f(pd.x, pd.y, 0.f) - pFilm);
        Ray testOut;
        if (!TraceLensesFromFilm(testRay, &testOut))
            continue;

        if (!Inside(pd, pupilBounds)) {
            fprintf(stderr,
                    "Aha! (%f,%f) went through, but outside bounds (%f,%f) - "
                    "(%f,%f)\n",
                    pd.x, pd.y, pupilBounds.pMin[0], pupilBounds.pMin[1],
                    pupilBounds.pMax[0], pupilBounds.pMax[1]);
            RenderExitPupil(
                (Float)pupilIndex / exitPupilBounds.size() * filmDiagonal / 2.f, 0.f,
                "low.exr");
            RenderExitPupil(
                (Float)(pupilIndex + 1) / exitPupilBounds.size() * filmDiagonal / 2.f,
                0.f, "high.exr");
            RenderExitPupil(pFilm.x, 0.f, "mid.exr");
            exit(0);
        }
    }
    fprintf(stderr, ".");
}

std::string RealisticCamera::ToString() const {
    return StringPrintf(
        "[ RealisticCamera %s elementInterfaces: %s exitPupilBounds: %s ]",
        CameraBase::ToString(), elementInterfaces, exitPupilBounds);
}

RealisticCamera *RealisticCamera::Create(const ParameterDictionary &parameters,
                                         const CameraTransform &cameraTransform,
                                         Film film, Medium medium, const FileLoc *loc,
                                         Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    // Realistic camera-specific parameters
    std::string lensFile = ResolveFilename(parameters.GetOneString("lensfile", ""));
    Float apertureDiameter = parameters.GetOneFloat("aperturediameter", 1.0);
    Float focusDistance = parameters.GetOneFloat("focusdistance", 10.0);

    if (lensFile.empty()) {
        Error(loc, "No lens description file supplied!");
        return nullptr;
    }
    // Load element data from lens description file
    std::vector<Float> lensParameters = ReadFloatFile(lensFile);
    if (lensParameters.empty()) {
        Error(loc, "Error reading lens specification file \"%s\".", lensFile);
        return nullptr;
    }
    if (lensParameters.size() % 4 != 0) {
        Error(loc,
              "%s: excess values in lens specification file; "
              "must be multiple-of-four values, read %d.",
              lensFile, (int)lensParameters.size());
        return nullptr;
    }

    int builtinRes = 256;
    auto rasterize = [&](pstd::span<const Point2f> vert) {
        Image image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"}, nullptr, alloc);

        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                Point2f p(-1 + 2 * (x + 0.5f) / image.Resolution().x,
                          -1 + 2 * (y + 0.5f) / image.Resolution().y);
                int windingNumber = 0;
                // Test against edges
                for (int i = 0; i < vert.size(); ++i) {
                    int i1 = (i + 1) % vert.size();
                    Float e = (p[0] - vert[i][0]) * (vert[i1][1] - vert[i][1]) -
                              (p[1] - vert[i][1]) * (vert[i1][0] - vert[i][0]);
                    if (vert[i].y <= p.y) {
                        if (vert[i1].y > p.y && e > 0)
                            ++windingNumber;
                    } else if (vert[i1].y <= p.y && e < 0)
                        --windingNumber;
                }

                image.SetChannel({x, y}, 0, windingNumber == 0 ? 0.f : 1.f);
            }

        return image;
    };

    std::string apertureName = parameters.GetOneString("aperture", "");
    Image apertureImage(alloc);
    if (!apertureName.empty()) {
        // built-in diaphragm shapes
        if (apertureName == "gaussian") {
            apertureImage = Image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"},
                                  nullptr, alloc);
            for (int y = 0; y < apertureImage.Resolution().y; ++y)
                for (int x = 0; x < apertureImage.Resolution().x; ++x) {
                    Point2f uv(-1 + 2 * (x + 0.5f) / apertureImage.Resolution().x,
                               -1 + 2 * (y + 0.5f) / apertureImage.Resolution().y);
                    Float r2 = Sqr(uv.x) + Sqr(uv.y);
                    Float sigma2 = 1;
                    Float v = std::max<Float>(
                        0, std::exp(-r2 / sigma2) - std::exp(-1 / sigma2));
                    apertureImage.SetChannel({x, y}, 0, v);
                }
        } else if (apertureName == "square") {
            apertureImage = Image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"},
                                  nullptr, alloc);
            for (int y = .25 * builtinRes; y < .75 * builtinRes; ++y)
                for (int x = .25 * builtinRes; x < .75 * builtinRes; ++x)
                    apertureImage.SetChannel({x, y}, 0, 4.f);
        } else if (apertureName == "pentagon") {
            // https://mathworld.wolfram.com/RegularPentagon.html
            Float c1 = (std::sqrt(5.f) - 1) / 4;
            Float c2 = (std::sqrt(5.f) + 1) / 4;
            Float s1 = std::sqrt(10.f + 2.f * std::sqrt(5.f)) / 4;
            Float s2 = std::sqrt(10.f - 2.f * std::sqrt(5.f)) / 4;
            // Vertices in CW order.
            Point2f vert[5] = {Point2f(0, 1), {s1, c1}, {s2, -c2}, {-s2, -c2}, {-s1, c1}};
            // Scale down slightly
            for (int i = 0; i < 5; ++i)
                vert[i] *= .8f;
            apertureImage = rasterize(vert);
        } else if (apertureName == "star") {
            // 5-sided. Vertices are two pentagons--inner and outer radius
            pstd::array<Point2f, 10> vert;
            for (int i = 0; i < 10; ++i) {
                // inner radius: https://math.stackexchange.com/a/2136996
                Float r =
                    (i & 1) ? 1.f : (std::cos(Radians(72.f)) / std::cos(Radians(36.f)));
                vert[i] = Point2f(r * std::cos(Pi * i / 5.f), r * std::sin(Pi * i / 5.f));
            }
            std::reverse(vert.begin(), vert.end());
            apertureImage = rasterize(vert);
        } else {
            ImageAndMetadata im = Image::Read(ResolveFilename(apertureName), alloc);
            apertureImage = std::move(im.image);
            if (apertureImage.NChannels() > 1) {
                ImageChannelDesc rgbDesc = apertureImage.GetChannelDesc({"R", "G", "B"});
                if (!rgbDesc)
                    ErrorExit("%s: didn't find R, G, B channels to average for "
                              "aperture image.",
                              apertureName);

                Image mono(PixelFormat::Float, apertureImage.Resolution(), {"Y"}, nullptr,
                           alloc);
                for (int y = 0; y < mono.Resolution().y; ++y)
                    for (int x = 0; x < mono.Resolution().x; ++x) {
                        Float avg = apertureImage.GetChannels({x, y}, rgbDesc).Average();
                        mono.SetChannel({x, y}, 0, avg);
                    }

                apertureImage = std::move(mono);
            }
        }

        if (apertureImage) {
            apertureImage.FlipY();

            // Normalize it so that brightness matches a circular aperture
            Float sum = 0;
            for (int y = 0; y < apertureImage.Resolution().y; ++y)
                for (int x = 0; x < apertureImage.Resolution().x; ++x)
                    sum += apertureImage.GetChannel({x, y}, 0);
            Float avg =
                sum / (apertureImage.Resolution().x * apertureImage.Resolution().y);

            Float scale = (Pi / 4) / avg;
            for (int y = 0; y < apertureImage.Resolution().y; ++y)
                for (int x = 0; x < apertureImage.Resolution().x; ++x)
                    apertureImage.SetChannel({x, y}, 0,
                                             apertureImage.GetChannel({x, y}, 0) * scale);
        }
    }

    return alloc.new_object<RealisticCamera>(cameraBaseParameters, lensParameters,
                                             focusDistance, apertureDiameter,
                                             std::move(apertureImage), alloc);
}
// --------------------------HUMAN EYE CAMERA STARTS HERE------------------
struct biconic_params {
    float Rx, Ry, Cx, Cy;
    Ray ray;
};

double BiconicSag(double t, void *params) {
    struct biconic_params *p;
    p = (struct biconic_params *)params;

    Point3f intersect = p->ray(t);
    float x, y, z;
    z = intersect.z;
    x = intersect.x;
    y = intersect.y;

    float f, g, g_term;
    f = (x * x) / p->Rx + (y * y) / p->Ry;
    g_term = 1 - (1 + p->Cx) * (x * x) / (p->Rx * p->Rx) -
             (1 + p->Cy) * (y * y) / (p->Ry * p->Ry);

    if (g_term < 0) {
        // TODO: What to do here? Let's just make it a small number to ensure that f/g =/=
        // z.
        g_term = 0.001;
    }

    g = 1 + std::sqrt(g_term);

    return z - f / g;
}

// HumanEyeCamera Method Definitions
HumanEyeCamera::HumanEyeCamera(CameraBaseParameters baseParameters,
                               pstd::vector<LensElementEye> &eyeInterfacesData,
                               Float pupilDiameter, Float retinaDistance,
                               Float retinaRadius, Float retinaSemiDiam,
                               pstd::vector<Spectrum> iorSpectra,
                               pstd::vector<Point3f> surfaceLookupTable,
                               bool diffractionEnabled, 
                               Allocator alloc)
    : CameraBase(baseParameters),
      lensEls(alloc),
      lookupTable(surfaceLookupTable, alloc),
      pupilDiameter(pupilDiameter),
      retinaDistance(retinaDistance),
      retinaRadius(retinaRadius),
      retinaSemiDiam(retinaSemiDiam),
      iorSpectra(iorSpectra),
      diffractionEnabled(diffractionEnabled) {
    // Initialize _elementInterfaces_ for camera
    lensEls = eyeInterfacesData;

    // Calculate physical extent (for computations with lookuptable)
    Float aspect = (Float)film.FullResolution().y / (Float)film.FullResolution().x;
    Float diagonal = film.Diagonal();
    Float x = std::sqrt(Sqr(diagonal) / (1 + Sqr(aspect)));
    Float y = aspect * x;
    physicalExtent = Bounds2f(Point2f(-x / 2, -y / 2), Point2f(x / 2, y / 2));

    // To calculate the "film diagonal", we use the retina semi-diameter. The film
    // diagonal is the diagonal of the rectangular image rendered out by PBRT, in real
    // units. Since we restrict samples to a circular image, we can calculate the film
    // diagonal to be the same as a square that circumscribes the circular image.
    retinaDiag = retinaSemiDiam * 1.4142 * 2;  // sqrt(2)*2

    // We are going to use our own error handling for gsl to prevent it from crashing the
    // moment a ray doesn't intersect.
    gsl_set_error_handler_off();

    // Set up GSL random number generator for diffraction modeling
    const gsl_rng_type *T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
}

pstd::optional<CameraRay> HumanEyeCamera::GenerateRay(CameraSample sample,
                                                      SampledWavelengths &lambda) const {
    //                         z=0         z= -filmDistance
    //                  |  |    ||           |
    //                  |  |    ||           |
    //  Scene <---------|--|----||---------> |
    //            +z    |  |    ||    -z     |
    //                  |  |    ||           |
    //                Lens Elements        Sensor
    //

    // Choose surface to project to

    Point3f startingPoint;

    // If a surface lookup table is defined, use the legacy code
    if (useLookupTable()) {
        startingPoint = mapLookupTable(sample.pFilm);

        // If a sphere is defined, use the legacy code
    } else {
        startingPoint = mapToSphere(sample.pFilm);
    }
    // printf("Starting point: %f,%f,%f
    // \n",startingPoint.x,startingPoint.y,startingPoint.z);
    Float lensU, lensV;
    Point2f lens = SampleUniformDiskConcentric(sample.pLens);
    lensU = lens.x;
    lensV = lens.y;

    // We need to shoot rays toward the disc that fits inside the curvature of first lens
    // surface. Since we no longer have spherical elements, we have to calculate it as
    // follows:
    //  TODO: This is just a guess. Is there a correct way to do this? Should we look at
    //  the exitPupil code in PBRTv3?

    Float lensP_semiDiam = lensEls[lensEls.size() - 1].semiDiameter;
    Float lensP_radius = lensEls[lensEls.size() - 1].radiusX;

    // sgn(lensP_radius)
    // It's very rare for the first lens element to have a negative radius (spherical
    // center toward sensor)...but just in case:
    Float sgn_radius = (lensP_radius > 0) - (lensP_radius < 0);
    Float discDistance =
        sgn_radius * BiconicZ(lensP_semiDiam, 0, lensEls[lensEls.size() - 1]);

    // Scale the normalized lens coordinates by the size of the first lens element
    lensU *= lensP_semiDiam;
    lensV *= lensP_semiDiam;

    Point3f pointOnLens = Point3f(
        lensU, lensV,
        discDistance);  // We aim the ray at a flat disk, will that cause problems later?

    Ray ray;
    ray.o = startingPoint;  // initialize ray origin
    ray.d = Normalize(Vector3f(pointOnLens - ray.o));
    ray.wavelength = lambda[0];

    // re-calculate correct lambda
    lambda = SampledWavelengths::SampleUniform(0.5);
    // DEBUG
    /*
        // Scene to retina
        startingPoint = Point(0,0.00081193431816,-16.319999973);
        pointOnLens = Point(0,1.544122632,0.21476753379);
        ray->o = startingPoint;
        ray->d = Normalize(pointOnLens - ray->o);
        ray->wavelength = 550;


        // Retina to scene
        startingPoint = Point(0,0,-16.3200);
        pointOnLens = Point(0,1.5294,0.2084);
        ray->o = startingPoint;
        ray->d = Normalize(pointOnLens - ray->o);
        ray->wavelength = 550;
        */

    // --------------------------------------------------------
    // --- Trace through the lens elements of the main lens ---
    // --------------------------------------------------------

    Float lensDistance =
        0;  // How far we are from the "0" on the z-axis (see diagram above)

    for (int i = lensEls.size() - 1; i >= 0; i--) {
        ray.o = startingPoint;
        lensDistance += lensEls[i].thickness;

        // If the ray direction is zero, there is probably internal reflection going on
        // somewhere. We will just terminate the ray here to avoid assert errors.
        if (ray.d == Vector3f(0, 0, 0)) {
            return {};
        }

        // DEBUG
        // ----
        /*
            std::cout << "\n" << std::endl;
            std::cout << "i = " << i << std::endl;
            std::cout << "start " << ray->o.x << " " << ray->o.y << " " << ray->o.z <<
           std::endl; std::cout << "dir " << ray->d.x << " " << ray->d.y << " " <<
           ray->d.z << std::endl;
            */
        // ----

        Float tHit = 0;
        bool intersected = false;
        Vector3f normalVec(0, 0, 1);
        Point3f intersectPoint(0, 0, 0);

        if (lensEls[i].radiusX == 0 && lensEls[i].radiusY == 0) {
            // ---------------------
            // --- APERTURE CASE ---
            // ---------------------

            Float tAperture = 0;
            if (i == lensEls.size() - 1)
                tAperture = retinaDistance /
                            ray.d.z;  // special case for when aperture is the 1st element
            else
                tAperture = (lensDistance - ray.o.z) / (ray.d.z);

            // Point where the ray intersects the aperture plane
            Point3f intersectPoint = (ray)(tAperture);
            normalVec = Vector3f(0, 0, 1);

            // Check if ray makes it through the aperture
            if ((intersectPoint.x * intersectPoint.x +
                 intersectPoint.y * intersectPoint.y) >
                (lensEls[i].semiDiameter * lensEls[i].semiDiameter)) {
                return {};
            }

            if (diffractionEnabled) {
                // DEBUG: Check that direction did change
                // printf("Before: %f, %f, %f \n", ray.d.x, ray.d.y, ray.d.z);
                // Adjust ray direction using HURB diffraction
                Vector3f newDiffractedDir;
                diffractHURB(intersectPoint, lensEls[i].semiDiameter, ray.wavelength,
                             ray.d, &newDiffractedDir);
                ray.d = newDiffractedDir;

                // DEBUG: Check that direction did change
                // printf("After: %f, %f, %f \n", ray.d.x,ray.d.y,ray.d.z);
            }

            startingPoint = intersectPoint;

        } else {
            // ----------------------------
            // --- REGULAR ELEMENT CASE ---
            // ----------------------------

            //---- Find intersection point ----

            // Since the surface is defined in object space with the edge at zero, so we
            // need to move the ray accordingly from world space to object space. This is
            // simply a translation according to the location of the surface.

            // z(x,y) is defined like this (flipped across the z axis for +r vs -r):
            //
            //   :   /
            //   :  /
            //   : /
            //   :|
            //   :|
            //   :|
            //   : \d_zema
            //   :  \
            //   :   \
            //  z=0
            //

            Float zShift = -lensDistance;

            // Does the ray intersect the lens surface? If so, where?
            intersected =
                IntersectLensElAspheric(ray, &tHit, lensEls[i], zShift, &normalVec);

            if (intersected) {
                intersectPoint = ray(tHit);

                // DEBUG
                //  std::cout << "Intersect point: " << intersectPoint.x << " " <<
                //  intersectPoint.y << " " << intersectPoint.z << std::endl;

                // ---- Apply Snell's Law ----
                //                  |
                // scene ..... n2   |   n1 ..... sensor
                //           [i-1]  |  [i]
                //                  |
                //                 lens
                //

                // The user can load IOR spectra for each ocular medium into ior1, ior2,
                // etc. In the lens file, they can then specify with medium they would
                // like to use. The number (X) corresponds to iorX.

                Float n1, n2;

                n1 = lookUpIOR(lensEls[i].mediumIndex, ray);

                // If we're at the lens surface closest to the scene, n2 should be air.
                if (i - 1 >= 0) {
                    n2 = lookUpIOR(lensEls[i - 1].mediumIndex, ray);

                    // Trisha: If we're entering the aperture (sometimes we put n2 == 0
                    // when that is the case) we skip the n2 == 0 and apply the next
                    // medium. (We put this in the current if statement so we can handle
                    // the unique case when the aperture is the first element.)

                    //                 |    /
                    //                 |   |
                    //                 |  |
                    // <-----            |    <------
                    //  n2, [i-2]      |  |        n1, [i]
                    //                 |   |
                    //                 |    \
                    //
                    //           aperture, [i-1]

                    if (n2 == 0)
                        n2 = lookUpIOR(lensEls[i - 2].mediumIndex, ray);

                } else {
                    n2 = 1;
                }

                // If n1 = n2 = 0, something is wrong!
                if (n1 == 0 & n2 == 0) {
                    Error("Index of refractions are set to zero. Something is wrong.");
                }
                applySnellsLaw(n1, n2, 0, normalVec, &ray);
                // --- Update ray starting point ---

                startingPoint = intersectPoint;

            } else {
                return {};
            }
        }
    }

    ray.o = startingPoint;

    // DEBUG

    // ----
    /*
        std::cout << "\n" << std::endl;
        std::cout << "final" << std::endl;
        std::cout << ray->o.x << " " << ray->o.y << " " << ray->o.z << std::endl;
        std::cout << ray->d.x << " " << ray->d.y << " " << ray->d.z << std::endl;
        */
    // ----

    // Move the origin to the front of the eye. This can be important for small distances
    // and accommodation measurements.
    Float current_wavelength = ray.wavelength; // keep wavelength after renderfromcamera
    ray.o.z = ray.o.z - frontThickness;
    ray = RenderFromCamera(ray);
    ray.d = Normalize(ray.d);
    ray.medium = medium;
    ray.wavelength = current_wavelength;

    // No weighting for now...we should add it in!
    // LensRearZ() is used for omni/realistic, not for humaneye yet.
    return CameraRay{ray};
}

void HumanEyeCamera::applySnellsLaw(Float n1, Float n2, Float lensRadius,
                                    Vector3f &normalVec, Ray *ray) const {
    //                  |
    // scene ..... n2   |   n1 ..... sensor
    //                  |
    //                  |
    //                 lens

    Vector3f s1 = ray->d;
    if (lensRadius > 0)
        normalVec = -normalVec;

    Float radicand =
        1 - (n1 / n2) * (n1 / n2) * Dot(Cross(normalVec, s1), Cross(normalVec, s1));
    if (radicand < 0) {
        ray->d = Vector3f(0, 0, 0);
        // Tlian: I put this return back in so the assert doesn't catch.
        return;  // reflection, no refraction - might want to change for lens flare!
    }

    Vector3f s2 = n1 / n2 * (Cross(normalVec, Cross(-1 * normalVec, s1))) -
                  normalVec * std::sqrt(radicand);
    ray->d = Normalize(s2);  // reassign the direction to the new direction
}

bool HumanEyeCamera::IntersectLensElAspheric(const Ray &r, Float *tHit,
                                             LensElementEye currElement, Float zShift,
                                             Vector3f *n) const {
    // This code lets us find the intersection with an aspheric surface. At tHit, the ray
    // will intersect the surface. Therefore: If (x,y,z) = ray_origin + thit *
    // ray_direction then: z - u(x,y) = 0 where u(x,y) is the SAG of the surface, defined
    // in Eq.1 of Einighammer et al. 2009 and in the Zemax help page under biconic
    // surfaces. We can use this fact to solve for thit. If the surface is not a sphere,
    // this is a messy polynomial. So instead we use a numeric root-finding method (Van
    // Wijingaarden-Dekker-Brent's Method.) This method is available in the GSL library.

    // DEBUG
    /*
        std::cout << "r.o = " << r.o.x << "," << r.o.y << "," << r.o.z << std::endl;
        std::cout << "r.d = " << r.d.x << "," << r.d.y << "," << r.d.z << std::endl;
        */

    // Move ray to object(lens) space.
    Ray objSpaceRay = r;
    objSpaceRay.o = objSpaceRay.o + Vector3f(0, 0, zShift);

    int status;
    int iter = 0, max_iter = 100;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double root = 0;
    double x_lo = 0.0;
    double x_hi;
    if (currElement.thickness == 0) {
        // Probably the surface closest to the retina.
        x_hi = retinaDistance * 2;
    } else {
        x_hi = currElement.thickness * 1.5;  // thit will probably be less than this
    }
    gsl_function F;

    // DEBUG
    /*
        std::cout << "radiusX = " << currElement.radiusX << std::endl;
        std::cout << "radiusY = " << currElement.radiusY << std::endl;
        std::cout << "conicConstantX = " << currElement.conicConstantX << std::endl;
        std::cout << "conicConstantY = " << currElement.conicConstantY << std::endl;
        std::cout << "objSpaceRay.o = " << objSpaceRay.o.x << "," << objSpaceRay.o.y <<
       "," << objSpaceRay.o.z << std::endl; std::cout << "objSpaceRay.d = " <<
       objSpaceRay.d.x << "," << objSpaceRay.d.y << "," << objSpaceRay.d.z << std::endl;
        */

    struct biconic_params params = {currElement.radiusX, currElement.radiusY,
                                    currElement.conicConstantX,
                                    currElement.conicConstantY, objSpaceRay};
    F.function = &BiconicSag;
    F.params = &params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc(T);

    status = gsl_root_fsolver_set(s, &F, x_lo, x_hi);
    if (status != 0) {
        // Ray probably does not intersect. This might depend on the x_hi set above, i.e.
        // if it's too small OR too large. TODO: Can we check this?
        gsl_root_fsolver_free(s);
        return false;
    }

    // DEBUG
    /*
        printf ("using %s method\n",
        gsl_root_fsolver_name (s));

        printf ("%5s [%9s, %9s] %9s %10s %9s\n",
        "iter", "lower", "upper", "root",
        "err", "err(est)");
        */

    do {
        iter++;
        gsl_root_fsolver_iterate(s);
        root = gsl_root_fsolver_root(s);
        x_lo = gsl_root_fsolver_x_lower(s);
        x_hi = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(x_lo, x_hi, 0, 0.0001);

        if (status == GSL_SUCCESS) {
            // DEBUG
            /*
                printf ("Converged:\n");
                printf ("%5d [%.7f, %.7f] %.7f %+.7f %.7f\n",
                iter, x_lo, x_hi,
                root, root - root,
                x_hi - x_lo);
                */

            gsl_root_fsolver_free(s);

            // DEBUG
            /*
                std::cout << "root = " << root << std::endl;
                std::cout << "r.o = " << r.o.x << "," << r.o.y << "," << r.o.z <<
               std::endl; std::cout << "r.d = " << r.d.x << "," << r.d.y << "," << r.d.z
               << std::endl;
                */

            *tHit = root;
            Point3f intersect = r(*tHit);

            // Check if intersection is within the semi-diameter of the lens, if not we
            // return false (no intersect) (If we don't do this here, we might get a
            // complex normal which would crash the rendering.)
            if (intersect.x * intersect.x + intersect.y * intersect.y >
                (currElement.semiDiameter * currElement.semiDiameter)) {
                return false;
            }

            // Calculate normal at intersection
            // These equations are from Eq 2, 3, and 4 in Einighammer 2009
            Float term1 = ((1 + currElement.conicConstantX) * intersect.x * intersect.x) /
                          (currElement.radiusX * currElement.radiusX);
            Float term2 = ((1 + currElement.conicConstantY) * intersect.y * intersect.y) /
                          (currElement.radiusY * currElement.radiusY);

            Float fprime_x = 2 * intersect.x / currElement.radiusX;
            Float gprime_x = (-1 * (1 + currElement.conicConstantX) * intersect.x) /
                             (currElement.radiusX * currElement.radiusX *
                              std::sqrt(1 - term1 - term2));

            Float fprime_y = 2 * intersect.y / currElement.radiusY;
            Float gprime_y = (-1 * (1 + currElement.conicConstantY) * intersect.y) /
                             (currElement.radiusY * currElement.radiusY *
                              std::sqrt(1 - term1 - term2));

            Float f = (intersect.x * intersect.x) / currElement.radiusX +
                      (intersect.y * intersect.y) / currElement.radiusY;
            Float g = 1 + std::sqrt(1 - term1 - term2);

            Float zprime_y = (fprime_y * g - gprime_y * f) / (g * g);
            Float zprime_x = (fprime_x * g - gprime_x * f) / (g * g);

            Vector3f v_x = Vector3f(1, 0, zprime_x);
            Vector3f v_y = Vector3f(0, 1, zprime_y);

            *n = Normalize(Cross(v_x, v_y));
            *n = FaceForward(*n, -r.d);

            return true;
        }

        // For debugging
        /*
            printf ("%5d [%.7f, %.7f] %.7f %+.7f %.7f\n",
            iter, x_lo, x_hi,
            root, root - root,
            x_hi - x_lo);
            */
    } while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fsolver_free(s);

    return false;
}

// Handy method to explicity solve for the z(x,y) at a given point (x,y),for the biconic
// SAG.
Float HumanEyeCamera::BiconicZ(Float x, Float y, LensElementEye currElement) const {
    Float f, g, g_term;
    Float Rx, Ry, Cx, Cy;
    Rx = currElement.radiusX;
    Ry = currElement.radiusY;
    Cx = currElement.conicConstantX;
    Cy = currElement.conicConstantY;

    f = (x * x) / Rx + (y * y) / Ry;
    g_term = 1 - (1 + Cx) * (x * x) / (Rx * Rx) - (1 + Cy) * (y * y) / (Ry * Ry);

    if (g_term < 0) {
        g_term = 0.001;
        Warning(
            "Encountered a complex value when solving for the z(x,y) of the biconic.");
    }

    g = 1 + sqrtf(g_term);

    return f / g;
}

// Given the mediumIndex, load up the right spectra from the ones read in through ior1,
// ior2, etc. Then find the corresponding IOR for the given ray wavelength.
Float HumanEyeCamera::lookUpIOR(int mediumIndex, const Ray &ray) const {
    Float n;

    // Standard media
    // If spectral renderer is used, then ray.wavelength will be given a value. Otherwise,
    // it will be a junk value. I'm not sure why it's not intialized to zero despite
    // having wavelength = 0 in the constructer. Anyhow, we can error check by looking for
    // values within the valid range.
    if ((std::abs(ray.wavelength) >= Lambda_min) &
        (std::abs(ray.wavelength) <= Lambda_max)) {
        // iorSpectra[mediumIndex-1].GetValueAtWavelength(ray.wavelength,&n);
        n = iorSpectra[mediumIndex - 1](ray.wavelength);
    } else {
        // iorSpectra[mediumIndex-1].GetValueAtWavelength(550,&n);
        n = iorSpectra[mediumIndex - 1](550);  // 550 nm
    }

    return n;
}

void HumanEyeCamera::diffractHURB(Point3f intersect, Float apertureRadius,
                                  const Float wavelength, const Vector3f oldDirection,
                                  Vector3f *newDirection) const {
    //        std::cout << "wavelength = " << wavelength << std::endl;

    double dist2Int = std::sqrt(intersect.x * intersect.x + intersect.y * intersect.y);
    Vector3f dirS = Normalize(Vector3f(intersect.x, intersect.y, 0));
    Vector3f dirL = Normalize(Vector3f(-1 * intersect.y, intersect.x, 0));
    Vector3f dirU = Vector3f(
        0, 0,
        1);  // Direction pointing normal to the aperture plane and toward the scene.

    double dist2EdgeS = apertureRadius - dist2Int;
    double dist2EdgeL = std::sqrt(apertureRadius * apertureRadius - dist2Int * dist2Int);

    // Calculate variance according to Freniere et al. 1999
    // If the scene is in meters, lensScaling = 0.001 and dist2Edge will be in meters.
    // if scene is in millimeters, lensScaling = 1 and dist2Edge will be in millimeters.
    // TL: Changed 2 to sqrt(2) to better match MTF's with airy disk
    // lensScaling = 0.001; // I'm not sure why this param is reset to be 0 here, I set it to be 0.001 again.
    // I think every scene we have now is using meters. 
    double sigmaS =
        atan(1 / (1.41 * dist2EdgeS * 2 * Pi / (wavelength * 1e-6 * 0.001)));
    double sigmaL =
        atan(1 / (1.41 * dist2EdgeL * 2 * Pi / (wavelength * 1e-6 * 0.001)));

    // Sample from bivariate gaussian
    double initS = 0;
    double initL = 0;
    double *noiseS = &initS;
    double *noiseL = &initL;
    gsl_ran_bivariate_gaussian(r, sigmaS, sigmaL, 0, noiseS, noiseL);

    // DEBUG:
    //        std::cout << "noiseS = " << *noiseS << std::endl;
    //        std::cout << "noiseL = " << *noiseL << std::endl;
    //        std::cout << *noiseS << " " << *noiseL << std::endl;

    // Decompose our original ray into dirS and dirL.
    Float projS = Dot(oldDirection, dirS) / Length(dirS);
    Float projL = Dot(oldDirection, dirL) / Length(dirL);
    Float projU = Dot(oldDirection, dirU) / Length(dirU);

    /*
        We have now decomposed the original, incoming ray into three orthogonal
        directions: directionS, directionL, and directionU.
        directionS is the direction along the shortest distance to the aperture
        edge.
        directionL is the orthogonal direction to directionS in the plane of the
        aperture.
        directionU is the direction normal to the plane of the aperture, pointing
        toward the scene.
        To orient our azimuth and elevation directions, imagine that the
        S-U-plane forms the "ground plane." "Theta_x" in the Freniere paper is
        therefore the deviation in the azimuth and "Theta_y" is the deviation in
        the elevation.
        */

    // Calculate current azimuth and elevation angles
    Float thetaA = atan(projS / projU);                                     // Azimuth
    Float thetaE = atan(projL / std::sqrt(projS * projS + projU * projU));  // Elevation

    // Deviate the angles
    thetaA = thetaA + *noiseS;
    thetaE = thetaE + *noiseL;

    // Recalculate the ray direction
    // Remember the ray direction is normalized, so it should have length = 1
    Float newProjL = sin(thetaE);
    Float newProjSU = cos(thetaE);
    Float newProjS = newProjSU * sin(thetaA);
    Float newProjU = newProjSU * cos(thetaA);

    // Add up the new projections to get a new direction
    *newDirection = Normalize(newProjS * dirS + newProjL * dirL + newProjU * dirU);
}

HumanEyeCamera *HumanEyeCamera::Create(const ParameterDictionary &parameters,
                                       const CameraTransform &cameraTransform, Film film,
                                       Medium medium, const FileLoc *loc,
                                       Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    // HumanEye specific parameters
    // The lens file should include  columns for:
    // [radiusX radiusY thickness mediumIndex semiDiameter conicConstantX conicConstantY].
    // The surfaces should be ordered from the world to the retina.
    // (E.g. The last surface in the text file is the retina.)
    std::string specfile = parameters.GetOneString("specfile", "");
    if (specfile == "") {
        // Try looking for "lensfile" instead of "specfile"
        specfile = parameters.GetOneString("lensfile", "");
        // If it's still empty, we have an issue.
        if (specfile == "") {
            Error("No lens specification file supplied!\n");
        }
    }

    // Read lookuptable if available
    std::string lookupTableFile = parameters.GetOneString("lookuptable", "");
    pstd::vector<Point3f> lookupTable(alloc);
    if (lookupTableFile == "") {
        lookupTable =
            pstd::vector<Point3f>(alloc);  // Empty lookup table means it will not be used
    } else {
        // Read out lookup table

        printf("Lookup table read: %s \n", lookupTableFile.c_str());

        auto endsWith = [](const std::string &str, const std::string &suffix) {
            return str.size() >= suffix.size() &&
                   0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
        };
        if (!endsWith(lookupTableFile, ".json")) {
            Error("Invalid format for lookuptable specification file \"%s\".",
                  lookupTableFile.c_str());
            return nullptr;
        }
        // read lens json file
        std::ifstream i(lookupTableFile);
        json j;
        if (i && (i >> j)) {
            // assert(j.is_object())
            // j["name"]
            // j["description"]
            auto jsurfaces = j["table"];

            auto toVec2 = [](json val) {
                if (val.is_number()) {
                    return Vector2f{(Float)val, (Float)val};
                } else if (val.is_array() && val.size() == 2) {
                    return Vector2f{val[0], val[1]};
                }
                return Vector2f();  // Default value
            };

            auto toPoint3f = [](json val) {
                if (val.is_number()) {
                    return Point3f{(Float)val, (Float)val, (Float)val};
                } else if (val.is_array() && val.size() == 3) {
                    return Point3f{val[0], val[1], val[2]};
                }
                return Point3f();  // Default value
            };

            auto toVec2i = [](json val) {
                if (val.is_number()) {
                    return Vector2i{(int)val, (int)val};
                } else if (val.is_array() && val.size() == 2) {
                    return Vector2i{val[0], val[1]};
                }
                return Vector2i();  // Default value
            };

            //   Inirialize Array2D
            //Point2i nbRowCols = Point2i(toVec2i(j["rowcols"]));
            // Initialize the lookuptable with the total number of points;
            int nbPoints = (int) j["numberofpoints"];
            lookupTable =
                pstd::vector<Point3f>(nbPoints,alloc);  // Empty lookup table means it will not be used

            
            if (jsurfaces.is_array() && jsurfaces.size() > 0) {
                for (auto jsurf : jsurfaces) {
                    int index = (int) jsurf["index"];
                    Point3f startingPoint = toPoint3f(jsurf["point"]);

                    printf("index: %i \n", index);
                    printf("STARTINGPOINT: %f,%f,%f \n", startingPoint.x, startingPoint.y,
                           startingPoint.z);
                    lookupTable[index] = startingPoint;
                }
            } else {
                Error("Error,  not a valid table in lookuptable file \"%s\".",
                      lookupTableFile.c_str());
                return nullptr;
            }

        } else {
            Error("Error reading lookup table file \"%s\".", lookupTableFile.c_str());
            return nullptr;
        }
    }

    // These are additional parameters we need to specify in the PBRT file.
    Float pupilDiameter = parameters.GetOneFloat("pupilDiameter", 4.0);      // mm
    Float retinaDistance = parameters.GetOneFloat("retinaDistance", 16.32);  // mm
    Float retinaRadius = parameters.GetOneFloat("retinaRadius", 12);         // mm
    Float retinaSemiDiam = parameters.GetOneFloat("retinaSemiDiam", 4);      // mm

    // Check for IORspectra slots
    Spectrum ior1 =
        parameters.GetOneSpectrum("ior1", nullptr, SpectrumType::Unbounded, alloc);
    Spectrum ior2 =
        parameters.GetOneSpectrum("ior2", nullptr, SpectrumType::Unbounded, alloc);
    Spectrum ior3 =
        parameters.GetOneSpectrum("ior3", nullptr, SpectrumType::Unbounded, alloc);
    Spectrum ior4 =
        parameters.GetOneSpectrum("ior4", nullptr, SpectrumType::Unbounded, alloc);
    Spectrum ior5 =
        parameters.GetOneSpectrum("ior5", nullptr, SpectrumType::Unbounded, alloc);
    Spectrum ior6 =
        parameters.GetOneSpectrum("ior6", nullptr, SpectrumType::Unbounded, alloc);

    // Put all the spectra into a single vector.
    pstd::vector<Spectrum> iorSpectra;
    iorSpectra.push_back(ior1);
    iorSpectra.push_back(ior2);
    iorSpectra.push_back(ior3);
    iorSpectra.push_back(ior4);
    iorSpectra.push_back(ior5);
    iorSpectra.push_back(ior6);

    // Flags
    bool flipRad = parameters.GetOneBool("flipLensRadius", 0.0);
    bool mmUnits = parameters.GetOneBool("mmUnits", 1.0);
    
    bool diffractionEnabled = parameters.GetOneBool("diffractionEnabled", false);

    // Scale units depending on the units of the scene
    Float lensScaling = 0.001;
    if (mmUnits) {
        lensScaling = 1;
    }

    pupilDiameter = pupilDiameter * lensScaling;
    retinaDistance = retinaDistance * lensScaling;
    retinaRadius = retinaRadius * lensScaling;
    retinaSemiDiam = retinaSemiDiam * lensScaling;
    // -------------------------
    // --- Read in lens file ---
    // -------------------------

    /*
        Note on sign convention:
        For the realistic eye code, we have gone with the Zemax convention for the lens
       radius. A positive lens radius means the center of the spherical lens is in the
       positive direction (toward the scene) relative to the location of the lenes. This
       is different than the other camera classes where the convention is flipped. If
       necessary, one can turn on the [flipLensRadius] flag.
        */

    // Find the complete path for the specfile
    std::string lensFile = ResolveFilename(specfile);

    // Check to see if there is valid input in the lens file.
    std::vector<Float> lensParameters = ReadFloatFile(lensFile);
    if (lensParameters.empty()) {
        Error(loc, "Error reading lens specification file \"%s\".", lensFile);
        return nullptr;
    }

    // The lens file should include  columns for [radiusX radiusY thickness mediumIndex
    // semiDiameter conicConstantX conicConstantY]. Let's check then that the file is a
    // multiple of 7 (not including the effective focal length at the top)
    if ((lensParameters.size() - 1) % 7 != 0) {
        Error("Wrong number of float values in lens file! Did you forget to specify the "
              "focal length? Is this a lens file with biconic surfaces? Do you have a "
              "carriage return at the end of the data?");
        return nullptr;
    }

    Float effectiveFocalLength =
        lensParameters[0] * lensScaling;  // Read the effective focal length.

    Float frontThickness = 0;
    pstd::vector<LensElementEye> eyeInterfacesData;

    for (int i = 1; i < lensParameters.size(); i += 7) {
        LensElementEye currentLensEl;
        currentLensEl.radiusX = lensParameters[i] * lensScaling;
        currentLensEl.radiusY = lensParameters[i + 1] * lensScaling;
        currentLensEl.thickness = lensParameters[i + 2] * lensScaling;
        currentLensEl.mediumIndex = lensParameters[i + 3];
        currentLensEl.semiDiameter = lensParameters[i + 4] * lensScaling;
        currentLensEl.conicConstantX = lensParameters[i + 5];
        currentLensEl.conicConstantY = lensParameters[i + 6];

        frontThickness += currentLensEl.thickness;

        // Note: Zemax and PBRT-spectral seem to have different conventions for what the
        // positive and negative sign of the radius is. In Zemax, a positive radius means
        // that the center of lens sphere is directed toward the positive Z-axis, and vice
        // versa. In previous PBRT-spectral iterations, this was flipped. Here I've
        // rewritten the lens tracing code to go with the Zemax convention, however to be
        // backward compatible we might want to have this ability to flip the radii.
        if (flipRad) {
            currentLensEl.radiusX = -1 * currentLensEl.radiusX;
            currentLensEl.radiusY = -1 * currentLensEl.radiusY;
            currentLensEl.conicConstantX = -1 * currentLensEl.conicConstantX;
            currentLensEl.conicConstantY = -1 * currentLensEl.conicConstantY;
            Warning("Flipping lens radius & conic convention.");
        }

        // A radius of zero in BOTH x and y directions indicates an aperture. We should be
        // careful of this though, since sometimes we may want to define a flat surface...
        // If the surface is an aperture, we set it's size to be equal to the pupil
        // diameter specified.
        if (currentLensEl.radiusX == 0 && currentLensEl.radiusY == 0) {
            currentLensEl.semiDiameter = pupilDiameter / 2;
        } else {
            // We have to do a semi-diameter check here. As we change accommodation, we
            // also change the radius of curvature, and we don't want the semi-diameter to
            // be bigger than the radius. This manifests as a square root of a negative
            // number in equation 1 on Einighammer et al. 2009.
            // TODO: This check is sort of hack-y, is there a better mathematical way to
            // do this? Note: This calculation should be done in millimeters.
            float smallerR = std::min(currentLensEl.radiusX, currentLensEl.radiusY) *
                             (1 / lensScaling);
            float biggerK =
                std::max(currentLensEl.conicConstantX, currentLensEl.conicConstantY);
            if (currentLensEl.semiDiameter * (1 / lensScaling) *
                    currentLensEl.semiDiameter * (1 / lensScaling) * (1 + biggerK) /
                    (smallerR * smallerR) >
                1.0f) {
                currentLensEl.semiDiameter =
                    0.95 *
                    std::sqrt(
                        (smallerR * smallerR /
                         (1 + biggerK)));  // 0.95 is to add some buffer zone, since rays
                                           // act very strangely when they get too close
                                           // to the edge of the conical surface.
            }
        }

        eyeInterfacesData.push_back(currentLensEl);
    }

    // Check thickness of last element. It should be zero, since we use the "retina
    // distance" parameter for this final "thickness."
    if (eyeInterfacesData[eyeInterfacesData.size() - 1].thickness != 0) {
        Error("Thickness of lens element closest to zero must be zero. Define thickness "
              "in 'retinaDistance' parameter instead.");
    }

    return alloc.new_object<HumanEyeCamera>(
        cameraBaseParameters, eyeInterfacesData, pupilDiameter, retinaDistance,
        retinaRadius, retinaSemiDiam, iorSpectra, lookupTable, diffractionEnabled, alloc);
}

// ------------------------------------------OMNI CAMERA STARTS
// HERE--------------------------------

Float sgn(Float val) {
    return Float((Float(0) < val) - (val < Float(0)));
}
static Float computeZOfLensElement(Float r,
                                   const OmniCamera::LensElementInterface &element) {
    if (element.asphericCoefficients.size() == 0) {
        Float R = std::abs(element.curvatureRadius.x);
        Float d = std::sqrt((R * R) - (r * r));
        Float z = R - d;
        return sgn(element.curvatureRadius.x) * z;
    } else {
        r *= 1000.0f;
        Float c = (Float)1.0 / (element.curvatureRadius.x * 1000.0f);
        Float kappa = element.conicConstant.x * 1000.0f;
        Float r2 = r * r;
        Float r2i = r2;
        // printf("r^2 = %g\n", r2);
        //  Equation 1 from https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.12953
        Float z = (c * r2) / (1.0 + std::sqrt(1.0 - ((1.0 + kappa) * c * c * r2)));
        Float zBefore = z;
        for (int j = 0; j < element.asphericCoefficients.size(); ++j) {
            r2i *= r2;
            int i = j + 2;  // First coefficient is at 2i=4, i = 2
            Float A2i = element.asphericCoefficients[j];
            z += (A2i * r2i);
        }
        return z / 1000.0f;
    }
}

// OmniCamera Method Definitions
OmniCamera::OmniCamera(CameraBaseParameters baseParameters,
                       pstd::vector<OmniCamera::LensElementInterface> &lensInterfaceData,
                       Float focusDistance, Float filmDistance, bool caFlag,
                       bool diffractionEnabled,
                       pstd::vector<OmniCamera::LensElementInterface> &microlensData,
                       Vector2i microlensDims,
                       pstd::vector<Vector2f> &microlensOffsetsVec,
                       Float microlensSensorOffset, int microlensSimulationRadiusVar,
                       Float setApertureDiameter, Image apertureImage, Allocator alloc)
    : CameraBase(baseParameters),
      elementInterfaces(alloc),
      microlensElementInterfaces(alloc),
      microlensOffsets(alloc),
      exitPupilBounds(alloc),
      caFlag(caFlag),
      diffractionEnabled(diffractionEnabled),
      apertureImage(std::move(apertureImage)) {
    // Compute film's physical extent
    Float aspect = (Float)film.FullResolution().y / (Float)film.FullResolution().x;
    Float diagonal = film.Diagonal();
    Float x = std::sqrt(Sqr(diagonal) / (1 + Sqr(aspect)));
    Float y = aspect * x;
    physicalExtent = Bounds2f(Point2f(-x / 2, -y / 2), Point2f(x / 2, y / 2));

    // Initialize _elementInterfaces_ for camera
    elementInterfaces = lensInterfaceData;

    for (LensElementInterface &element : elementInterfaces) {
        // Compute bounding planes for the aspherics
        if (element.asphericCoefficients.size() > 0) {
            // Compute sampled min and max of Z (making sure to sample extrema in r)
            // Compute largest change, use that to conservatively expand bounds.
            Float maxR = element.apertureRadius.x;
            Float zMin = computeZOfLensElement(maxR, element);
            Float zMax = zMin;
            int sampleCount = 128;
            Float stepR = maxR / sampleCount;
            Float lastZ = computeZOfLensElement(0.0, element);
            Float largestDiff = 0.0;
            for (Float R = 0.0; R < maxR; R += stepR) {
                Float newZ = computeZOfLensElement(R, element);
                zMin = std::min(zMin, newZ);
                zMax = std::max(zMax, newZ);

                Float zDiff = std::abs(lastZ - newZ);
                largestDiff = std::max(largestDiff, zDiff);
                lastZ = newZ;
            }
            element.zMin = zMin - largestDiff;
            element.zMax = zMax + largestDiff;
        }
    }

    if (microlensData.size() > 0) {
        microlens.elementInterfaces =
            pstd::vector<OmniCamera::LensElementInterface>(alloc);

        microlens.elementInterfaces = microlensData;

        microlens.offsets = pstd::vector<pbrt::Vector2f>(alloc);
        microlens.offsets = microlensOffsetsVec;
        microlens.dimensions = microlensDims;
        microlens.offsetFromSensor = microlensSensorOffset;
        microlens.simulationRadius = microlensSimulationRadius;

        // TG New code without putting vectors in struct
        microlensElementInterfaces = microlensData;
        microlensOffsets = microlensOffsetsVec;
        microlensDimensions = microlensDims;
        microlensSimulationRadius = microlensSimulationRadiusVar;
        microlensOffsetFromSensor = microlensSensorOffset;
    }

    if (filmDistance == 0) {
        // Float fb = FocusBinarySearch(focusDistance); // may be useful, disable for now
        // Compute lens--film distance for given focus distance
        elementInterfaces.back().thickness = FocusThickLens(focusDistance);
    } else {
        // Use given film distance
        elementInterfaces.back().thickness = filmDistance;
    }
    // Compute exit pupil bounds at sampled points on the film
    int nSamples = 64;
    exitPupilBounds.resize(nSamples);
    ParallelFor(0, nSamples, [&](int i) {
        Float r0 = (Float)i / nSamples * film.Diagonal() / 2;
        Float r1 = (Float)(i + 1) / nSamples * film.Diagonal() / 2;
        exitPupilBounds[i] = BoundExitPupil(r0, r1);
    });

    // Compute minimum differentials for _OmniCamera_
    FindMinimumDifferentials(this);
}

struct aspheric_params {
    Float c;
    Float kappa;
    std::vector<Float> *A;
    Ray ray;
};

double AsphericIntersect(double t, void *params) {
    struct aspheric_params *p;
    p = (struct aspheric_params *)params;

    Point3f X = p->ray(t);
    // Compute distance from intersect to lens along z-axis

    const Float c = p->c;
    const Float kappa = p->kappa;
    const Float r = std::sqrt(X.x * X.x + X.y * X.y);
    Float r2 = r * r;
    Float r2i = r2;
    // Equation 1 from https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.12953
    Float z = (c * r2) / (1.0 + std::sqrt(1.0 - ((1.0 + kappa) * c * c * r2)));
    Float zBefore = z;
    for (int j = 0; j < p->A->size(); ++j) {
        r2i *= r2;
        int i = j + 2;  // First coefficient is at 2i=4, i = 2
        Float A2i = (*p->A)[j];
        z += (A2i * r2i);
    }
    // if (std::abs(p->ray.d.x) > std::abs(p->ray.d.z)) printf("t, z, X.z: %g, %g, %g\n",
    // t, z, X.z);
    return X.z - z;
}

Float OmniCamera::TToBackLens(const Ray &rCamera,
                              const pstd::vector<LensElementInterface> &elementInterfaces,
                              const Transform LensFromCamera = Scale(1, 1, -1),
                              const ConvexQuadf &bounds = ConvexQuadf()) const {
    // Transform _rCamera_ from camera to lens system space
    Ray rLens = LensFromCamera(rCamera);

    const LensElementInterface &element = elementInterfaces[elementInterfaces.size() - 1];
    // Compute intersection of ray with lens element
    Float t;
    Normal3f n;
    bool isStop = false;
    Float elementZ = -element.thickness;
    IntersectResult result = TraceElement(element, rLens, elementZ, t, n, isStop, bounds);
    if (result != HIT)
        return Infinity;
    return t;
}

bool IntersectAsphericalElement(const OmniCamera::LensElementInterface &element,
                                const Float elementZ, const Ray &r, Float *tHit,
                                Normal3f *n) {
    // Computation is done in meters
    const Float c = 1.0f / (element.curvatureRadius.x * 1000.0f);
    const Float kappa = element.conicConstant.x * 1000.0f;
    std::vector<Float> A = element.asphericCoefficients;
    // This code lets us find the intersection with an aspheric surface. At tHit, the ray
    // will intersect the surface. Therefore: If (x,y,z) = ray_origin + thit *
    // ray_direction then: z - u(x,y) = 0 where u(x,y) is the SAG of the surface, defined
    // in Eq.1 of Einighammer et al. 2009 and in the Zemax help page under biconic
    // surfaces. We can use this fact to solve for thit. If the surface is not a sphere,
    // this is a messy polynomial. So instead we use a numeric root-finding method (Van
    // Wijingaarden-Dekker-Brent's Method.) This method is available in the GSL library.

    // Move ray to object(lens) space.
    Ray lensRay = r;
    lensRay.o = (lensRay.o - Vector3f(0, 0, elementZ)) * 1000.0f;
    int status;
    int iter = 0, max_iter = 100;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double root = 0;

    gsl_function F;

    struct aspheric_params params = {c, kappa, &A, lensRay};
    F.function = &AsphericIntersect;
    F.params = &params;

    // TODO: Use intersection with tight bounding spherical geometry to define the initial
    // bounds Recommended by
    // https://graphics.tudelft.nl/Publications-new/2016/JKLEL16/pdf.pdf We currently use
    // planes as proxies because its easier...
    // TODO: Make sure this works for rays from scene to sensor as well
    double t_lo = (element.zMax * 1000.0f - lensRay.o.z) / lensRay.d.z;
    double t_hi = (element.zMin * 1000.0f - lensRay.o.z) / lensRay.d.z;
    // printf("Before aperture t_lo, t_hi: %g %g\n", t_lo, t_hi);
    {
        // (ox + t dx)^2 + (oy + t dy)^2 - r^2 = 0
        // (dx^2 + dy^2)t^2 = (2dxox+2dyoy)t + (ox^2 + oy^2 - r^2)
        Float dx = lensRay.d.x;
        Float dy = lensRay.d.y;
        Float ox = lensRay.o.x;
        Float oy = lensRay.o.y;
        Float R = element.apertureRadius.x * 1000.0f;
        bool startsInAperture = (ox * ox + oy * oy) <= R * R;
        Float t_a0, t_a1;
        Float A = (dx * dx + dy * dy);
        Float B = 2 * (dx * ox + dy * oy);
        Float C = ox * ox + oy * oy - R * R;
        Quadratic(A, B, C, &t_a0, &t_a1);
        bool bothNegative = (t_a0 < 0) && (t_a1 < 0);
        bool bothPositive = (t_a0 >= 0) && (t_a1 >= 0);
        Float minT = std::min(t_a0, t_a1);
        Float maxT = std::max(t_a0, t_a1);
        Float nonnegativeT = (t_a0 < 0) ? t_a1 : t_a0;  // only valid if !bothNegative
        if (startsInAperture) {
            if (!bothNegative) {
                if (bothPositive) {
                    t_hi = std::min(t_hi, (double)minT);
                } else {
                    t_hi = std::min(t_hi, (double)nonnegativeT);
                }
            }
        } else {
            if (bothNegative) {
                t_lo = t_hi;  // No valid solutions
            } else {
                if (bothPositive) {
                    t_lo = std::max(t_lo, (double)minT);
                    t_hi = std::min(t_hi, (double)maxT);
                } else {
                    t_lo = std::max(t_lo, (double)nonnegativeT);
                }
            }
        }
    }

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc(T);
    // We'll handle errors
    gsl_set_error_handler_off();
    status = gsl_root_fsolver_set(s, &F, t_lo, t_hi);
    if (status != 0) {
        // Ray probably does not intersect. This might depend on the t_hi set above, i.e.
        // if it's too small OR too large. TODO: Can we check this?
        gsl_root_fsolver_free(s);
        return false;
    }

    do {
        iter++;
        gsl_root_fsolver_iterate(s);
        root = gsl_root_fsolver_root(s);
        t_lo = gsl_root_fsolver_x_lower(s);
        t_hi = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(t_lo, t_hi, 0, 0.00001);

        if (status == GSL_SUCCESS) {
            gsl_root_fsolver_free(s);

            *tHit = root / 1000.0f;  // Convert back to meters
            Point3f intersect = r(*tHit);

            // Compute normal through finite differences TODO: compute closed form?
            const Float EPS = 0.000001f;

            auto offsetOnLens = [intersect, element, elementZ](Vector2f off) {
                Vector2f off2 = Vector2f(intersect.x + off.x, intersect.y + off.y);
                return Point3f(off2.x, off2.y,
                               computeZOfLensElement(Length(off2), element) + elementZ);
            };

            Vector3f dx = offsetOnLens({EPS, 0}) - offsetOnLens({-EPS, 0});
            Vector3f dy = offsetOnLens({0, EPS}) - offsetOnLens({0, -EPS});

            *n = Normal3f(Normalize(Cross(dx, dy)));
            *n = FaceForward(*n, -r.d);

            return true;
        }

    } while (status == GSL_CONTINUE && iter < max_iter);
    printf("Couldn't find root!\n");
    gsl_root_fsolver_free(s);
    return false;
}

void OmniCamera::diffractHURB(Ray &rLens, const LensElementInterface &element,
                              const Float t) const {
    // Convert the rays in this element space
    auto invTransform = Inverse(element.transform);
    Ray rElement = invTransform(rLens);

    Point3f intersect = rElement(t);
    Float apertureRadius = element.apertureRadius.x;
    Float wavelength = rElement.wavelength;
    Vector3f oldDirection = rElement.d;

    // ZLY: Directly copied from realisticEye of pbrt-v3
    // Modified wavelength unit conversion as I think it is always assumed to be in unit
    // of meters std::cout << "wavelength = " << wavelength << std::endl;
    Float dist2Int = std::sqrt(intersect.x * intersect.x + intersect.y * intersect.y);
    Vector3f dirS = Normalize(Vector3f(intersect.x, intersect.y, 0));
    Vector3f dirL = Normalize(Vector3f(-1 * intersect.y, intersect.x, 0));
    Vector3f dirU = Vector3f(
        0, 0,
        1);  // Direction pointing normal to the aperture plane and toward the scene.

    Float dist2EdgeS = apertureRadius - dist2Int;
    Float dist2EdgeL = std::sqrt(apertureRadius * apertureRadius - dist2Int * dist2Int);

    // Calculate variance according to Freniere et al. 1999
    double sigmaS = atan(1 / (1.41 * dist2EdgeS * 2 * Pi / (wavelength * 1e-9)));
    double sigmaL = atan(1 / (1.41 * dist2EdgeL * 2 * Pi / (wavelength * 1e-9)));

    // Sample from bivariate gaussian
    double initS = 0;
    double initL = 0;
    double *noiseS = &initS;
    double *noiseL = &initL;

    gsl_ran_bivariate_gaussian(r, sigmaS, sigmaL, 0, noiseS, noiseL);

    // DEBUG:
    //        std::cout << "noiseS = " << *noiseS << std::endl;
    //        std::cout << "noiseL = " << *noiseL << std::endl;
    //        std::cout << *noiseS << " " << *noiseL << std::endl;

    // Decompose our original ray into dirS and dirL.
    Float projS = Dot(oldDirection, dirS) / Length(dirS);
    Float projL = Dot(oldDirection, dirL) / Length(dirL);
    Float projU = Dot(oldDirection, dirU) / Length(dirU);

    /*
     We have now decomposed the original, incoming ray into three orthogonal
     directions: directionS, directionL, and directionU.
     directionS is the direction along the shortest distance to the aperture
     edge.
     directionL is the orthogonal direction to directionS in the plane of the
     aperture.
     directionU is the direction normal to the plane of the aperture, pointing
     toward the scene.
     To orient our azimuth and elevation directions, imagine that the
     S-U-plane forms the "ground plane." "Theta_x" in the Freniere paper is
     therefore the deviation in the azimuth and "Theta_y" is the deviation in
     the elevation.
     */

    // Calculate current azimuth and elevation angles
    Float thetaA = atan(projS / projU);                                     // Azimuth
    Float thetaE = atan(projL / std::sqrt(projS * projS + projU * projU));  // Elevation

    // Deviate the angles
    thetaA = thetaA + *noiseS;
    thetaE = thetaE + *noiseL;

    // Recalculate the ray direction
    // Remember the ray direction is normalized, so it should have length = 1
    Float newProjL = sin(thetaE);
    Float newProjSU = cos(thetaE);
    Float newProjS = newProjSU * sin(thetaA);
    Float newProjU = newProjSU * cos(thetaA);

    // Add up the new projections to get a new direction
    rElement.d = Normalize(-newProjS * dirS - newProjL * dirL - newProjU * dirU);
    rLens = element.transform(rElement);
}

OmniCamera::IntersectResult OmniCamera::TraceElement(
    const LensElementInterface &element, const Ray &rLens, const Float &elementZ,
    Float &t, Normal3f &n, bool &isStop,
    const ConvexQuadf &bounds = ConvexQuadf()) const {
    auto invTransform = Inverse(element.transform);
    Ray rElement = invTransform(rLens);
    if (isStop) {
        // The refracted ray computed in the previous lens element
        // interface may be pointed "backwards" in some
        // extreme situations; in such cases, 't' becomes negative.
        t = (elementZ - rElement.o.z) / rElement.d.z;
        if (rElement.d.z == 0.0 || t < 0)
            return MISS;
    } else {
        if (0) {
            // if (element.asphericCoefficients.size() > 0) {
            // if (!IntersectAsphericalElement(element, elementZ, rElement, &t, &n))
            //     return MISS; // Aspherical is not working for GPU now --Zhenyi
        } else {
            Float radius = element.curvatureRadius.x;
            Float zCenter = elementZ + element.curvatureRadius.x;
            if (!IntersectSphericalElement(radius, zCenter, rElement, &t, &n))
                return MISS;
        }
    }
    CHECK_GE(t, 0);
    // Transform the normal back into the original space.
    n = element.transform(n);

    // Test intersection point against element aperture
    Point3f pHit = rElement(t);
    Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
    Float apertureRadius2 = element.apertureRadius.x * element.apertureRadius.x;
    Point2f pHit2f;
    pHit2f.x = pHit.x;
    pHit2f.y = pHit.y;
    // std::cout << "- " << element.apertureRadius.x << "," << element.apertureRadius.y <<
    // "\n";
    if (r2 > apertureRadius2)
        return CULLED_BY_APERTURE;
    if (!bounds.Contains(Point2f(pHit2f))) {
        return CULLED_BY_APERTURE;
    }
    return HIT;
};

Float OmniCamera::TraceLensesFromFilm(
    const Ray &rCamera, const pstd::vector<LensElementInterface> &interfaces, Ray *rOut,
    const Transform LensFromCamera = Scale(1, 1, -1),
    const ConvexQuadf &bounds = ConvexQuadf()) const {
    Float elementZ = 0, weight = 1;

    // Transform _rCamera_ from camera to lens system space
    // const Transform LensFromCamera = Scale(1, 1, -1);
    Ray rLens = LensFromCamera(rCamera);

    if (rOut) {
        rLens.wavelength = rOut->wavelength;
    } else {
        rLens.wavelength = 550;
    }

    for (int i = interfaces.size() - 1; i >= 0; --i) {
        const LensElementInterface &element = interfaces[i];
        // Update ray from film accounting for interaction with _element_
        elementZ -= element.thickness;
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        bool isStop = (element.curvatureRadius.x == 0);
        // bool isStop = (element.curvatureRadius.x == 0) &&
        // (element.asphericCoefficients.size() == 0); Omni Camera method
        IntersectResult result =
            TraceElement(element, rLens, elementZ, t, n, isStop, bounds);
        if (result != HIT)
            return 0;

        /* PBRT Method
        if (isStop) {
            // Compute _t_ at plane of aperture stop
            t = (elementZ - rLens.o.z) / rLens.d.z;
            if (t < 0)
                return 0;

        } else {
            // Intersect ray with element to compute _t_ and _n_
            Float radius = element.curvatureRadius.x;
            Float zCenter = elementZ + element.curvatureRadius.x;
            if (!IntersectSphericalElement(radius, zCenter, rLens, &t, &n))
                return 0;
        }
        DCHECK_GE(t, 0);
        */

        // Test intersection point against element aperture
        Point3f pHit = rLens(t);
        if (isStop && apertureImage) {
            // Check intersection point against _apertureImage_
            Point2f uv((pHit.x / element.apertureRadius.x + 1) / 2,
                       (pHit.y / element.apertureRadius.x + 1) / 2);
            weight = apertureImage.BilerpChannel(uv, 0, WrapMode::Black);
            if (weight == 0)
                return 0;

        } else {
            // Check intersection point against spherical aperture
            if (Sqr(pHit.x) + Sqr(pHit.y) > Sqr(element.apertureRadius.x))
                return 0;
        }
        // rLens.o = pHit;

        // Update ray path for element interface interaction
        if (!isStop) {
            // Update ray path for element interface interaction
            rLens.o = rLens(t);
            Vector3f w;

            // Thomas Goossens spectral modification using SampeldSpectrum
            // Float spectralEtaI;
            // element.etaspectral.GetValueAtWavelength(rLens.wavelength,&spectralEtaI);

            // Float spectralEtaT;
            // if (i > 0){
            //     elementInterfaces[i -
            //     1].etaspectral.GetValueAtWavelength(rLens.wavelength,&spectralEtaT);
            // }

            // Float eta_i = spectralEtaI;
            // Float eta_t = (i > 0 && spectralEtaT != 0)
            //                  ? spectralEtaT
            //                  : 1;

            Float eta_i = element.eta;
            Float eta_t = (i > 0 && elementInterfaces[i - 1].eta != 0)
                              ? elementInterfaces[i - 1].eta
                              : 1;

            // Added by Trisha and Zhenyi (5/18)
            if (caFlag && (rLens.wavelength >= 400) && (rLens.wavelength <= 700)) {
                if (eta_i != 1)
                    eta_i = (rLens.wavelength - 550) * -.04 / (300) + eta_i;
                if (eta_t != 1)
                    eta_t = (rLens.wavelength - 550) * -.04 / (300) + eta_t;
            }

            if (!Refract(Normalize(-rLens.d), n, eta_t / eta_i, nullptr, &w))
                return 0;
            rLens.d = w;
        } else {  // We are add stop, so we optionally enable HURB diffraction
            if (diffractionEnabled) {
                // DEBUG: Check direction did change
                // std::cout << "rLens.d = (" << rLens.d.x << "," << rLens.d.y << "," <<
                // rLens.d.z << ")" << std::endl; Adjust ray direction using HURB
                // diffraction
                // diffractHURB(rLens, element, t);
            }
        }
    }
    // Transform _rLens_ from lens system space back to camera space
    if (rOut) {
        *rOut = Inverse(LensFromCamera)(rLens);
        // *rOut = Ray(Point3f(rLens.o.x, rLens.o.y, -rLens.o.z),
        //             Vector3f(rLens.d.x, rLens.d.y, -rLens.d.z), rLens.time);
    }

    return weight;
}

void OmniCamera::ComputeCardinalPoints(Ray rIn, Ray rOut, Float *pz, Float *fz) {
    Float tf = -rOut.o.x / rOut.d.x;
    *fz = -rOut(tf).z;
    Float tp = (rIn.o.x - rOut.o.x) / rOut.d.x;
    *pz = -rOut(tp).z;
}

void OmniCamera::ComputeThickLensApproximation(Float pz[2], Float fz[2]) const {
    // Find height $x$ from optical axis for parallel rays
    Float x = .001f * film.Diagonal();

    // Compute cardinal points for film side of lens system
    Ray rScene(Point3f(x, 0, LensFrontZ() + 1), Vector3f(0, 0, -1));
    Ray rFilm;
    if (!TraceLensesFromScene(rScene, &rFilm))
        ErrorExit("Unable to trace ray from scene to film for thick lens "
                  "approximation. Is aperture stop extremely small?");
    ComputeCardinalPoints(rScene, rFilm, &pz[0], &fz[0]);

    // Compute cardinal points for scene side of lens system
    rFilm = Ray(Point3f(x, 0, LensRearZ() - 1), Vector3f(0, 0, 1));
    if (TraceLensesFromFilm(rFilm, elementInterfaces, &rScene) == 0)
        ErrorExit("Unable to trace ray from film to scene for thick lens "
                  "approximation. Is aperture stop extremely small?");
    ComputeCardinalPoints(rFilm, rScene, &pz[1], &fz[1]);
}

Float OmniCamera::FocusThickLens(Float focusDistance) {
    Float pz[2], fz[2];
    ComputeThickLensApproximation(pz, fz);
    LOG_VERBOSE("Cardinal points: p' = %f f' = %f, p = %f f = %f.\n", pz[0], fz[0], pz[1],
                fz[1]);
    LOG_VERBOSE("Effective focal length %f\n", fz[0] - pz[0]);
    // Compute translation of lens, _delta_, to focus at _focusDistance_
    Float f = fz[0] - pz[0];
    Float z = -focusDistance;
    Float c = (pz[1] - z - pz[0]) * (pz[1] - z - 4 * f - pz[0]);
    if (c <= 0)
        ErrorExit("Coefficient must be positive. It looks focusDistance %f "
                  " is too short for a given lenses configuration",
                  focusDistance);
    Float delta = (pz[1] - z + pz[0] - std::sqrt(c)) / 2;

    return elementInterfaces.back().thickness + delta;
}

Float OmniCamera::FocusDistance(Float filmDistance) {
    // Find offset ray from film center through lens
    Bounds2f bounds = BoundExitPupil(0, .001 * film.Diagonal());

    const std::array<Float, 3> scaleFactors = {0.1f, 0.01f, 0.001f};
    Float lu = 0.0f;

    Ray ray;

    // Try some different and decreasing scaling factor to find focus ray
    // more quickly when `aperturediameter` is too small.
    // (e.g. 2 [mm] for `aperturediameter` with wide.22mm.dat),
    bool foundFocusRay = false;
    for (Float scale : scaleFactors) {
        lu = scale * bounds.pMax[0];
        if (TraceLensesFromFilm(Ray(Point3f(0, 0, LensRearZ() - filmDistance),
                                    Vector3f(lu, 0, filmDistance)),
                                elementInterfaces, &ray)) {
            foundFocusRay = true;
            break;
        }
    }

    if (!foundFocusRay) {
        Error("Focus ray at lens pos(%f,0) didn't make it through the lenses "
              "with film distance %f?!??\n",
              lu, filmDistance);
        return Infinity;
    }

    // Compute distance _zFocus_ where ray intersects the principal axis
    Float tFocus = -ray.o.x / ray.d.x;
    Float zFocus = ray(tFocus).z;
    if (zFocus < 0)
        zFocus = Infinity;
    return zFocus;
}

Bounds2f OmniCamera::BoundExitPupil(Float filmX0, Float filmX1) const {
    Bounds2f pupilBounds;
    // Sample a collection of points on the rear lens to find exit pupil
    const int nSamples = 1024 * 1024;
    // Compute bounding box of projection of rear element on sampling plane
    Float rearRadius = RearElementRadius();
    Point3f finalElementTranslation =
        elementInterfaces[elementInterfaces.size() - 1].transform(Point3f(0, 0, 0));
    Point2f xy(finalElementTranslation.x, finalElementTranslation.y);
    Bounds2f projRearBounds(Point2f(-1.5f * rearRadius, -1.5f * rearRadius) + xy,
                            Point2f(1.5f * rearRadius, 1.5f * rearRadius) + xy);

    if (microlens.elementInterfaces.size() > 0) {
        return projRearBounds;
    }

    for (int i = 0; i < nSamples; ++i) {
        // Find location of sample points on $x$ segment and rear lens element
        Point3f pFilm(Lerp((i + 0.5f) / nSamples, filmX0, filmX1), 0, 0);
        Float u[2] = {RadicalInverse(0, i), RadicalInverse(1, i)};
        Point3f pRear(Lerp(u[0], projRearBounds.pMin.x, projRearBounds.pMax.x),
                      Lerp(u[1], projRearBounds.pMin.y, projRearBounds.pMax.y),
                      LensRearZ());

        // Expand pupil bounds if ray makes it through the lens system
        if (!Inside(Point2f(pRear.x, pRear.y), pupilBounds) &&
            TraceLensesFromFilm(Ray(pFilm, pRear - pFilm), elementInterfaces, nullptr))
            pupilBounds = Union(pupilBounds, Point2f(pRear.x, pRear.y));
    }

    // Return degenerate bounds if no rays made it through the lens system
    if (pupilBounds.IsDegenerate()) {
        LOG_VERBOSE("Unable to find exit pupil in x = [%f,%f] on film.", filmX0, filmX1);
        return pupilBounds;
    }

    // Expand bounds to account for sample spacing
    pupilBounds =
        Expand(pupilBounds, 2 * Length(projRearBounds.Diagonal()) / std::sqrt(nSamples));

    return pupilBounds;
}

pstd::optional<ExitPupilSample> OmniCamera::SampleExitPupil(Point2f pFilm,
                                                            Point2f uLens) const {
    // Find exit pupil bound for sample distance from film center
    Float rFilm = std::sqrt(Sqr(pFilm.x) + Sqr(pFilm.y));
    int rIndex = rFilm / (film.Diagonal() / 2) * exitPupilBounds.size();
    rIndex = std::min<int>(exitPupilBounds.size() - 1, rIndex);
    Bounds2f pupilBounds = exitPupilBounds[rIndex];
    if (pupilBounds.IsDegenerate())
        return {};

    // Generate sample point inside exit pupil bound
    Point2f pLens = pupilBounds.Lerp(uLens);
    Float pdf = 1 / pupilBounds.Area();

    // Return sample point rotated by angle of _pFilm_ with $+x$ axis
    Float sinTheta = (rFilm != 0) ? pFilm.y / rFilm : 0;
    Float cosTheta = (rFilm != 0) ? pFilm.x / rFilm : 1;
    Point3f pPupil(cosTheta * pLens.x - sinTheta * pLens.y,
                   sinTheta * pLens.x + cosTheta * pLens.y, LensRearZ());
    return ExitPupilSample{pPupil, pdf};
}

// static Vector2f mapMul(Vector2f v0, Vector2f v1) {
//     return Vector2f(v0.x*v1.x, v0.y*v1.y);
// }
// static Vector2f mapDiv(Vector2f v0, Vector2f v1) {
//     return Vector2f(v0.x/v1.x, v0.y/v1.y);
// }
// static Vector2f mapDiv(Vector2f v0, Vector2i v1) {
//     return Vector2f(v0.x / (pbrt::Float)v1.x, v0.y / (pbrt::Float)v1.y);
// }
// static Point2f mapDiv(Point2f v0, Vector2f v1) {
//     return Point2f(v0.x / v1.x, v0.y / v1.y);
// }

Point2f OmniCamera::MicrolensIndex(const Point2f p) const {
    Bounds2f extent = physicalExtent;
    Vector2f p_tmp = p - extent.pMin;
    Vector2f extent_size = extent.pMax - extent.pMin;
    Vector2f normFilm = Vector2f(p_tmp.x / (pbrt::Float)extent_size.x,
                                 p_tmp.y / (pbrt::Float)extent_size.y);
    // Vector2f normFilm = mapDiv(p - extent.pMin, extent.pMax - extent.pMin);
    Vector2i d = microlens.dimensions;
    Vector2f df = Vector2f(d.x, d.y);
    // Vector2f lensSpace = mapMul(normFilm, df);
    Vector2f lensSpace = Vector2f(normFilm.x * df.x, normFilm.y * df.y);
    return Point2f(pstd::floor(lensSpace.x), pstd::floor(lensSpace.y));
}

pstd::optional<ExitPupilSample> OmniCamera::SampleMicrolensPupil(Point2f pFilm,
                                                                 Point2f uLens) const {
    Bounds2f extent = physicalExtent;
    Point2f lensIndex = Point2f(MicrolensIndex(pFilm));
    Float R = (Float)microlens.simulationRadius;
    lensIndex -= Vector2f(R, R);
    Float diameter = (R * 2.0) + 1.0;
    Vector2i d = microlens.dimensions;
    Vector2f df = Vector2f(d.x, d.y);
    lensIndex = lensIndex + (uLens * diameter);
    Point2f sampledLensSpacePt = Point2f(lensIndex.x / df.x, lensIndex.y / df.y);
    // Point2f sampledLensSpacePt = mapDiv(lensIndex + (uLens*diameter), df);

    Float pdf = 1 / (extent.Area() * diameter * diameter /
                     (df.x * df.y));  // Number are too samll, --zhenyi
    // sample on microlens
    Point2f result2 = extent.Lerp(sampledLensSpacePt);

    return ExitPupilSample{Point3f(result2.x, result2.y, microlens.offsetFromSensor),
                           pdf};
}

// Transform OmniCamera::MicrolensElement::ComputeCameraToMicrolens() const {
//     return Scale(1, 1, -1)*Translate({ -center.x, -center.y, 0.0f });
// }

// Idx are here two integers that indicate the row and column number of the microlens
Point2f OmniCamera::MicrolensCenterFromIndex(const Point2f idx) const {
    Bounds2f extent = physicalExtent;
    Vector2f indexf(idx.x, idx.y);
    // const Vector2i& d = microlens.dimensions;
    const Vector2i &d = microlensDimensions;
    indexf = indexf + Vector2f(0.5f, 0.5f);
    // Vector2f normalizedLensCenter = mapDiv(indexf + Vector2f(0.5f, 0.5f), d); //
    // Compute proportional position
    Vector2f normalizedLensCenter = Vector2f(indexf.x / d.x, indexf.y / d.y);
    // Use this proportion to get position in film space. This is the physical position of
    // the microlens.
    Point2f lensCenter = extent.Lerp(Point2f(normalizedLensCenter));

    // If the lens index has valid bounds (0 < index < maxnumber of elements)
    // Then apply the local offset as read from the JSON file.
    if (idx.x >= 0 && idx.y >= 0 && idx.x < d.x && idx.y < d.y) {
        // This code is supposed to be implemented to anble microlens offsets
        // Zhenyi commented this out when porting

        // Original does not run onGPU lensCenter += microlens.offsets[idx.y*d.x + idx.x];
        // /
        lensCenter += microlensOffsets[idx.y * d.x + idx.x];
        // lensCenter += Vector2f(0.f, 0.f);
    }
    return lensCenter;
}

OmniCamera::MicrolensElement OmniCamera::MicrolensElementFromIndex(
    const Point2f idx) const {
    MicrolensElement element;
    element.index = idx;
    element.center = MicrolensCenterFromIndex(idx);
    const Point2f offsets[4] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    Point2f corners[4];
    for (int i = 0; i < 4; ++i) {
        corners[i] = Point2f(0, 0);
        Point2f cornerIdx = idx - Vector2i(offsets[i]);
        for (const auto &off : offsets) {
            Point2f neighborCenter = MicrolensCenterFromIndex(cornerIdx + off);

            corners[i] += neighborCenter;
        }
        corners[i] *= (Float)0.25;
        corners[i] += -element.center;  // Center the corners
    }

    // A Rectangle is used to bound the area of the microlens

    element.centeredBounds = ConvexQuadf(corners[0], corners[1], corners[2], corners[3]);
    return element;
}

OmniCamera::MicrolensElement OmniCamera::ComputeMicrolensElement(
    const Ray filmRay) const {
    Point3f pointOnMicrolens(
        filmRay(microlens.offsetFromSensor /
                filmRay.d.z));  // project ray in direction to the plane of microlens
                                // (offsetFrom Sensor)
    Point2f pointOnMicrolens2(
        pointOnMicrolens.x,
        pointOnMicrolens.y);  // Keep only (x,y) values
    return MicrolensElementFromIndex(MicrolensIndex(
        pointOnMicrolens2));  // Generate a microlensElement from this information
}

Float OmniCamera::TraceFullLensSystemFromFilm(const Ray &rIn, Ray *rOut) const {
    if (microlens.elementInterfaces.size() > 0) {
        MicrolensElement centerElement = ComputeMicrolensElement(rIn);

        Float tMin = Infinity;
        int R = microlens.simulationRadius;
        Point2f cIdx = centerElement.index;
        MicrolensElement toTrace = centerElement;
        // Check to find the first microlens we intersect with
        // Could be sped up by only checking the directions of the projection of the ray
        // R is here how far around the central micro lens we are looking for possible
        // intersections.

        for (int y = -R; y <= R; ++y) {
            for (int x = -R; x <= R; ++x) {
                const MicrolensElement el =
                    MicrolensElementFromIndex(cIdx + Vector2i(x, y));

                Transform CameraToMicrolens =
                    Scale(1, 1, -1) * Translate({-el.center.x, -el.center.y, 0.0f});

                // Original , does not work on gpu float newT = TToBackLens(rIn,
                // microlens.ElementInterfaces, CameraToMicrolens, el.centeredBounds);
                float newT = TToBackLens(rIn, microlensElementInterfaces,
                                         CameraToMicrolens, el.centeredBounds);

                if (newT < tMin) {
                    tMin = newT;
                    toTrace = el;
                }
            }
        }

        // After choosing which microlens to trace from, trace it. This produce the ray
        // after microlens 'rAfterMicroLens'. This Ray is then traced through the main
        // camera lens.
        if (tMin < Infinity) {
            // 1. Trace ray through chosen microlens
            Ray rAfterMicrolens;
            if (rOut)
                rAfterMicrolens = *rOut;
            Transform CameraToMicrolensToTrace =
                Scale(1, 1, -1) * Translate({-toTrace.center.x, -toTrace.center.y, 0.0f});

            if (!TraceLensesFromFilm(rIn, microlensElementInterfaces, &rAfterMicrolens,
                                     CameraToMicrolensToTrace, toTrace.centeredBounds)) {
                return false;
            }

            // 2. Trace the ray after microlens through the main lens;
            Float result = TraceLensesFromFilm(rAfterMicrolens, elementInterfaces, rOut);

            /*
            static int paths = 0;
            if (result && cIdx != toTrace.index) {
                ++paths;
                if (paths % 100 == 0) {
                    printf("Contributing non-center microlens path count: %d\n", paths);
                }
            }*/
            return result;
        } else {
            return false;
        }
    } else {
        // If there is no microlens, just trace through the system
        return TraceLensesFromFilm(rIn, elementInterfaces, rOut);
    }
}

pstd::optional<CameraRay> OmniCamera::GenerateRay(CameraSample sample,
                                                  SampledWavelengths &lambda) const {
    // Find point on film, _pFilm_, corresponding to _sample.pFilm_
    Point2f s(sample.pFilm.x / film.FullResolution().x,
              sample.pFilm.y / film.FullResolution().y);
    Point2f pFilm2 = physicalExtent.Lerp(s);
    Point3f pFilm(-pFilm2.x, pFilm2.y, 0);

    // Trace ray from _pFilm_ through lens system
    pstd::optional<ExitPupilSample> eps;
    if (microlens.elementInterfaces.size() > 0) {
        eps = SampleMicrolensPupil(Point2f(pFilm.x, pFilm.y), sample.pLens);
    } else {
        eps = SampleExitPupil(Point2f(pFilm.x, pFilm.y), sample.pLens);
    }
    if (!eps)
        return {};

    Ray rFilm(pFilm, eps->pPupil - pFilm);

    Ray ray;

    Float weight = TraceFullLensSystemFromFilm(rFilm, &ray);

    if (weight == 0)
        return {};

    // Finish initialization of _OmniCamera_ ray
    ray.time = SampleTime(sample.time);
    ray.medium = medium;
    ray = RenderFromCamera(ray);
    ray.d = Normalize(ray.d);

    // Compute weighting for _OmniCamera_ ray
    Float cosTheta = Normalize(rFilm.d).z;
    if (microlens.elementInterfaces.size() > 0) {
        Float simDiameter = (microlens.simulationRadius * 2.0 + 1.0);
        weight *=
            Pow<4>(cosTheta) * (simDiameter * simDiameter);  // tmp use simple weighting
        // weight *= Pow<4>(cosTheta) / (eps->pdf * Sqr(LensRearZ()));
    } else {
        weight *= Pow<4>(cosTheta) / (eps->pdf * Sqr(LensRearZ()));
    }
    return CameraRay{ray, SampledSpectrum(weight)};
}

// STAT_PERCENT("Camera/Rays vignetted by lens system", vignettedRays, totalRays);

std::string OmniCamera::LensElementInterface::ToString() const {
    return StringPrintf("[ LensElementInterface curvatureRadius: %f thickness: %f "
                        "eta: %f apertureRadius: %f ]",
                        curvatureRadius.x, thickness, eta, apertureRadius.x);
}

Float OmniCamera::TraceLensesFromScene(const Ray &rCamera, Ray *rOut) const {
    Float elementZ = -LensFrontZ();
    // Transform _rCamera_ from camera to lens system space
    const Transform LensFromCamera = Scale(1, 1, -1);
    Ray rLens = LensFromCamera(rCamera);
    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        bool isStop = (element.curvatureRadius.x == 0);
        // bool isStop = (element.curvatureRadius.x == 0) &&
        // (element.asphericCoefficients.size() == 0); Omni method
        IntersectResult result = TraceElement(element, rLens, elementZ, t, n, isStop);
        if (result != HIT)
            return 0;

        /* PBRT method
        if (isStop) {
            t = (elementZ - rLens.o.z) / rLens.d.z;
            if (t < 0)
                return 0;
        } else {
            Float radius = element.curvatureRadius.x;
            Float zCenter = elementZ + element.curvatureRadius.x;
            if (!IntersectSphericalElement(radius, zCenter, rLens, &t, &n))
                return 0;
        }
        */

        // Test intersection point against element aperture
        // Don't worry about the aperture image here.
        Point3f pHit = rLens(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        if (r2 > element.apertureRadius.x * element.apertureRadius.x)
            return 0;
        rLens.o = pHit;

        // Update ray path for from-scene element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = (i == 0 || elementInterfaces[i - 1].eta == 0)
                              ? 1
                              : elementInterfaces[i - 1].eta;
            Float eta_t = (elementInterfaces[i].eta != 0) ? elementInterfaces[i].eta : 1;
            if (!Refract(Normalize(-rLens.d), n, eta_t / eta_i, nullptr, &wt))
                return 0;
            rLens.d = wt;
        }
        elementZ += element.thickness;
    }
    // Transform _rLens_ from lens system space back to camera space
    if (rOut)
        *rOut = Ray(Point3f(rLens.o.x, rLens.o.y, -rLens.o.z),
                    Vector3f(rLens.d.x, rLens.d.y, -rLens.d.z), rLens.time);
    return 1;
}

void OmniCamera::DrawLensSystem() const {
    Float sumz = -LensFrontZ();
    Float z = sumz;

    auto toZXCoord = [](const Point3f &p, const Transform &xForm) {
        const Point3f p3 = xForm(p);
        return Point2f(p3.z, p3.x);
    };

    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        const Transform &xForm = element.transform;
        if (element.asphericCoefficients.size() == 0) {
            Float r = element.curvatureRadius.x;
            if (r == 0) {
                Point2f startU =
                    toZXCoord(Point3f(element.apertureRadius.x, 0, z), xForm);
                Point2f startL =
                    toZXCoord(Point3f(-element.apertureRadius.x, 0, z), xForm);
                Point2f endU =
                    toZXCoord(Point3f(2 * element.apertureRadius.x, 0, z), xForm);
                Point2f endL =
                    toZXCoord(Point3f(-2 * element.apertureRadius.x, 0, z), xForm);
                // stop
                printf("{Thick, Line[{{%f, %f}, {%f, %f}}], ", startU.x, startU.y, endU.x,
                       endU.y);
                printf("Line[{{%f, %f}, {%f, %f}}]}, ", startL.x, startL.y, endL.x,
                       endL.y);
            } else {
                // TODO compute proper tilt
                Point2f C = toZXCoord(Point3f(0, 0, z + r), xForm);
                Float theta = std::abs(std::asin(element.apertureRadius.x / r));
                if (r > 0) {
                    // convex as seen from front of lens
                    Float t0 = Pi - theta;
                    Float t1 = Pi + theta;
                    printf("Circle[{%f, %f}, %f, {%f, %f}], ", C.x, C.y, r, t0, t1);
                } else {
                    // concave as seen from front of lens
                    Float t0 = -theta;
                    Float t1 = theta;
                    printf("Circle[{%f, %f}, %f, {%f, %f}], ", C.x, C.y, -r, t0, t1);
                }
                if (element.eta != 0 && element.eta != 1) {
                    // TODO: re-enable
                    /*
                    // connect top/bottom to next element
                    CHECK_LT(i + 1, elementInterfaces.size());
                    Float nextApertureRadius =
                        elementInterfaces[i + 1].apertureRadius.x;
                    Float h = std::max(element.apertureRadius.x, nextApertureRadius);
                    Float hlow =
                        std::min(element.apertureRadius.x, nextApertureRadius);

                    Float zp0, zp1;
                    if (r > 0) {
                        zp0 = z + element.curvatureRadius.x -
                              element.apertureRadius.x / std::tan(theta);
                    } else {
                        zp0 = z + element.curvatureRadius.x +
                              element.apertureRadius.x / std::tan(theta);
                    }

                    Float nextCurvatureRadius =
                        elementInterfaces[i + 1].curvatureRadius.x;
                    Float nextTheta = std::abs(
                        std::asin(nextApertureRadius / nextCurvatureRadius));
                    if (nextCurvatureRadius > 0) {
                        zp1 = z + element.thickness + nextCurvatureRadius -
                              nextApertureRadius / std::tan(nextTheta);
                    } else {
                        zp1 = z + element.thickness + nextCurvatureRadius +
                              nextApertureRadius / std::tan(nextTheta);
                    }

                    // Connect tops
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, h, zp1, h);
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, -h, zp1, -h);

                    // vertical lines when needed to close up the element profile
                    if (element.apertureRadius.x < nextApertureRadius) {
                        printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, h, zp0, hlow);
                        printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, -h, zp0, -hlow);
                    } else if (element.apertureRadius.x > nextApertureRadius) {
                        printf("Line[{{%f, %f}, {%f, %f}}], ", zp1, h, zp1, hlow);
                        printf("Line[{{%f, %f}, {%f, %f}}], ", zp1, -h, zp1, -hlow);
                    }
                    */
                }
            }
        } else {
            Float yStep = element.apertureRadius.x / 16;
            Float rInit = element.apertureRadius.x;
            Float lensZInit = z + computeZOfLensElement(std::abs(rInit), element);
            Point2f lastPoint = toZXCoord(Point3f(rInit, 0, lensZInit), xForm);
            for (Float r = element.apertureRadius.x; r >= -element.apertureRadius.x;
                 r -= yStep) {
                Float lensZ = z + computeZOfLensElement(std::abs(r), element);
                Point2f thisPoint = toZXCoord(Point3f(r, 0, lensZ), xForm);
                printf("Line[{{%f, %f}, {%f, %f}}], ", lastPoint.x, lastPoint.y,
                       thisPoint.x, thisPoint.y);
                lastPoint = thisPoint;
            }
        }
        z += element.thickness;
    }

    float filmRadius = film.Diagonal() / (2.0f * std::sqrt(2.0f));
    printf("Line[{{0, %f}, {0, %f}}], ", -filmRadius, filmRadius);
    // optical axis
    printf("Line[{{0, 0}, {%f, 0}}] ", 1.2f * sumz);
}

void OmniCamera::DrawRayPathFromFilm(const Ray &r, bool arrow,
                                     bool toOpticalIntercept) const {
    Float elementZ = 0;
    // Transform _ray_ from camera to lens system space
    static const Transform LensFromCamera = Scale(1, 1, -1);
    Ray ray = LensFromCamera(r);
    printf("{ ");
    if (TraceLensesFromFilm(r, elementInterfaces, nullptr) == 0) {
        printf("Dashed, RGBColor[.8, .5, .5]");
    } else
        printf("RGBColor[.5, .5, .8]");

    for (int i = elementInterfaces.size() - 1; i >= 0; --i) {
        const LensElementInterface &element = elementInterfaces[i];
        elementZ -= element.thickness;
        bool isStop = (element.curvatureRadius.x == 0);
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        if (isStop)
            t = -(ray.o.z - elementZ) / ray.d.z;
        else {
            Float radius = element.curvatureRadius.x;
            Float zCenter = elementZ + element.curvatureRadius.x;
            if (!IntersectSphericalElement(radius, zCenter, ray, &t, &n))
                goto done;
        }
        CHECK_GE(t, 0);

        printf(", Line[{{%f, %f}, {%f, %f}}]", ray.o.z, ray.o.x, ray(t).z, ray(t).x);

        // Test intersection point against element aperture
        Point3f pHit = ray(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        Float apertureRadius2 = element.apertureRadius.x * element.apertureRadius.x;
        if (r2 > apertureRadius2)
            goto done;
        ray.o = pHit;

        // Update ray path for element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = element.eta;
            Float eta_t = (i > 0 && elementInterfaces[i - 1].eta != 0)
                              ? elementInterfaces[i - 1].eta
                              : 1;
            if (!Refract(Normalize(-ray.d), n, eta_t / eta_i, nullptr, &wt))
                goto done;
            ray.d = wt;
        }
    }

    ray.d = Normalize(ray.d);
    {
        Float ta = std::abs(elementZ / 4);
        if (toOpticalIntercept) {
            ta = -ray.o.x / ray.d.x;
            printf(", Point[{%f, %f}]", ray(ta).z, ray(ta).x);
        }
        printf(", %s[{{%f, %f}, {%f, %f}}]", arrow ? "Arrow" : "Line", ray.o.z, ray.o.x,
               ray(ta).z, ray(ta).x);

        // overdraw the optical axis if needed...
        if (toOpticalIntercept)
            printf(", Line[{{%f, 0}, {%f, 0}}]", ray.o.z, ray(ta).z * 1.05f);
    }

done:
    printf("}");
}

void OmniCamera::DrawRayPathFromScene(const Ray &r, bool arrow,
                                      bool toOpticalIntercept) const {
    Float elementZ = LensFrontZ() * -1;

    // Transform _ray_ from camera to lens system space
    static const Transform LensFromCamera = Scale(1, 1, -1);
    Ray ray = LensFromCamera(r);
    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        bool isStop = (element.curvatureRadius.x == 0);
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        if (isStop)
            t = -(ray.o.z - elementZ) / ray.d.z;
        else {
            Float radius = element.curvatureRadius.x;
            Float zCenter = elementZ + element.curvatureRadius.x;
            if (!IntersectSphericalElement(radius, zCenter, ray, &t, &n))
                return;
        }
        CHECK_GE(t, 0.f);

        printf("Line[{{%f, %f}, {%f, %f}}],", ray.o.z, ray.o.x, ray(t).z, ray(t).x);

        // Test intersection point against element aperture
        Point3f pHit = ray(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        Float apertureRadius2 = element.apertureRadius.x * element.apertureRadius.x;
        if (r2 > apertureRadius2)
            return;
        ray.o = pHit;

        // Update ray path for from-scene element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = (i == 0 || elementInterfaces[i - 1].eta == 0.f)
                              ? 1.f
                              : elementInterfaces[i - 1].eta;
            Float eta_t =
                (elementInterfaces[i].eta != 0.f) ? elementInterfaces[i].eta : 1.f;
            if (!Refract(Normalize(-ray.d), n, eta_t / eta_i, nullptr, &wt))
                return;
            ray.d = wt;
        }
        elementZ += element.thickness;
    }

    // go to the film plane by default
    {
        Float ta = -ray.o.z / ray.d.z;
        if (toOpticalIntercept) {
            ta = -ray.o.x / ray.d.x;
            printf("Point[{%f, %f}], ", ray(ta).z, ray(ta).x);
        }
        printf("%s[{{%f, %f}, {%f, %f}}]", arrow ? "Arrow" : "Line", ray.o.z, ray.o.x,
               ray(ta).z, ray(ta).x);
    }
}

void OmniCamera::RenderExitPupil(Float sx, Float sy, const char *filename) const {
    Point3f pFilm(sx, sy, 0);

    const int nSamples = 2048;
    Image image(PixelFormat::Float, {nSamples, nSamples}, {"Y"});

    for (int y = 0; y < nSamples; ++y) {
        Float fy = (Float)y / (Float)(nSamples - 1);
        Float ly = Lerp(fy, -RearElementRadius(), RearElementRadius());
        for (int x = 0; x < nSamples; ++x) {
            Float fx = (Float)x / (Float)(nSamples - 1);
            Float lx = Lerp(fx, -RearElementRadius(), RearElementRadius());

            Point3f pRear(lx, ly, LensRearZ());

            if (lx * lx + ly * ly > RearElementRadius() * RearElementRadius())
                image.SetChannel({x, y}, 0, 1.);
            else if (TraceLensesFromFilm(Ray(pFilm, pRear - pFilm), elementInterfaces,
                                         nullptr))
                image.SetChannel({x, y}, 0, 0.5);
            else
                image.SetChannel({x, y}, 0, 0.);
        }
    }

    image.Write(filename);
}

void OmniCamera::TestExitPupilBounds() const {
    Float filmDiagonal = film.Diagonal();

    static RNG rng;

    Float u = rng.Uniform<Float>();
    Point3f pFilm(u * filmDiagonal / 2, 0, 0);

    Float r = pFilm.x / (filmDiagonal / 2);
    int pupilIndex = std::min<int>(exitPupilBounds.size() - 1,
                                   pstd::floor(r * (exitPupilBounds.size() - 1)));
    Bounds2f pupilBounds = exitPupilBounds[pupilIndex];
    if (pupilIndex + 1 < (int)exitPupilBounds.size())
        pupilBounds = Union(pupilBounds, exitPupilBounds[pupilIndex + 1]);

    // Now, randomly pick points on the aperture and see if any are outside
    // of pupil bounds...
    for (int i = 0; i < 1000; ++i) {
        Point2f u2{rng.Uniform<Float>(), rng.Uniform<Float>()};
        Point2f pd = SampleUniformDiskConcentric(u2);
        pd *= RearElementRadius();

        Ray testRay(pFilm, Point3f(pd.x, pd.y, 0.f) - pFilm);
        Ray testOut;
        if (!TraceLensesFromFilm(testRay, elementInterfaces, &testOut))
            continue;

        if (!Inside(pd, pupilBounds)) {
            fprintf(stderr,
                    "Aha! (%f,%f) went through, but outside bounds (%f,%f) - "
                    "(%f,%f)\n",
                    pd.x, pd.y, pupilBounds.pMin[0], pupilBounds.pMin[1],
                    pupilBounds.pMax[0], pupilBounds.pMax[1]);
            RenderExitPupil(
                (Float)pupilIndex / exitPupilBounds.size() * filmDiagonal / 2.f, 0.f,
                "low.exr");
            RenderExitPupil(
                (Float)(pupilIndex + 1) / exitPupilBounds.size() * filmDiagonal / 2.f,
                0.f, "high.exr");
            RenderExitPupil(pFilm.x, 0.f, "mid.exr");
            exit(0);
        }
    }
    fprintf(stderr, ".");
}

std::string OmniCamera::ToString() const {
    return StringPrintf("[ OmniCamera %s elementInterfaces: %s exitPupilBounds: %s ]",
                        CameraBase::ToString(), elementInterfaces, exitPupilBounds);
}

OmniCamera *OmniCamera::Create(const ParameterDictionary &parameters,
                               const CameraTransform &cameraTransform, Film film,
                               Medium medium, const FileLoc *loc, Allocator alloc) {
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    // Omni camera-specific parameters
    std::string lensFile = ResolveFilename(parameters.GetOneString("lensfile", ""));
    Float apertureDiameter = parameters.GetOneFloat("aperturediameter", 1.0);
    Float focusDistance = parameters.GetOneFloat("focusdistance", 10.0);
    // Microlens parameters
    Float microlensSensorOffset = parameters.GetOneFloat("microlenssensoroffset", 0.001);
    int microlensSimulationRadius = parameters.GetOneInt("microlenssimulationradius", 0);
    // Hard set the film distance
    Float filmDistance = parameters.GetOneFloat("filmdistance", 0);
    // Chromatic aberration flag
    bool caFlag = parameters.GetOneBool("chromaticAberrationEnabled", false);
    // Diffraction limit calculation flag
    bool diffractionEnabled = parameters.GetOneBool("diffractionEnabled", false);

    if (microlensSimulationRadius != 0) {
        Warning("We only currently support simulating one microlens per pixel, switching "
                "microlensSimulationRadius to 0");
    }

    if (lensFile.empty()) {
        Error(loc, "No lens description file supplied!");
        return nullptr;
    }
    // Load element data from lens description file: main lens
    pstd::vector<OmniCamera::LensElementInterface> lensInterfaceData;

    // Load element data from lens description file: microlens
    pstd::vector<OmniCamera::LensElementInterface> microlensData;

    pstd::vector<Vector2f> microlensOffsets;

    Vector2i microlensDims;

    auto endsWith = [](const std::string &str, const std::string &suffix) {
        return str.size() >= suffix.size() &&
               0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
    };

    if (!endsWith(lensFile, ".json")) {
        Error("Invalid format for lens specification file \"%s\".", lensFile.c_str());
        return nullptr;
    }
    // read lens json file
    std::ifstream i(lensFile);
    json j;
    if (i && (i >> j)) {
        // assert(j.is_object())
        // j["name"]
        // j["description"]
        // if (j["name"].is_string()) {
        //     j["name"].get<std::string>().c_str();
        // }
        // if (j["description"].is_string()) {
        //     j["description"].get<std::string>().c_str();
        // }
        auto jsurfaces = j["surfaces"];

        auto toVec2 = [](json val) {
            if (val.is_number()) {
                return Vector2f{(Float)val, (Float)val};
            } else if (val.is_array() && val.size() == 2) {
                return Vector2f{val[0], val[1]};
            }
            return Vector2f();  // Default value
        };
        auto toVec2i = [](json val) {
            if (val.is_number()) {
                return Vector2i{(int)val, (int)val};
            } else if (val.is_array() && val.size() == 2) {
                return Vector2i{val[0], val[1]};
            }
            return Vector2i();  // Default value
        };

        auto toTransform = [lensFile](json t) {
            // Stored in columns in json, but pbrt stores matrices
            // in row-major order.
            if (t.is_null()) {  // Perfectly fine to have no transform
                return Transform();
            }
            if (!(t.size() == 4 && t[0].size() == 3 && t[1].size() == 3 &&
                  t[2].size() == 3 && t[3].size() == 3)) {
                Error("Invalid transform in lens specification file \"%s\", must be an "
                      "array of 4 arrays of 3 floats each (column-major transform).",
                      lensFile.c_str());
            }
            Float m[4][4];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    m[r][c] = (Float)t[c][r];
                }
                // Translation specified in mm, needs to be converted to m
                m[r][3] = (Float)t[3][r] * (Float).001;
            }
            m[3][0] = m[3][1] = m[3][2] = (Float)0.0;
            m[3][3] = (Float)1.0;
            return Transform(m);
        };
        auto toIORSpectrum = [lensFile](json jiors) {
            // Stored in columns in json, but pbrt stores matrices
            // in row-major order.
            if (jiors.is_number()) {  // Perfectly fine to have no transform
                SampledSpectrum iors((Float)jiors);
                // return (Float)jiors;
                return iors;
            }
            if (!(jiors.is_array()) || (jiors.size() != 2) || (!jiors[0].is_array()) ||
                (!jiors[1].is_array()) || (jiors[0].size() != jiors[1].size()) ||
                (!(jiors[0][0].is_number())) || (!(jiors[1][0].is_number()))) {
                Error("Invalid ior in lens specification file \"%s\","
                      " must be either a single float, or a pair of parallel arrays."
                      " The first array should be a list of wavelengths (in nm),"
                      " The second array should be the ior at those wavelengths.",
                      lensFile.c_str());
            }
            size_t numSamples = jiors[0].size();
            std::vector<Float> wavelengths(numSamples);
            std::vector<Float> iors(numSamples);
            for (int i = 0; i < numSamples; ++i) {
                wavelengths[i] = (Float)jiors[0][i];
                iors[i] = (Float)jiors[1][i];
            }
            // Resample spectrum
            SampledWavelengths lambda = SampledWavelengths::SampleUniform(0);
            PiecewiseLinearSpectrum s_tmp(pstd::MakeSpan(wavelengths),
                                          pstd::MakeSpan(iors));
            SampledSpectrum s;
            for (int i = 0; i < NSpectrumSamples; ++i) {
                s[i] = s_tmp(lambda[i]);
            }
            /* TG: This original piece of code checked whether the spectrum is constant
            when given, because it was not yet supported for (int i = 0; i < numSamples-1;
            ++i) { if (!(s[i] == s[i + 1])) { Error("Invalid ior in lens specification
            file \"%s\"," " spectrum must be constant (wavelength-varying ior NYI)",
                        lensFile.c_str());
                }
            }
            */
            return s;
        };

        auto toAsphericCoefficients = [lensFile](json jaspherics) {
            std::vector<Float> coefficients;
            if (jaspherics.is_null()) {  // Perfectly fine to have no aspherics
                return coefficients;
            }

            if (!(jaspherics.is_array())) {
                Error("Invalid aspheric coefficients in lens specification file \"%s\","
                      " must be an array of floats",
                      lensFile.c_str());
            }
            for (int i = 0; i < jaspherics.size(); ++i) {
                json coeff = jaspherics[i];
                if (!coeff.is_number()) {
                    Error(
                        "Invalid aspheric coefficients in lens specification file \"%s\","
                        " must be an array of floats",
                        lensFile.c_str());
                }
                coefficients.push_back((Float)coeff);
            }

            return coefficients;
        };

        auto toLensElementInterface = [toVec2, toTransform, toIORSpectrum,
                                       apertureDiameter,
                                       toAsphericCoefficients](json surf) {
            OmniCamera::LensElementInterface result;
            // Convert mm to m
            result.apertureRadius = toVec2(surf["semi_aperture"]) * (Float).001;
            result.conicConstant = toVec2(surf["conic_constant"]) * (Float).001;
            result.curvatureRadius = toVec2(surf["radius"]) * (Float).001;
            result.etaspectral = toIORSpectrum(surf["ior"]);
            result.eta =
                result
                    .etaspectral[0];  // for backwards compatibility during refactor (TG)

            result.thickness = Float(surf["thickness"]) * (Float).001;
            result.transform = toTransform(surf["transform"]);

            if (result.curvatureRadius.x == 0.0f) {
                Float apertureRadius = apertureDiameter * (Float).001 / Float(2.);
                if (apertureRadius > result.apertureRadius.x) {
                    Warning("Specified aperture radius %f is greater than maximum "
                            "possible %f.  Clamping it.",
                            apertureRadius, result.apertureRadius.x);
                } else {
                    result.apertureRadius.x = apertureRadius;
                    result.apertureRadius.y = apertureRadius;
                }
            }
            result.asphericCoefficients =
                toAsphericCoefficients(surf["aspheric_coefficients"]);
            return result;
        };

        if (jsurfaces.is_array() && jsurfaces.size() > 0) {
            for (auto jsurf : jsurfaces) {
                lensInterfaceData.push_back(toLensElementInterface(jsurf));
            }
        } else {
            Error("Error, lens specification file without a valid surface array \"%s\".",
                  lensFile.c_str());
            return nullptr;
        }
        auto microlens = j["microlens"];
        if (!microlens.is_null()) {
            microlensDims = toVec2i(microlens["dimensions"]);

            if (microlensDims.x <= 0 || microlensDims.y <= 0) {
                Error(
                    "Error, microlens specification without valid dimensions in \"%s\".",
                    lensFile.c_str());
                return nullptr;
            }

            auto mljOffsets = microlens["offsets"];
            if (mljOffsets.is_array() && mljOffsets.size() > 0) {
                if (mljOffsets.size() != microlensDims.x * microlensDims.y) {
                    Error("Error, microlens dimensions (%d x %d ) mismatch number of "
                          "offsets (%d) in \"%s\".",
                          microlensDims.x, microlensDims.y, (int)mljOffsets.size(),
                          lensFile.c_str());
                    return nullptr;
                }
                for (auto offset : mljOffsets) {
                    microlensOffsets.push_back(toVec2(offset));
                    // microlensOffsets.push_back(Float(0.f));
                }
            }

            auto mljSurfaces = microlens["surfaces"];

            if (mljSurfaces.is_array() && mljSurfaces.size() > 0) {
                for (auto jsurf : mljSurfaces) {
                    microlensData.push_back(toLensElementInterface(jsurf));
                }
            } else {
                Error("Error, microlens specification without a valid surface array in "
                      "\"%s\".",
                      lensFile.c_str());
                return nullptr;
            }
        }
    } else {
        Error("Error reading lens specification file \"%s\".", lensFile.c_str());
        return nullptr;
    }

    // aperture image
    int builtinRes = 256;
    auto rasterize = [&](pstd::span<const Point2f> vert) {
        Image image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"}, nullptr, alloc);

        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                Point2f p(-1 + 2 * (x + 0.5f) / image.Resolution().x,
                          -1 + 2 * (y + 0.5f) / image.Resolution().y);
                int windingNumber = 0;
                // Test against edges
                for (int i = 0; i < vert.size(); ++i) {
                    int i1 = (i + 1) % vert.size();
                    Float e = (p[0] - vert[i][0]) * (vert[i1][1] - vert[i][1]) -
                              (p[1] - vert[i][1]) * (vert[i1][0] - vert[i][0]);
                    if (vert[i].y <= p.y) {
                        if (vert[i1].y > p.y && e > 0)
                            ++windingNumber;
                    } else if (vert[i1].y <= p.y && e < 0)
                        --windingNumber;
                }

                image.SetChannel({x, y}, 0, windingNumber == 0 ? 0.f : 1.f);
            }

        return image;
    };

    std::string apertureName = ResolveFilename(parameters.GetOneString("aperture", ""));
    Image apertureImage(alloc);
    if (!apertureName.empty()) {
        // built-in diaphragm shapes
        if (apertureName == "gaussian") {
            apertureImage = Image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"},
                                  nullptr, alloc);
            for (int y = 0; y < apertureImage.Resolution().y; ++y)
                for (int x = 0; x < apertureImage.Resolution().x; ++x) {
                    Point2f uv(-1 + 2 * (x + 0.5f) / apertureImage.Resolution().x,
                               -1 + 2 * (y + 0.5f) / apertureImage.Resolution().y);
                    Float r2 = Sqr(uv.x) + Sqr(uv.y);
                    Float sigma2 = 1;
                    Float v = std::max<Float>(
                        0, std::exp(-r2 / sigma2) - std::exp(-1 / sigma2));
                    apertureImage.SetChannel({x, y}, 0, v);
                }
        } else if (apertureName == "square") {
            apertureImage = Image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"},
                                  nullptr, alloc);
            for (int y = 0; y < apertureImage.Resolution().y; ++y)
                for (int x = 0; x < apertureImage.Resolution().x; ++x)
                    apertureImage.SetChannel({x, y}, 0, 1.f);
        } else if (apertureName == "pentagon") {
            // https://mathworld.wolfram.com/RegularPentagon.html
            Float c1 = (std::sqrt(5.f) - 1) / 4;
            Float c2 = (std::sqrt(5.f) + 1) / 4;
            Float s1 = std::sqrt(10.f + 2.f * std::sqrt(5.f)) / 4;
            Float s2 = std::sqrt(10.f - 2.f * std::sqrt(5.f)) / 4;
            // Vertices in CW order.
            Point2f vert[5] = {Point2f(0, 1), {s1, c1}, {s2, -c2}, {-s2, -c2}, {-s1, c1}};
            // Scale down slightly
            for (int i = 0; i < 5; ++i)
                vert[i] *= .8f;
            apertureImage = rasterize(vert);
        } else if (apertureName == "star") {
            // 5-sided. Vertices are two pentagons--inner and outer radius
            pstd::array<Point2f, 10> vert;
            for (int i = 0; i < 10; ++i) {
                // inner radius: https://math.stackexchange.com/a/2136996
                Float r =
                    (i & 1) ? 1.f : (std::cos(Radians(72.f)) / std::cos(Radians(36.f)));
                vert[i] = Point2f(r * std::cos(Pi * i / 5.f), r * std::sin(Pi * i / 5.f));
            }
            std::reverse(vert.begin(), vert.end());
            apertureImage = rasterize(vert);
        } else {
            ImageAndMetadata im = Image::Read(apertureName, alloc);
            apertureImage = std::move(im.image);
            if (apertureImage.NChannels() > 1) {
                ImageChannelDesc rgbDesc = apertureImage.GetChannelDesc({"R", "G", "B"});
                if (!rgbDesc)
                    ErrorExit("%s: didn't find R, G, B channels to average for "
                              "aperture image.",
                              apertureName);

                Image mono(PixelFormat::Float, apertureImage.Resolution(), {"Y"}, nullptr,
                           alloc);
                for (int y = 0; y < mono.Resolution().y; ++y)
                    for (int x = 0; x < mono.Resolution().x; ++x) {
                        Float avg = apertureImage.GetChannels({x, y}, rgbDesc).Average();
                        mono.SetChannel({x, y}, 0, avg);
                    }

                apertureImage = std::move(mono);
            }
        }

        if (apertureImage) {
            apertureImage.FlipY();

            // Normalize it so that brightness matches a circular aperture
            Float sum = 0;
            for (int y = 0; y < apertureImage.Resolution().y; ++y)
                for (int x = 0; x < apertureImage.Resolution().x; ++x)
                    sum += apertureImage.GetChannel({x, y}, 0);
            Float avg =
                sum / (apertureImage.Resolution().x * apertureImage.Resolution().y);

            Float scale = (Pi / 4) / avg;
            for (int y = 0; y < apertureImage.Resolution().y; ++y)
                for (int x = 0; x < apertureImage.Resolution().x; ++x)
                    apertureImage.SetChannel({x, y}, 0,
                                             apertureImage.GetChannel({x, y}, 0) * scale);
        }
    }

    return alloc.new_object<OmniCamera>(
        cameraBaseParameters, lensInterfaceData, focusDistance, filmDistance, caFlag,
        diffractionEnabled, microlensData, microlensDims, microlensOffsets,
        microlensSensorOffset, microlensSimulationRadius, apertureDiameter,
        std::move(apertureImage), alloc);
}


// RTFCamera Method Definitions
RTFCamera::RTFCamera(CameraBaseParameters baseParameters, 
                    std::string bbmode,
                    Float filmDistance, bool caFlag, Float apertureDiameter,
                    Float planeOffsetInput, Float planeOffsetOutput, Float lensThickness,
                    pstd::vector<pstd::vector< RTFCamera::LensPolynomialTerm>> polynomialMaps,
                   pstd::vector<std::shared_ptr<PassNoPass>> passNoPassPerWavelength,
                    pstd::vector<Float> polyWavelengths_nm,
                    Allocator alloc)
    : CameraBase(baseParameters),
      exitPupilBounds(alloc),
      caFlag(caFlag), 
      filmDistance(filmDistance), 
      planeOffsetInput(planeOffsetInput),planeOffsetOutput(planeOffsetOutput),
      lensThickness(lensThickness),
      polynomialMaps(polynomialMaps),
      bbmode(bbmode),
      passNoPassPerWavelength(passNoPassPerWavelength),
      polyWavelengths_nm(polyWavelengths_nm) {
    
        // Compute pupil bounding box to improve performance buy having a higher chance of finding a passing ray
        // Motivation: PBRT randomly chooses samples on the film (film pixel samples). It is then the job of the camera
        // to generate a ray into the scene. In principle we only need to sample rays that are passed by the pass no pass function
        // However, these passnopass functions are encoded for off-axis positions ON THE INPUTPLANE of the RTF.
        // PBRT on the other chooses positions ON THE FILM. Therefore we have a deadlock because one cannot know the position on hte
        // input plane before knowing what ray to shoot, and to know what ray to shoot, we need to know the off-axis position.
        // The procedure below works around this by remapping the passnopassfunctions for off-axis positions on the FILM.
        // This is done by calculating a bounding box for 64 offaxis position on the film. 
        // The function SampleExitPupil then, during tracing, uses these bounding boxes to generate a ray which is then intersected with the actual inputplane

    //std::cout << "Compute exit pupil bounds" << "\n";
    // Compute film's physical extent
    Float diagonal = film.Diagonal();
    Float aspect = (Float)film.FullResolution().y / (Float)film.FullResolution().x;
    Float x = std::sqrt(Sqr(diagonal) / (1 + Sqr(aspect)));
    Float y = aspect * x;
    physicalExtent = Bounds2f(Point2f(-x / 2, -y / 2), Point2f(x / 2, y / 2));

    int nSamples = 64; 
    exitPupilBoundsRTF.resize(nSamples);
    
    ParallelFor(0, nSamples, [&](int64_t i) {
        Float r0 = (Float)i / nSamples * diagonal / 2;
        Float r1 = (Float)(i + 1) / nSamples * diagonal / 2;
        exitPupilBoundsRTF[i] = BoundExitPupilRTF(r0, r1);

    });
        // Compute minimum differentials for _RealisticCamera_
    FindMinimumDifferentials(this); // Thomas: Why was this added
}

// This returns a vector with two elements [radius, degree]
inline Vector2f RTFCamera::Pos2RadiusRotation(const Point3f pos) const{
    // Convert position to radius
    Float radius = std::sqrt(pos.x * pos.x + pos.y * pos.y);
    Float deg = std::atan2(pos.y,pos.x);

    Vector2f res = Vector2f(radius, Degrees(deg));
    
    return res;
}

// This is much faster for small exponents than using std::pow
// See https://baptiste-wicht.com/posts/2017/09/cpp11-performance-tip-when-to-use-std-pow.html
inline Float powerLoop(Float number,Float power){
    Float temp=1;
    for(int i = 0; i <power;i++){
        temp = temp*number;
    }
    return temp;
}

// Take (rho,dx,du) input and apply apply polynomial (single LensPolynomialTerm)
Float RTFCamera::PolynomialCal(Float rho, Float dx, Float dy, LensPolynomialTerm &polyTerm) const{
    Float res= 0;
    for (int i = 0; i < polyTerm.termr.size(); i++) {
       res += (powerLoop(rho * 1000, polyTerm.termr[i]) * powerLoop(dx, polyTerm.termu[i]) * powerLoop(dy, polyTerm.termv[i])) * polyTerm.coeff[i]; //Faster
    }

    return res;
}


Ray RTFCamera::ApplyPolynomial(Float rho, Vector3f dir, pstd::vector< RTFCamera::LensPolynomialTerm>  &polynomialMap) const{

    
    // Evaluate polynomials  
    /// Position on output plane
    Float x = PolynomialCal(rho,dir.x,dir.y,polynomialMap[0]) * 0.001f; //mm to meter
    //std::cout << "Applypoly X:" << x << "\n";
    Float y = PolynomialCal(rho,dir.x,dir.y,polynomialMap[1]) * 0.001f; // mm to meter
    Float z = PolynomialCal(rho,dir.x,dir.y,polynomialMap[2]) * 0.001f; // mm to meter  Z measured from rear vertex of reverse lens (object side)

    /// Direction vector first two components
    /// The RTF generates only two components o the output direction vector, the other two can be deduced    Float dx = PolynomialCal(radiusAndDegree.x,dir.x,dir.y,polynomialMap[3]); 
    // normalized direction vector needs no unit transformation
    Float dx = PolynomialCal(rho,dir.x,dir.y,polynomialMap[3]);
    Float dy = PolynomialCal(rho,dir.x,dir.y,polynomialMap[4]);
    Float dz; // See below

    //Float dz = std::sqrt(std::abs(1 - dx * dx - dy * dy));

     // If there is a direction vector dz polynomial given, use it,
    // If not, then deduce it from dx dy assuming the ray is not travelling backwards (dz>0)
    // This is used to enable RTF systems where outcoming rays travel backward (e.g. for wide angle lenses/fisheye)
    if(polynomialMap.size()==6){
        // Explicitly approxmiate dz
        dz = PolynomialCal(rho,dir.x,dir.y,polynomialMap[5]);
        

        // Depending on the sign we normalize the direction vector
        Float sign = 1;
        if(dz<0){ sign =-1;}
        dz = sign*std::sqrt(std::abs(1 - dx * dx - dy * dy));

        
        //std::cout << "Calculate dz: " << dz << "\n";
     }else{ 
         // Deduce dz
        dz = std::sqrt(std::abs(1 - dx * dx - dy * dy));
        //std::cout << "Deduce dz: " << dz << "\n";
    }

    if (1 - dx * dx - dy * dy < 0) {
        Warning("Problematic ray fitting.");
    }


   
    // The outocming Ray starts at the output plane: 
    Float outputPlane_z = (filmDistance+lensThickness+z); /// Lens thickness: WIHOUT input plane
  
    Ray rOut = Ray(Point3f(x, y, outputPlane_z), Vector3f(dx, dy, dz)); // Original

    return rOut;
}

// Rotate the around the Z axis by a given angle
inline Ray RTFCamera::RotateRays(const Ray &thisRay, Float deg) const{
    Transform rot = Rotate(deg, Vector3f(0, 0, 1));
    Ray rRot = rot(thisRay);
    return rRot;
}

// coefficients[i]*x^i
Float evalPolynomial(pstd::vector<Float> coefficients,Float x){
             Float result=0;
             for (int i = 0; i < coefficients.size(); i++) {
                 result = result + coefficients[i]*powerLoop(x,i);
             }
             return result;
}
/*
// A ray is valid if it passes through all circles on the circle plane
bool RTFCamera::IsValidRayCircles(const Ray &rotatedRayOnInputplane,RTFCamera::RTFVignettingTerms &vignetting) const{
    Vector3f dir = (rotatedRayOnInputplane.d);

    // The off axis distance on the input plane determines how much the circles change
    Float offaxisDistanceInputplane=rotatedRayOnInputplane.o.y; // Off axis distance IN input plane

     // To calculate the intersections with all circles we project the ray to the circle plane
     Float alpha = vignetting.circlePlaneZ / dir.z;  //  not needed
     Point3f pointOnCirclePlane = rotatedRayOnInputplane.o + alpha * dir; 
     //std::cout << rotatedRayOnInputplane.o.z << "\n";
     Float pointOnCirclePlaneXsquared = pointOnCirclePlane.x * pointOnCirclePlane.x; // only compute once
     // Because it is rotated, this x coordinate should be zero. Maybe we can do this to improve performance
     
     // The ray intersection with the input plane should be within all circles
     // If not, by construction the ray is vignetted (should not be traced)
         for (int i = 0; i < vignetting.circleRadii.size(); i++) {
            // Calculate the length
            
            
            Float radius;Float distanceFromCenterY;

            // This can be unified
            if(i==vignetting.exitpupilIndex){
                Float radius0= vignetting.circleRadii[i]; // This radius is going to be scaled by the polynomial

                // Apply nonlinear transform
                radius=radius0*evalPolynomial(vignetting.circleRadiusPoly,offaxisDistanceInputplane);
                distanceFromCenterY= pointOnCirclePlane.y-offaxisDistanceInputplane*evalPolynomial(vignetting.circleSensitivityPoly,offaxisDistanceInputplane);
                
            }else{
                // Apply linear transform
                distanceFromCenterY = pointOnCirclePlane.y-offaxisDistanceInputplane*vignetting.circleSensitivities[i];
                radius = vignetting.circleRadii[i];
            }

            // Test whether the point projected onto circle plane lies within all circles
            // Returns false when it finds the first circle which which did not intersect with the ray
            if (std::sqrt(pointOnCirclePlaneXsquared+ distanceFromCenterY*distanceFromCenterY) >= radius) {
                return false;
            }
    } 
    return true;
}
*/

// TODO Document
bool RTFCamera::TraceLensesFromFilm(
    const Ray &rayOnInputPlane, Ray *rOut, int wlIndex) const {

        Ray rLens = (rayOnInputPlane);  // No need to flip it for rtf calculations ( compare with omni)

    // STEP 1. Rotating so that the origin of the ray lies on the y axis.
    Vector2f radiusAndRotation = Pos2RadiusRotation(rLens.o);
    Ray rotatedRay = RotateRays(rLens, 90 - radiusAndRotation.y);
   // std::cout <<  "rotatedRay" << rotatedRay <<"\n";    
    
    // STEP 2. Determine whether the ray will be vignetted or not
    auto passnopass = passNoPassPerWavelength[wlIndex];
    if (!passnopass->isValidRay(rotatedRay)){
        //std::cout <<  "nopass" <<"\n";    
        return false;
    }
       // std::cout <<  "Pass" <<"\n";    
    
    // I think it should be normalized already
    Vector3f dir = Normalize(rotatedRay.d);

    // Select polynomial for given wavelength and apply it
    pstd::vector<RTFCamera::LensPolynomialTerm> polynomialMap = polynomialMaps[wlIndex];

    rLens = ApplyPolynomial(radiusAndRotation.x, dir, polynomialMap);
  //  std::cout <<  "rlens" << rLens <<"\n";    
    
    // Rotate the output ray back
    
    rLens = RotateRays(rLens, radiusAndRotation.y - 90);
    //std::cout <<  "roteded rlens" << rLens <<"\n";    

    if ((rOut != nullptr)) {
        *rOut = (rLens);
    } else {
        return false;
    }
    return true;
}

/*
inline Float RTFCamera::distanceCirclePlaneFromFilm() const {
    // In principle this should be made wavelength dependent, now just using the
    // vignetting term corresdponding to the first wavelength
    return (filmDistance - planeOffsetInput + vignettingTerms[0].circlePlaneZ);
}
*/
// BoundExitPupilRTF() takes an interval along the  axis on the film.
// It samples a series of points along the interval [pfilmX0 pfilmX1]
// For each point, it also samples a point on the bounding box of the rear lens
// element’s extent on the plane tangent to its rear. It computes the bounding
// box on the tangent plane of all of the rays that make it through the lens
// system from points along the interval.
//
// This adaptation uses the function isValidRayCircles  to determine the pass no
// pass region rather than having to trace through the lens.


Bounds2f RTFCamera::BoundExitPupilRTF(Float pFilmX0, Float pFilmX1) const {
// The two samples are along the X axis

    // Take first wavelength (one can consider extending to full spectral
    // information)
    int wavelengthIndex = 0;
   // RTFVignettingTerms vignetting = vignettingTerms[wavelengthIndex];

    auto passnopass = passNoPassPerWavelength[wavelengthIndex];
    
    
       


    Bounds2f pupilBounds;
    // Sample a collection of points on the rear lens to find exit pupil
    const int nSamples = 1024 * 1024;
    int nExitingRays = 0;

    //(std::cout <<"Nsamples " << nSamples << "\n";
    // Compute bounding box of projection of rear element on sampling plane
    Float rearRadius = passnopass->getOnAxisRadiusEstimate(); // Fix rear readius obtaining
    //std::cout <<"Rear Radius " << rearRadius << "\n";

    // Film distance is measured from the actual rear element
    // CirclePlaneZ is measured from the input plane of the RTF
    /// Offset is the offset of the input plane from the rear element
    // Float circlePlaneZFromFilm = distanceCirclePlaneFromFilm();

    
    Float pupilPlaneZFromFilm = (filmDistance-planeOffsetInput)+passnopass->distanceInputToIntersectPlane();
    //std::cout <<"PupilPlaneFromFilm " << pupilPlaneZFromFilm << "\n";

    // This is the default region. I make it twice as large in the hope that it will be enough to accomodate possible pupil walking
    // Make radius large enough so it covers 40   degrees half cone angle.
    
    // Create a margin of pupil bounds. The margin is chosen to get a 40 deg half cone pupil
    Float scale = pupilPlaneZFromFilm *std::tan(40*Pi/180) / rearRadius;
    //std::cout <<"Scale " << scale << "\n";

    //std::cout << "Scale " <<scale <<"\n";
    Bounds2f projRearBounds(Point2f(-scale * rearRadius, -scale * rearRadius),
                            Point2f(scale * rearRadius, scale * rearRadius));

   // generate samples
    for (int i = 0; i < nSamples; ++i) {
        // Find location of sample points on $x$ segment and rear lens element
        Point3f pFilm(Lerp((i + 0.5f) / nSamples, pFilmX0, pFilmX1), 0, 0); // Along X axis
        Float u[2] = {RadicalInverse(0, i), RadicalInverse(1, i)};

        // Generate a sample on the pupil plane. It uses the predefined maximum boundries to find a point in between.
        // This is problematic is the pupil starts walking ( changing in position with off axis distance).
        // It might be better to generate bounds using spherical coordinates, because then it actually starts looking for  the position
        Point3f pOnPupilPlane(
            Lerp(u[0], projRearBounds.pMin.x, projRearBounds.pMax.x),
            Lerp(u[1], projRearBounds.pMin.y, projRearBounds.pMax.y),   // set to zero
            pupilPlaneZFromFilm); 

        // Expand pupil bounds if ray makes it through the lens system
        // This ray should be continued to input plane
        Vector3f direction = Normalize(pOnPupilPlane - pFilm);
        Float alpha =
            (filmDistance-planeOffsetInput) /
            direction.z;  // Scaling factor to project it to input plane
        Point3f pointOnInputPlane = pFilm + alpha * direction;
        



        // Rotate the ray to make it ready for evaluation by the PassNoPass function
        Ray r = Ray(pointOnInputPlane, direction);
        auto radiusRotation = Pos2RadiusRotation(pointOnInputPlane);
        Ray rotatedRay = RotateRays(r, 90 - radiusRotation.y);

        if (Inside(Point2f(pOnPupilPlane.x, pOnPupilPlane.y), pupilBounds) ||
            passnopass->isValidRay(rotatedRay)) {
            pupilBounds =
                Union(pupilBounds, Point2f(pOnPupilPlane.x, pOnPupilPlane.y));
            ++nExitingRays;
        }
    }

    // Return entire element bounds if no rays made it through the lens system
    if (nExitingRays == 0) {
        LOG_VERBOSE("Unable to find exit pupil in x = [%f,%f] on film.", pFilmX0, pFilmX1);
        std::cout << "unable to find exit pupil \n";
        return projRearBounds;
    }
    
    // Expand bounds to account for sample spacing
    pupilBounds = Expand(pupilBounds, 2 * Length(projRearBounds.Diagonal()) /
                                          std::sqrt(nSamples));

    std::cout << 1e3*pFilmX0 << ":" << 1e3*pupilBounds.pMin << " - "  <<1e3*pupilBounds.pMax  <<  "\n";
    
    return pupilBounds;
}

// The function samples generates ray candidates based on the bounding boxes that
// were estimated during the creation of the object
//1. It generates a random sample in the box 
//2. Generates a corresponding ray
//3. Rotates to the proper off axis angle (because boxes where estimated for a specific off axis direction)
Point3f RTFCamera::SampleExitPupil(const Point2f &pFilm,
                                         const Point2f &lensSample,
                                         Float *sampleBoundsArea) const {
    
    // Find exit pupil bound for sample distance from film center
    Float diagonal = film.Diagonal();
    Float rFilm = std::sqrt(pFilm.x * pFilm.x + pFilm.y * pFilm.y);
    int rIndex = rFilm / (diagonal / 2) * exitPupilBoundsRTF.size();
    rIndex = std::min((int)exitPupilBoundsRTF.size() - 1, rIndex);
    Bounds2f pupilBounds = exitPupilBoundsRTF[rIndex];
    if (sampleBoundsArea) *sampleBoundsArea = pupilBounds.Area();

    // Generate sample point inside exit pupil bound
    Point2f pPupilPlane = pupilBounds.Lerp(lensSample);
    Float inputPlane_z=filmDistance-planeOffsetInput;

    //RTFVignettingTerms vignetting = vignettingTerms[0];
    auto passnopass = passNoPassPerWavelength[0];
    
    Float pupilPlaneZFromFilm = (filmDistance-planeOffsetInput)+passnopass->distanceInputToIntersectPlane();

    // Return sample point rotated by angle of _pFilm_ with $+x$ axis
    // THe exit pupil bounds were calculated for a single direction while the filmposition pFIlm 
    // can be be off axis in any direction. Therefore we first rotate the pupilplane position to the same off axis direction
    Float sinTheta = (rFilm != 0) ? pFilm.y / rFilm : 0;
    Float cosTheta = (rFilm != 0) ? pFilm.x / rFilm : 1;
    Point2f pRotatedOnPupilPlane(cosTheta * pPupilPlane.x - sinTheta * pPupilPlane.y,
                   sinTheta * pPupilPlane.x + cosTheta * pPupilPlane.y);


    // This functions needs to return a point that lies on the inputPlane_Z of the raytransfer function
    // Float ratio = inputPlane_z / (inputPlane_z + vignettingTerms[0].pupilPos[vignettingTerms[0].exitpupilIndex]);
//    Point2f pPointOnInputPlane = Lerp(ratio, pFilm, pRotatedOnPupilPlane);

    // Find interesectio on input plane
    Point3f film3d = Point3f(pFilm.x,pFilm.y,0);
    Vector3f dir = Point3f(pRotatedOnPupilPlane.x,pRotatedOnPupilPlane.y,pupilPlaneZFromFilm)-film3d;
    Float alpha = inputPlane_z/dir.z;
    Point3f pPointOnInputPlane=film3d+alpha*dir;


    return pPointOnInputPlane;
}


// This function is supposed to generate the output ray
pstd::optional<CameraRay> RTFCamera::GenerateRay(CameraSample sample,
                                                       SampledWavelengths &lambda) const {
    // Find point on film, _pFilm_, corresponding to _sample.pFilm_
    Point2f s(sample.pFilm.x / film.FullResolution().x,
              sample.pFilm.y / film.FullResolution().y);
    Point2f pFilm2 = physicalExtent.Lerp(s);
    Point3f pFilm(-pFilm2.x, pFilm2.y, 0);

    // We want to find the sample on the lens, this information is in sample.pLens. But how?
    Point3f pOnInputPlane;
        
    // Trace ray from _pFilm_ through RTF
    Float exitPupilBoundsArea; // Variable will be filled based on chosen sampling strategy below


    /// SAMPLING STRATEGY 1 : Use precomputed bounding pboxes (defective at the moment, but this will become the preferred way )
    pOnInputPlane = SampleExitPupil(Point2f(pFilm.x, pFilm.y), sample.pLens, &exitPupilBoundsArea);
    Float pupilArea=exitPupilBoundsArea;
    //std::cout << "poninputplane" <<pOnInputPlane <<"\n";
    //std::cout << "pupilarea" << pOnInputPlane <<"\n";


    // Construct the ray coming from the film to the point on the input plane
    Ray rFilm(pFilm, pOnInputPlane - pFilm);
    Ray rOriginOnInputPlane = Ray(pOnInputPlane, pOnInputPlane - pFilm);
    //std::cout <<  "roriigninput" << rOriginOnInputPlane <<"\n";    

    // CHoose which wavelength to use (and hence which polynomial)
    // The wavelengths provided by the lens file might not be equal to the wavelenth of the ray
    // The current strategy is to find the closest wavelength and use that for simulation.
    //  This assumes that the list of wavelenghts (polyWavelenghts_nm is an ordered list)  
    Ray ray;
    
    int wlIndex=0; 
    Float minError=INFINITY;
    for (int i=0; i<polyWavelengths_nm.size();i++){
        Float error = std::abs(polyWavelengths_nm[i]-ray.wavelength);
        
        if(error < minError){
            minError = error;
            wlIndex=i;
        }
        
    }
    bool result = TraceLensesFromFilm(rOriginOnInputPlane, &ray, wlIndex);
    //std::cout <<  "Tracelensfromfilm" << result <<"\n";    
    if (result == 0)
        return {};
    // Finish initialization of _OmniCamera_ ray
    ray.time = SampleTime(sample.time);
    ray.medium = medium;
    ray = RenderFromCamera(ray);
    ray.d = Normalize(ray.d);

    Float weight = 1;
    // Compute weighting for _OmniCamera_ ray
    Float cosTheta = Normalize(rFilm.d).z;
    // weight *= Pow<4>(cosTheta) / (eps->pdf * Sqr(LensRearZ()));
    weight *=Pow<4>(cosTheta) * pupilArea / (filmDistance*filmDistance);

    return CameraRay{ray, SampledSpectrum(weight)};
}



std::string RTFCamera::ToString() const
{
    return StringPrintf(
        "[ RTFCamera %s exitPupilBounds: %s ]",
        CameraBase::ToString(), exitPupilBounds);
}

RTFCamera *RTFCamera::Create(const ParameterDictionary &parameters,
                             const CameraTransform &cameraTransform,
                             Film film, Medium medium, const FileLoc *loc,
                             Allocator alloc)
{
    CameraBaseParameters cameraBaseParameters(cameraTransform, film, medium, parameters,
                                              loc);

    // RTF camera-specific parameters
    std::string lensFile = ResolveFilename(parameters.GetOneString("lensfile", ""));
    Float apertureDiameter = parameters.GetOneFloat("aperturediameter", -1);

    // Hard set the film distance
    Float filmDistance = parameters.GetOneFloat("filmdistance", 0);
    // Chromatic aberration flag
    bool caFlag = parameters.GetOneBool("chromaticAberrationEnabled", false);

    Point2f exitPupilBounds(0, INFINITY);

    if (lensFile.empty())
    {
        Error(loc, "No lens description file supplied!");
        return nullptr;
    }
    // Load element data from lens description file: main lens
    pstd::vector<RTFCamera::LensPolynomialTerm> poly;

    pstd::vector<Float> polyWavelengths_nm;                                   // wavelengths read from file
    pstd::vector<pstd::vector<RTFCamera::LensPolynomialTerm> > polynomialMaps; // Each element has corresponding wavelength
    pstd::vector<RTFCamera::RTFVignettingTerms> vignettingTerms;
    pstd::vector<std::shared_ptr<PassNoPass>> passNoPassPerWavelength;

    int pupilIndex = -1;

    auto endsWith = [](const std::string &str, const std::string &suffix)
    {
        return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
    };

    // if (!endsWith(lensFile, ".json")) {
    //     Error("Invalid format for lens specification file \"%s\".",
    //         lensFile.c_str());
    //     return nullptr;
    // }

    Float lensThickness = 0;
    Float planeOffsetInput = 0;
    Float planeOffsetOutput = 0;

    std::vector<Float> circleRadii;
    std::vector<Float> circleSensitivities;

    // read lens json file
    std::ifstream i(lensFile);
    json j;
    if (endsWith(lensFile, ".json"))
    {
        /*
         The format of polynomial lens would be:
         {
          "description": "equivalent lens poly",
          "name": "polynomial",
          "thickness": xxxx,
          "poly": [
           {
            "outputname": "outx",
            "termr":
            "termu":
            "termv":
            "coeff":
           },

          ]
         }

         Term and coeff will be stored in map.
         */
        // Polynomial lens case
        std::ifstream i(lensFile);
        json j;
        if (i && (i >> j))
        {
            // Write name and description
            if (j["name"].is_string())
            {
                LOG_VERBOSE("Loading polynomial lens %s.\n",
                                          j["name"].get<std::string>().c_str());
                std::cout << j["name"].get<std::string>().c_str() << "\n";
            }
            if (j["description"].is_string())
            {
                LOG_VERBOSE("%s\n", j["description"].get<std::string>().c_str());
            }

            if (j["thickness"].is_number())
            {
                lensThickness = (Float)j["thickness"] * 0.001f;
            }
            if (j["planeoffsetinput"].is_number())
            {
                planeOffsetInput = (Float)j["planeoffsetinput"] * 0.001f;
            }
            if (j["planeoffsetoutput"].is_number())
            {
                planeOffsetOutput = (Float)j["planeoffsetoutput"] * 0.001f;
            }
            if (j["planeoffset"].is_number())
            {
                Error("Deprecated variable 'planeoffset' in json, use separate variables for input and output instead: planeoffsetinput, planeoffsetoutput");
            }

            // When the the input plane is placed at the film, numerical rounding might cause it to be placed underneath the film, hencetracing rays in the wrong direction
            if ((filmDistance - planeOffsetInput) < 0)
            {
                Error("The input plane cannot be below the film. This may be a numerical rounding error when the input plane equals the input plane. Try adding a small offset to the film.");
                //                planeOffsetInput=filmDistance;
            }

            auto toTerms = [](json jterms)
            {
                pstd::vector<Float> res;
                if (jterms.is_null())
                {
                    return res;
                }

                for (int i = 0; i < jterms.size(); ++i)
                {
                    json thisnum = jterms[i];
                    if (!thisnum.is_number())
                    {
                        Error("Invalid polynomial in lens specification file must be an array of floats");
                    }
                    res.push_back((Float)thisnum);
                }
                return res;
            };

            auto toPolynomialStruct = [toTerms](json jp)
            {
                RTFCamera::LensPolynomialTerm result;
                result.name = jp["outputname"].get<std::string>();
                result.termr = toTerms(jp["termr"]);
                result.termu = toTerms(jp["termdx"]);
                result.termv = toTerms(jp["termdy"]);
                result.coeff = toTerms(jp["coeff"]);
                return result;
            };

            auto toLensPolynomialTerms = [toTerms](json jp)
            {
                RTFCamera::LensPolynomialTerm result;
                result.name = jp["outputname"].get<std::string>();
                result.termr = toTerms(jp["termr"]);
                result.termu = toTerms(jp["termdx"]);
                result.termv = toTerms(jp["termdy"]);
                result.coeff = toTerms(jp["coeff"]);
                return result;
            };
            
            

        auto toPassNoPassEllipse = [apertureDiameter, exitPupilBounds, pupilIndex, toTerms](json sp)
            {
                Float mm_to_meter = 0.001f;

                pstd::vector<Float> positions = toTerms(sp["positions"]);
                pstd::vector<Float> radiiX = toTerms(sp["radiiX"]);
                pstd::vector<Float> radiiY = toTerms(sp["radiiY"]);
                pstd::vector<Float> centersX = toTerms(sp["centersX"]);
                pstd::vector<Float> centersY = toTerms(sp["centersY"]);
                Float circlePlaneZ = mm_to_meter*((Float) sp["intersectPlaneDistance"]);



                // Convert mm to meter
                
                for (int i = 0; i < positions.size(); i++)
                {
                    // Convert from mm to meter
                    positions[i] = mm_to_meter * positions[i];
                    radiiX[i] = mm_to_meter * radiiX[i];
                    radiiY[i] = mm_to_meter * radiiY[i];
                    centersX[i] = mm_to_meter * centersX[i];
                    centersY[i] = mm_to_meter * centersY[i];
                }

                
                return std::make_shared<PassNoPassEllipse>(PassNoPassEllipse(positions,radiiX,radiiY,centersX,centersY,circlePlaneZ));
            };

//  This function could perhaps be encapsulated in the class as a statcif cuntion
            auto toRayPassCircles = [apertureDiameter, exitPupilBounds, pupilIndex, toTerms](json sp)
            {
                Float mm_to_meter = 0.001f;

                pstd::vector<Float> radii = toTerms(sp["radii"]);
                pstd::vector<Float> sensitivities = toTerms(sp["sensitivities"]);
                Float rayPassPlaneDistanceFromInput = mm_to_meter*((Float) sp["intersectPlaneDistance"]);



                // Convert mm to meter
                
                for (int i = 0; i < radii.size(); i++)
                {
                    // Convert from mm to meter
                    radii[i] = mm_to_meter * radii[i];

                    // No conversion required
                    sensitivities[i] = 1.0f*sensitivities[i];
                 }

                
                return std::make_shared<PassNoPassCircleIntersection>(PassNoPassCircleIntersection(radii,sensitivities,rayPassPlaneDistanceFromInput));
            };


    
          auto determinePassNoPass = [toPassNoPassEllipse,toRayPassCircles,apertureDiameter, exitPupilBounds, pupilIndex, toTerms](json sp)
            {

                auto method = (std::string)sp["method"];
                if(method == "minimalellipse"){
                    return std::dynamic_pointer_cast<PassNoPass>(toPassNoPassEllipse(sp));
                 }else if(method == "circles"){ 
                    // Perhaps ecapsulate this function within the class with a static function?
                    return std::dynamic_pointer_cast<PassNoPass>(toRayPassCircles(sp)); 
                 }else{
                    Error("Error unknown Ray Pass method:",
                                         method.c_str());
                    
                   return  std::dynamic_pointer_cast<PassNoPass>(toPassNoPassEllipse(sp)); 
                }
            };

            
            // TG: so after this loop exitpupilBounds = Point2f  (0, pupilRadii[pupilIndex])


           // std::cout << "Loop over all wavelengths" <<"\n";
            // Loop over all wavelengths
            int wlIndex = 0; // index counter for forloop
            auto polynomials = j["polynomials"];
            polyWavelengths_nm = pstd::vector<Float>(polynomials.size());
            vignettingTerms = pstd::vector<RTFCamera::RTFVignettingTerms>(polynomials.size());

            passNoPassPerWavelength = pstd::vector<std::shared_ptr<PassNoPass>>(polynomials.size());

            // Initialize map of polynomials
                // For each wavelength (outer vector), we have a vector of 6 polynomials (x,y,z,dx,dy,dz) (position and direciton vector)
                
                // Initialize vector depending on how many polynomials we use
                // This depends on whether the direction vector z polynomial is used.
                // By default we do not use and hence we assume that all rays come out with a positive value for dz = sqrt(1-dx^2 -dy^2)
                // When rays can exit the output plane or surface in arbitrary directions, we can use the polynomial for dz.
                // TG: Ideally what I would like to see is to leverage polymorphism and make a seperate class (like raypass) for the transfer functions.
                // This will encapsulate the specific implementation details about whether polynomials or for example neural
                // networks are used or whether planar of spherical output spheres are used.
                // For now, this dirty hack determines when to the polynomial use dz or not
                bool useDZpolynomial=false;
                try{

                    // Read from the json RTF file if present
                 useDZpolynomial = (bool) j["useDZpolynomial"];
                } catch (const std::exception& e) { /* */ } // The parameter was absent or not boolean.
                

                int nbTerms =5; // By default do not use dz polynomial
                if(useDZpolynomial){
                     nbTerms = 6; // Extra polynomial for dz
                }


           // std::cout << "Loop over polynomials" <<"\n";
            // nbTerms is used to initialize the vector of this length
            polynomialMaps = pstd::vector<pstd::vector<RTFCamera::LensPolynomialTerm>>(polynomials.size(), pstd::vector<RTFCamera::LensPolynomialTerm>(nbTerms));
            
            if (polynomials.is_array() && polynomials.size() > 0)
            {
                for (auto sp : polynomials)
                {
                    Float wavelength = (Float)sp["wavelength_nm"];
                    polyWavelengths_nm[wlIndex] = (Float)wavelength;

                    
                    // Read Ray Pass FUnction section from JSON file
                   json raypass = sp["raypass"];
                    if(raypass.is_null()){
                        raypass = sp["passnopass"]; //Backwards compatibility
                    }
                    // COnstruct raypass function for each wavelength
                    std::shared_ptr<PassNoPass> passNoPass = determinePassNoPass(raypass);
                    passNoPassPerWavelength[wlIndex] = passNoPass;
                    passNoPass->print();

                    // Read Polynomial Terms
                    auto jpoly = sp["poly"];
                    if (jpoly.is_array() && jpoly.size() > 0)
                    {

                        for (auto jp : jpoly)
                        {

                            // Polynomials for (x,y,z,d_x,d_y,d_z) are stored in a ordered list (vector)
                            std::string outputname = jp["outputname"].get<std::string>();

                            // There can be six polynomials, in principle.
                            // Altough (for now) we only use
                            // Based on the name in json file, assign a corresponding index to the polu
                            int polynomialmapIndex = -1;
                            if (outputname == "outx")
                            {
                                polynomialmapIndex = 0;
                            }
                            else if (outputname == "outy")
                            {
                                polynomialmapIndex = 1;
                            }
                            else if (outputname == "outz")
                            {
                                polynomialmapIndex = 2;

                                // Direction vectors
                            }
                            else if ((outputname == "outdx") || (outputname == "outu"))
                            {
                                polynomialmapIndex = 3;
                            }
                            else if ((outputname == "outdy") || (outputname == "outv"))
                            {
                                polynomialmapIndex = 4;

                                // Currently unused, could be used to determine sign of the Z component
                                // Which cannot be modeled by dx and dy alone. This would be relevant for a fisheye lens where outgoing rays can travel back toward the scene
                            }
                            else if ((outputname == "outdz") || (outputname == "outw"))
                            {
                                 if(useDZpolynomial){
                                       polynomialmapIndex=5;
                                 }else{
                                    continue; // Skip it if we dont need it
                                 }
                            }
                            else
                            {
                                Error("Error, output name  \"%s\" does not match known value.",
                                      outputname.c_str());
                            }

                            polynomialMaps[wlIndex][polynomialmapIndex] = toLensPolynomialTerms(jp);
                        }
                    }
                    else
                    {
                        Error("Error, invalid polynomial specification \"%s\".",
                              lensFile.c_str());
                        return nullptr;
                    }
                    wlIndex = wlIndex + 1; // Next wavelength index
                }
            }
            else
            {
                Error("Error, invalid polynomial specification \"%s\".",
                      lensFile.c_str());
                return nullptr;
            }
        }
        else
        {
            Error("Error reading lens specification file \"%s\".",
                  lensFile.c_str());
            return nullptr;
        }
    }
    else if (endsWith(lensFile, ".py"))
    {
        // Might use a neural network as equivalent lens model in the future
    }
    else
    {
        Error("Invalid format for lens specification file \"%s\".",
              lensFile.c_str());
        return nullptr;
    }

    std::string bbmode = parameters.GetOneString("bbmode", "polynomial");

    return alloc.new_object<RTFCamera>(cameraBaseParameters, 
                                        bbmode, 
                                        filmDistance, caFlag, apertureDiameter,
                                        planeOffsetInput,planeOffsetOutput,lensThickness, 
                                        polynomialMaps, 
                                        passNoPassPerWavelength,
                                        polyWavelengths_nm,
                                        alloc);
}


}  // namespace pbrt
