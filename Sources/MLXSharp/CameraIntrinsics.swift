import CoreGraphics
import Foundation
import ImageIO

/// Helpers for resolving the focal length value that
/// `SharpPipeline.callAsFunction(_:focalLengthPixels:)` expects.
///
/// Mirrors the conversion + EXIF-fallback logic in
/// `python/ml-sharp/src/sharp/utils/io.py:load_rgb`.
public enum SharpCameraIntrinsics {

    /// Convert a 35mm-equivalent focal length in millimetres to pixels at the
    /// given image dimensions, using a full-frame sensor diagonal as the
    /// reference. Pure function; no I/O.
    ///
    /// `f_px = f_mm * sqrt(W² + H²) / sqrt(36² + 24²)`
    public static func focalLengthPixels(
        focalMM: Float,
        imageWidth: Int,
        imageHeight: Int
    ) -> Float {
        let diag = sqrt(Float(imageWidth * imageWidth + imageHeight * imageHeight))
        let sensorDiag = sqrt(Float(36 * 36 + 24 * 24))
        return focalMM * diag / sensorDiag
    }

    /// EXIF-aware variant. Reads `FocalLengthIn35mmFilm` if present (and ≥ 1),
    /// otherwise falls back to plain `FocalLength` — applying the
    /// `< 10mm → ×8.4` heuristic for sub-10mm values, matching
    /// `sharp/utils/io.py:load_rgb`. If neither tag is present, uses
    /// `defaultFocalMM`.
    public static func focalLengthPixels(
        from imageSource: CGImageSource,
        imageWidth: Int,
        imageHeight: Int,
        defaultFocalMM: Float = 30
    ) -> Float {
        let mm = focalLengthMM(from: imageSource) ?? defaultFocalMM
        return focalLengthPixels(focalMM: mm, imageWidth: imageWidth, imageHeight: imageHeight)
    }

    /// Resolve the 35mm-equivalent focal length (in millimetres) from an
    /// image's EXIF metadata, or `nil` if neither expected tag is present.
    public static func focalLengthMM(from imageSource: CGImageSource) -> Float? {
        guard let props = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any],
              let exif  = props[kCGImagePropertyExifDictionary] as? [CFString: Any]
        else { return nil }

        if let f35 = exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? Double, f35 >= 1 {
            return Float(f35)
        }
        if let f = exif[kCGImagePropertyExifFocalLength] as? Double {
            return Float(f < 10 ? f * 8.4 : f)
        }
        return nil
    }
}
