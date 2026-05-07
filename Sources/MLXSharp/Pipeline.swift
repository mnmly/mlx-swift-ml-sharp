import CoreGraphics
import Foundation
@preconcurrency import MLX

// MARK: - SharpPrediction

/// Result of a Sharp forward pass, including Gaussians and input metadata.
public struct SharpPrediction {
    /// Predicted 3D Gaussians flattened to [B, N, *].
    public let gaussians: Gaussians3D
    /// Width of the image that was fed to the model (after resizing).
    public let processedWidth: Int
    /// Height of the image that was fed to the model (after resizing).
    public let processedHeight: Int
    /// Original input image width in pixels.
    public let originalWidth: Int
    /// Original input image height in pixels.
    public let originalHeight: Int
    /// Camera focal length in pixels at the original image width.
    public let focalLengthPixels: Float
}

// MARK: - SharpPipeline

public struct SharpPipeline {
    public let predictor: RGBGaussianPredictor
    /// Resolution the model runs at internally (width, height).
    public let internalResolution: (width: Int, height: Int)

    public init(
        predictor: RGBGaussianPredictor,
        internalResolution: (width: Int, height: Int) = (1536, 1536)
    ) {
        self.predictor = predictor
        self.internalResolution = internalResolution
    }

    // MARK: Preprocessing

    /// Convert a `CGImage` to an NCHW float32 `MLXArray` in `[0, 1]`.
    ///
    /// The image is rescaled to `internalResolution` (no padding, no aspect-ratio
    /// preservation) because the SPN sliding-pyramid operates on a fixed grid.
    /// Resampling uses bilinear interpolation with `align_corners=True` semantics
    /// (output corner pixels exactly sample input corner pixels), matching
    /// PyTorch's `F.interpolate(mode="bilinear", align_corners=True)` used by
    /// `python/ml-sharp/src/sharp/cli/predict.py:predict_image`.
    /// Returns shape `[1, 3, H, W]`.
    public func preprocess(_ image: CGImage) -> MLXArray {
        let dstW = internalResolution.width
        let dstH = internalResolution.height
        let srcW = image.width
        let srcH = image.height

        // 1) Draw the source CGImage at its native resolution into an RGBA8 buffer.
        //    No interpolation happens here — CGContext just transfers pixels and
        //    handles colorspace/alpha. This is intentional: PyTorch reads the
        //    image at native resolution, normalizes to fp32 [0,1], and only then
        //    resamples — doing the resample in fp32 (not uint8) avoids
        //    quantization noise.
        let bytesPerPixel = 4
        let srcRowBytes = srcW * bytesPerPixel
        var srcPixels = [UInt8](repeating: 0, count: srcH * srcRowBytes)
        guard
            let cs = CGColorSpace(name: CGColorSpace.sRGB),
            let srcCtx = CGContext(
                data: &srcPixels,
                width: srcW, height: srcH,
                bitsPerComponent: 8,
                bytesPerRow: srcRowBytes,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        else {
            fatalError("SharpPipeline: failed to create CGContext for preprocessing")
        }
        srcCtx.draw(image, in: CGRect(x: 0, y: 0, width: srcW, height: srcH))

        // 2) Split into per-channel fp32 [0,1] planes at native resolution.
        let srcCount = srcH * srcW
        var rPlane = [Float](repeating: 0, count: srcCount)
        var gPlane = [Float](repeating: 0, count: srcCount)
        var bPlane = [Float](repeating: 0, count: srcCount)
        srcPixels.withUnsafeBufferPointer { px in
            for i in 0..<srcCount {
                rPlane[i] = Float(px[i * 4 + 0]) / 255.0
                gPlane[i] = Float(px[i * 4 + 1]) / 255.0
                bPlane[i] = Float(px[i * 4 + 2]) / 255.0
            }
        }

        // 3) Bilinear-resample to (dstH, dstW) with align_corners=True.
        let r = SharpPipeline.bilinearAlignCorners(
            plane: rPlane, srcH: srcH, srcW: srcW, dstH: dstH, dstW: dstW)
        let g = SharpPipeline.bilinearAlignCorners(
            plane: gPlane, srcH: srcH, srcW: srcW, dstH: dstH, dstW: dstW)
        let b = SharpPipeline.bilinearAlignCorners(
            plane: bPlane, srcH: srcH, srcW: srcW, dstH: dstH, dstW: dstW)

        // 4) Stack channels: [1, 1, H, W] x3 → [1, 3, H, W].
        return concatenated([
            MLXArray(r, [1, 1, dstH, dstW]),
            MLXArray(g, [1, 1, dstH, dstW]),
            MLXArray(b, [1, 1, dstH, dstW])
        ], axis: 1)
    }

    /// Bilinear resample a single fp32 plane with `align_corners=True` semantics.
    /// Output pixel (j, i) samples input position
    /// `(j*(srcH-1)/(dstH-1), i*(srcW-1)/(dstW-1))`, so output corner pixels
    /// exactly coincide with input corner pixels.
    private static func bilinearAlignCorners(
        plane: [Float], srcH: Int, srcW: Int, dstH: Int, dstW: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: dstH * dstW)
        let scaleY: Float = (srcH > 1) ? Float(srcH - 1) / Float(dstH - 1) : 0
        let scaleX: Float = (srcW > 1) ? Float(srcW - 1) / Float(dstW - 1) : 0

        // Precompute X-axis sample indices and weights — reused across rows.
        var x0s = [Int](repeating: 0, count: dstW)
        var x1s = [Int](repeating: 0, count: dstW)
        var wxs = [Float](repeating: 0, count: dstW)
        for i in 0..<dstW {
            let x = Float(i) * scaleX
            let x0 = Int(x.rounded(.down))
            x0s[i] = x0
            x1s[i] = min(x0 + 1, srcW - 1)
            wxs[i] = x - Float(x0)
        }

        plane.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for j in 0..<dstH {
                    let y = Float(j) * scaleY
                    let y0 = Int(y.rounded(.down))
                    let y1 = min(y0 + 1, srcH - 1)
                    let wy = y - Float(y0)
                    let row0 = y0 * srcW
                    let row1 = y1 * srcW
                    let dstBase = j * dstW
                    for i in 0..<dstW {
                        let x0 = x0s[i], x1 = x1s[i], wx = wxs[i]
                        let p00 = src[row0 + x0], p01 = src[row0 + x1]
                        let p10 = src[row1 + x0], p11 = src[row1 + x1]
                        let p0 = p00 * (1 - wx) + p01 * wx
                        let p1 = p10 * (1 - wx) + p11 * wx
                        dst[dstBase + i] = p0 * (1 - wy) + p1 * wy
                    }
                }
            }
        }
        return out
    }

    // MARK: Inference

    /// Run inference on a pre-processed NCHW image `[1, 3, H, W]` in `[0, 1]`.
    public func predict(imageNCHW: MLXArray, focalLengthPixels: Float) -> Gaussians3D {
        let width = imageNCHW.shape[3]
        let disparityFactor = MLXArray([focalLengthPixels / Float(width)], [1])
        return predictor(imageNCHW, disparityFactor: disparityFactor)
    }

    /// End-to-end inference from a `CGImage`.
    ///
    /// - Parameters:
    ///   - image: Input image in any size/color space supported by CoreGraphics.
    ///   - focalLengthPixels: Camera focal length in pixels at the *original* image
    ///     width.  Compute as  f_mm * sqrt(w²+h²) / sqrt(36²+24²)  and default to
    ///     30 mm when the true 35mm-equivalent focal length is unknown.
    public func callAsFunction(
        _ image: CGImage,
        focalLengthPixels: Float
    ) -> SharpPrediction {
        let originalWidth = image.width
        let originalHeight = image.height
        let nchw = preprocess(image)
        // Python uses f_px / original_width (not processed width) for disparity factor.
        let disparityFactor = MLXArray([focalLengthPixels / Float(originalWidth)], [1])
        let gaussians = predictor(nchw, disparityFactor: disparityFactor)
        eval(gaussians.meanVectors, gaussians.singularValues,
             gaussians.quaternions, gaussians.colors, gaussians.opacities)
        return SharpPrediction(
            gaussians: gaussians,
            processedWidth: internalResolution.width,
            processedHeight: internalResolution.height,
            originalWidth: originalWidth,
            originalHeight: originalHeight,
            focalLengthPixels: focalLengthPixels
        )
    }
}

// MARK: - fromPretrained factory

public extension SharpPipeline {
    /// Load a `SharpPipeline` from a safetensors weights file on disk.
    static func fromPretrained(
        _ url: URL,
        internalResolution: (width: Int, height: Int) = (width: 1536, height: 1536)
    ) throws -> SharpPipeline {
        let predictor = try loadSharpModel(from: url)
        return SharpPipeline(predictor: predictor, internalResolution: internalResolution)
    }
}

// MARK: - Sharp namespace

/// Top-level convenience namespace matching the `mlx-swift-*` library convention.
///
/// Usage:
/// ```swift
/// let pipeline = try Sharp.fromPretrained(weightsURL)
/// let result = pipeline(cgImage, focalLengthPixels: 1200)
/// ```
public enum Sharp {
    /// Load a `SharpPipeline` from a safetensors weights file.
    public static func fromPretrained(
        _ url: URL,
        internalResolution: (width: Int, height: Int) = (width: 1536, height: 1536)
    ) throws -> SharpPipeline {
        try SharpPipeline.fromPretrained(url, internalResolution: internalResolution)
    }
}
