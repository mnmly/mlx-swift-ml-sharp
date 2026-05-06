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
    /// Returns shape `[1, 3, H, W]`.
    public func preprocess(_ image: CGImage) -> MLXArray {
        let W = internalResolution.width
        let H = internalResolution.height

        let bytesPerPixel = 4
        let bytesPerRow = W * bytesPerPixel
        var pixels = [UInt8](repeating: 0, count: H * bytesPerRow)

        guard
            let cs = CGColorSpace(name: CGColorSpace.sRGB),
            let ctx = CGContext(
                data: &pixels,
                width: W, height: H,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        else {
            fatalError("SharpPipeline: failed to create CGContext for preprocessing")
        }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: W, height: H))

        // RGBA UInt8 → RGB Float32 [0,1] in NCHW
        var r = [Float](repeating: 0, count: H * W)
        var g = [Float](repeating: 0, count: H * W)
        var b = [Float](repeating: 0, count: H * W)
        for i in 0..<H * W {
            r[i] = Float(pixels[i * 4 + 0]) / 255.0
            g[i] = Float(pixels[i * 4 + 1]) / 255.0
            b[i] = Float(pixels[i * 4 + 2]) / 255.0
        }
        // Stack channels: [3, H*W] → [1, 3, H, W]
        let rgb = concatenated([MLXArray(r, [1, 1, H, W]),
                                MLXArray(g, [1, 1, H, W]),
                                MLXArray(b, [1, 1, H, W])], axis: 1)
        return rgb
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
    ///     width. Pass `imageWidth * 0.58` as a reasonable default for typical
    ///     smartphone images when the true focal length is unknown.
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
