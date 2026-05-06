@preconcurrency import MLX
@preconcurrency import MLXNN
import Darwin

// MARK: - LearnedAlignment (scale_map_estimator)
// Uses a UNet to estimate a per-pixel scale map aligning monodepth to gt depth.

final class LearnedAlignment: Module {
    @ModuleInfo var encoder: UNetEncoder
    @ModuleInfo var decoder: UNetDecoder
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(
        dimIn: Int = 2,
        steps: Int = 4,
        stride: Int = 1,
        baseWidth: Int = 16,
        numGroups: Int = 4
    ) {
        let widths = (0...steps).map { i in min(baseWidth << i, 1024) }
        let stepsDecoder = steps - Int(log2f(Float(max(stride, 1))))
        // @ModuleInfo var (no key) — assign before super.init()
        encoder = UNetEncoder(dimIn: dimIn, widths: widths, steps: steps, numGroups: numGroups)
        decoder = UNetDecoder(dimOut: widths[0], widths: widths,
                              steps: max(stepsDecoder, 1), numGroups: numGroups)
        super.init()
        // @ModuleInfo(key:) — assign after super.init()
        convOut = Conv2d(inputChannels: widths[0], outputChannels: 1, kernelSize: 1)
    }

    /// src, tgt: NHWC [B, H, W, 1] (single-channel depth maps, metric)
    /// Returns alignment map NHWC [B, H, W, 1]
    func callAsFunction(
        _ src: MLXArray,
        _ tgt: MLXArray,
        depthDecoderFeatures: MLXArray? = nil
    ) -> MLXArray {
        let invSrc = 1.0 / clip(src, min: MLXArray(1e-4 as Float))
        let invTgt = 1.0 / clip(tgt, min: MLXArray(1e-4 as Float))
        let input = concatenated([invSrc, invTgt], axis: -1)  // [B, H, W, 2]
        let features = encoder(input)
        let out = convOut(decoder(features))
        let alignMap = exp(out)  // exp activation
        // Upsample to src resolution if needed
        let srcH = src.shape[1], srcW = src.shape[2]
        let outH = alignMap.shape[1], outW = alignMap.shape[2]
        if outH != srcH || outW != srcW {
            return Upsample(
                scaleFactor: .array([Float(srcH) / Float(outH), Float(srcW) / Float(outW)]),
                mode: .linear(alignCorners: false)
            )(alignMap)
        }
        return alignMap
    }
}

// MARK: - DepthAlignmentModule
// Wraps LearnedAlignment; key "scale_map_estimator" matches Python attribute name.

final class DepthAlignmentModule: Module {
    @ModuleInfo(key: "scale_map_estimator") var scaleMapEstimator: LearnedAlignment

    init(scaleMapEstimator sm: LearnedAlignment) {
        super.init()
        scaleMapEstimator = sm
    }

    /// monodepth: NHWC [B, H, W, numLayers] (metric depth)
    /// depth: optional NHWC gt depth [B, H, W, 1]
    func callAsFunction(
        _ monodepth: MLXArray,
        _ depth: MLXArray?,
        _ depthDecoderFeatures: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        guard let depth = depth else {
            return (monodepth, MLXArray.ones(monodepth.shape))
        }
        let firstLayer = monodepth[0..., 0..., 0..., 0..<1]
        let alignmentMap = scaleMapEstimator(firstLayer, depth)
        return (alignmentMap * monodepth, alignmentMap)
    }
}

extension DepthAlignmentModule: DepthAligning {}
