@preconcurrency import MLX
@preconcurrency import MLXNN

// MARK: - MultiresConvDecoder
//
// Implements DPT-style multi-resolution decoder.
// All features are in NHWC [N, H, W, C] format.
//
// Python key layout:
//   convs.0   – Conv2d or Identity (no key when Identity)
//   convs.1..4 – Conv2d
//   fusions.0..4 – FeatureFusionBlock2d (fusions.0 has no deconv)

final class MultiresConvDecoder: Module {
    // convs[0] may be an identity passthrough (nil = identity)
    @ModuleInfo var convs: [Conv2d]
    @ModuleInfo var fusions: [FeatureFusionBlock2d]
    let conv0IsIdentity: Bool

    init(dimsEncoder: [Int], dimsDecoder: [Int]) {
        precondition(dimsEncoder.count == dimsDecoder.count)
        let n = dimsEncoder.count
        conv0IsIdentity = dimsEncoder[0] == dimsDecoder[0]

        // Build convs: index 0 if not identity, then 1..n-1
        var convList: [Conv2d] = []
        if !conv0IsIdentity {
            convList.append(Conv2d(inputChannels: dimsEncoder[0], outputChannels: dimsDecoder[0],
                                   kernelSize: 1, bias: false))
        }
        for i in 1..<n {
            convList.append(Conv2d(inputChannels: dimsEncoder[i], outputChannels: dimsDecoder[i],
                                   kernelSize: 3, padding: 1, bias: false))
        }
        convs = convList

        // Build fusions: fusions[0] has no upsampling, fusions[1..] do
        var fusionList: [FeatureFusionBlock2d] = []
        fusionList.append(FeatureFusionBlock2d(dimIn: dimsDecoder[0], dimOut: dimsDecoder[0],
                                               upsample: false))
        for i in 1..<n {
            let dimOut = dimsDecoder[i - 1]
            fusionList.append(FeatureFusionBlock2d(dimIn: dimsDecoder[i], dimOut: dimOut,
                                                   upsample: true))
        }
        fusions = fusionList
        super.init()
    }

    /// encodings: list of NHWC feature maps, ordered from highest to lowest resolution
    func callAsFunction(_ encodings: [MLXArray]) -> MLXArray {
        let n = encodings.count

        func applyConv(_ i: Int, _ feat: MLXArray) -> MLXArray {
            if conv0IsIdentity && i == 0 { return feat }
            let ci = conv0IsIdentity ? i - 1 : i
            return convs[ci](feat)
        }

        // Process from lowest resolution (index n-1) to highest (index 0)
        var features = applyConv(n - 1, encodings[n - 1])
        features = fusions[n - 1](features)
        for i in stride(from: n - 2, through: 0, by: -1) {
            let skip = applyConv(i, encodings[i])
            features = fusions[i](features, skip)
        }
        return features
    }
}
