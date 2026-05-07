@preconcurrency import MLX
@preconcurrency import MLXNN

// MARK: - UNet conv_in wrapper
// Python: nn.Sequential(Conv2d[0], GroupNorm[1], ReLU[2])
// @ModuleInfo var (no key); assign before super.init().

final class UNetConvIn: Module {
    @ModuleInfo var conv: Conv2d
    @ModuleInfo var norm: GroupNorm

    init(dimIn: Int, dimOut: Int, numGroups: Int) {
        conv = Conv2d(inputChannels: dimIn, outputChannels: dimOut, kernelSize: 3, padding: 1)
        norm = GroupNorm(groupCount: numGroups, dimensions: dimOut, pytorchCompatible: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(norm(conv(x)))
    }
}

// MARK: - UNet encoder step
// Python: nn.Sequential(AvgPool2d[0], GNResBlock[1], GNResBlock[2])
// AvgPool2d has no weights; applied inline. @ModuleInfo var (no key); before super.init().

final class UNetEncoderStep: Module {
    @ModuleInfo var res1: GNResidualBlock
    @ModuleInfo var res2: GNResidualBlock

    init(dimIn: Int, dimOut: Int, numGroups: Int) {
        let dimHidden = dimOut / 2
        res1 = GNResidualBlock(dimIn: dimIn, dimHidden: dimHidden, dimOut: dimOut,
                               numGroups: numGroups)
        res2 = GNResidualBlock(dimIn: dimOut, dimHidden: dimHidden, dimOut: dimOut,
                               numGroups: numGroups)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pooled = AvgPool2d(kernelSize: IntOrPair(2), stride: IntOrPair(2))(x)
        return res2(res1(pooled))
    }
}

// MARK: - UNetEncoder

final class UNetEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: UNetConvIn
    @ModuleInfo(key: "convs_down") var convsDown: [UNetEncoderStep]

    init(dimIn: Int, widths: [Int], steps: Int, numGroups: Int = 8) {
        let steps_ = steps
        let widths_ = widths
        let numGroups_ = numGroups
        super.init()
        convIn = UNetConvIn(dimIn: dimIn, dimOut: widths_[0], numGroups: numGroups_)
        convsDown = (0..<steps_).map { i in
            UNetEncoderStep(dimIn: widths_[i], dimOut: widths_[i + 1], numGroups: numGroups_)
        }
    }

    /// x: NHWC → [feature0, feature1, ..., featureN] all NHWC
    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var features: [MLXArray] = []
        var feat = convIn(x)
        features.append(feat)
        for step in convsDown {
            feat = step(feat)
            features.append(feat)
        }
        return features
    }
}

// MARK: - UNet decoder step
// Python: nn.Sequential(Upsample[0], GNResBlock[1], GNResBlock[2])
// Upsample has no weights; applied inline. @ModuleInfo var (no key); before super.init().

final class UNetDecoderStep: Module {
    @ModuleInfo var res1: GNResidualBlock
    @ModuleInfo var res2: GNResidualBlock

    init(dimIn: Int, dimOut: Int, numGroups: Int) {
        let dimHidden = dimOut / 2
        res1 = GNResidualBlock(dimIn: dimIn, dimHidden: dimHidden, dimOut: dimOut,
                               numGroups: numGroups)
        res2 = GNResidualBlock(dimIn: dimOut, dimHidden: dimHidden, dimOut: dimOut,
                               numGroups: numGroups)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let up = Upsample(scaleFactor: [2, 2], mode: .nearest)(x)
        return res2(res1(up))
    }
}

// MARK: - UNet conv_out wrapper
// Python: nn.Sequential(GroupNorm[0], ReLU[1], Conv2d[2], GroupNorm[3], ReLU[4])
// @ModuleInfo var (no key); assign before super.init().

final class UNetConvOut: Module {
    @ModuleInfo var norm1: GroupNorm
    @ModuleInfo var conv: Conv2d
    @ModuleInfo var norm2: GroupNorm

    init(dimIn: Int, dimOut: Int, numGroups: Int) {
        norm1 = GroupNorm(groupCount: numGroups, dimensions: dimIn, pytorchCompatible: true)
        conv = Conv2d(inputChannels: dimIn, outputChannels: dimOut, kernelSize: 1)
        norm2 = GroupNorm(groupCount: numGroups, dimensions: dimOut, pytorchCompatible: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(norm2(conv(relu(norm1(x)))))
    }
}

// MARK: - UNetDecoder

final class UNetDecoder: Module {
    @ModuleInfo(key: "convs_up") var convsUp: [UNetDecoderStep]
    @ModuleInfo(key: "conv_out") var convOut: UNetConvOut

    init(dimOut: Int, widths: [Int], steps: Int, numGroups: Int = 8) {
        let inputDims = Array(widths.reversed().prefix(steps + 1))
        let lastWidth = inputDims.last!
        super.init()
        convsUp = (0..<steps).map { i in
            let dimIn = i == 0 ? inputDims[0] : inputDims[i] * 2
            let dimOutStep = inputDims[i + 1]
            return UNetDecoderStep(dimIn: dimIn, dimOut: dimOutStep, numGroups: numGroups)
        }
        convOut = UNetConvOut(dimIn: lastWidth * 2, dimOut: dimOut, numGroups: numGroups)
    }

    /// features: encoder outputs from shallow to deep [feat0, feat1, ..., featN]
    func callAsFunction(_ features: [MLXArray]) -> MLXArray {
        var iFeature = features.count - 1
        var out = convsUp[0](features[iFeature])
        iFeature -= 1
        for step in convsUp.dropFirst() {
            out = step(concatenated([out, features[iFeature]], axis: -1))
            iFeature -= 1
        }
        return convOut(concatenated([out, features[iFeature]], axis: -1))
    }
}
