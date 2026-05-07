@preconcurrency import MLX
@preconcurrency import MLXNN

// MARK: - Affine normalizer (no weights)

/// Maps input_range → output_range linearly. Default: [0,1] → [-1,1]  (2x - 1).
struct AffineNormalizer {
    let scale: Float
    let bias: Float

    init(inputRange: (Float, Float) = (0, 1), outputRange: (Float, Float) = (-1, 1)) {
        scale = (outputRange.1 - outputRange.0) / (inputRange.1 - inputRange.0)
        bias = outputRange.0 - inputRange.0 * scale
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * scale + bias
    }
}

// MARK: - FeatureFusionBlock residual (ReLU → Conv → ReLU → Conv)
// Python: nn.Sequential(ReLU[0], Conv2d[1], ReLU[2], Conv2d[3])
// @ModuleInfo var (no key) → property name is the lookup key; assign before super.init().

final class FFBResidualSequence: Module {
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var conv2: Conv2d

    init(features: Int) {
        conv1 = Conv2d(inputChannels: features, outputChannels: features,
                       kernelSize: 3, padding: 1)
        conv2 = Conv2d(inputChannels: features, outputChannels: features,
                       kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2(relu(conv1(relu(x))))
    }
}

final class FFBResidualBlock: Module {
    @ModuleInfo var residual: FFBResidualSequence

    init(features: Int) {
        residual = FFBResidualSequence(features: features)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + residual(x)
    }
}

// MARK: - GroupNorm residual (GN → ReLU → Conv → GN → ReLU → Conv)
// Python: nn.Sequential(GN[0], ReLU[1], Conv2d[2], GN[3], ReLU[4], Conv2d[5])
// @ModuleInfo var (no key); assign before super.init().

final class GNResidualSequence: Module {
    @ModuleInfo var norm1: GroupNorm
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var norm2: GroupNorm
    @ModuleInfo var conv2: Conv2d

    init(dimIn: Int, dimHidden: Int, dimOut: Int, numGroups: Int = 8) {
        norm1 = GroupNorm(groupCount: numGroups, dimensions: dimIn, pytorchCompatible: true)
        conv1 = Conv2d(inputChannels: dimIn, outputChannels: dimHidden, kernelSize: 3, padding: 1)
        norm2 = GroupNorm(groupCount: numGroups, dimensions: dimHidden, pytorchCompatible: true)
        conv2 = Conv2d(inputChannels: dimHidden, outputChannels: dimOut, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = norm1(x)
        out = relu(out)
        out = conv1(out)
        out = norm2(out)
        out = relu(out)
        out = conv2(out)
        return out
    }
}

final class GNResidualBlock: Module {
    @ModuleInfo var residual: GNResidualSequence
    @ModuleInfo var shortcut: Conv2d?

    init(dimIn: Int, dimHidden: Int, dimOut: Int, numGroups: Int = 8) {
        residual = GNResidualSequence(dimIn: dimIn, dimHidden: dimHidden, dimOut: dimOut,
                                      numGroups: numGroups)
        shortcut = dimIn != dimOut
            ? Conv2d(inputChannels: dimIn, outputChannels: dimOut, kernelSize: 1)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let res = residual(x)
        let skip = shortcut?(x) ?? x
        return skip + res
    }
}

// MARK: - FeatureFusionBlock2d

final class FeatureFusionBlock2d: Module {
    @ModuleInfo var resnet1: FFBResidualBlock
    @ModuleInfo var resnet2: FFBResidualBlock
    @ModuleInfo var deconv: ConvTransposed2d?
    @ModuleInfo(key: "out_conv") var outConv: Conv2d

    init(dimIn: Int, dimOut: Int? = nil, upsample: Bool = false) {
        let out = dimOut ?? dimIn
        resnet1 = FFBResidualBlock(features: dimIn)
        resnet2 = FFBResidualBlock(features: dimIn)
        deconv = upsample
            ? ConvTransposed2d(inputChannels: dimIn, outputChannels: dimIn,
                               kernelSize: 2, stride: 2, bias: false)
            : nil
        super.init()
        outConv = Conv2d(inputChannels: dimIn, outputChannels: out, kernelSize: 1)
    }

    /// x0: main features, x1: optional skip connection features
    func callAsFunction(_ x0: MLXArray, _ x1: MLXArray? = nil) -> MLXArray {
        var x = x0
        if let x1 {
            x = x + resnet1(x1)
        }
        x = resnet2(x)
        if let deconv {
            x = deconv(x)
        }
        x = outConv(x)
        return x
    }
}

// MARK: - ProjectUpsampleBlock (Conv2d + up to 3× ConvTranspose2d)
// Python: nn.Sequential(Conv2d[0], ConvT[1], ConvT[2]?, ConvT[3]?)
// @ModuleInfo var (no key); assign before super.init().

final class ProjectUpsampleBlock: Module {
    @ModuleInfo var proj: Conv2d
    @ModuleInfo var up1: ConvTransposed2d
    @ModuleInfo var up2: ConvTransposed2d?
    @ModuleInfo var up3: ConvTransposed2d?

    init(dimIn: Int, dimOut: Int, numUpsampleLayers: Int, dimIntermediate: Int? = nil) {
        let mid = dimIntermediate ?? dimOut
        proj = Conv2d(inputChannels: dimIn, outputChannels: mid, kernelSize: 1, bias: false)
        up1 = ConvTransposed2d(inputChannels: mid, outputChannels: dimOut,
                               kernelSize: 2, stride: 2, bias: false)
        up2 = numUpsampleLayers >= 2
            ? ConvTransposed2d(inputChannels: dimOut, outputChannels: dimOut,
                               kernelSize: 2, stride: 2, bias: false)
            : nil
        up3 = numUpsampleLayers >= 3
            ? ConvTransposed2d(inputChannels: dimOut, outputChannels: dimOut,
                               kernelSize: 2, stride: 2, bias: false)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = proj(x)
        out = up1(out)
        if let up2 { out = up2(out) }
        if let up3 { out = up3(out) }
        return out
    }
}
