@preconcurrency import MLX
@preconcurrency import MLXNN

// MARK: - SkipConvBackbone
// A thin wrapper around a single Conv2d used as the image encoder in GaussianDPT.

final class SkipConvBackbone: Module {
    @ModuleInfo var conv: Conv2d

    init(dimIn: Int, dimOut: Int, kernelSize: Int, stride: Int) {
        let padding = (kernelSize - 1) / 2
        conv = Conv2d(inputChannels: dimIn, outputChannels: dimOut,
                      kernelSize: IntOrPair(kernelSize), stride: IntOrPair(stride),
                      padding: IntOrPair(padding))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x)
    }
}

// MARK: - GNHead
// Python: nn.Sequential(GNResBlock[0], GNResBlock[1], ReLU[2], Conv2d[3], ReLU[4])

// Python: nn.Sequential(GNResBlock[0], GNResBlock[1], ReLU[2], Conv2d[3])
// @ModuleInfo var (no key); assign before super.init().
final class GNHead: Module {
    @ModuleInfo var res1: GNResidualBlock
    @ModuleInfo var res2: GNResidualBlock
    @ModuleInfo var conv: Conv2d

    init(dimDecoder: Int, dimOut: Int, numGroups: Int = 8) {
        let dimHidden = dimDecoder / 2
        res1 = GNResidualBlock(dimIn: dimDecoder, dimHidden: dimHidden, dimOut: dimDecoder,
                               numGroups: numGroups)
        res2 = GNResidualBlock(dimIn: dimDecoder, dimHidden: dimHidden, dimOut: dimDecoder,
                               numGroups: numGroups)
        conv = Conv2d(inputChannels: dimDecoder, outputChannels: dimOut, kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(conv(relu(res2(res1(x)))))
    }
}

// MARK: - GaussianDensePredictionTransformer
// feature_model in the predictor. Fuses monodepth encoder features with image features.

final class GaussianDPT: Module {
    @ModuleInfo var decoder: MultiresConvDecoder
    @ModuleInfo(key: "image_encoder") var imageEncoder: SkipConvBackbone
    @ModuleInfo var fusion: FeatureFusionBlock2d
    @ModuleInfo(key: "texture_head") var textureHead: GNHead
    @ModuleInfo(key: "geometry_head") var geometryHead: GNHead
    // upsample is Identity when stride=2; no weights, no ModuleInfo needed.

    let stride: Int
    let useDepthInput: Bool

    init(
        dimsEncoderFeatures: [Int],
        dimsDecoder: [Int],
        dimIn: Int,
        dimOut: Int,
        stride: Int = 2,
        useDepthInput: Bool = true,
        numGroups: Int = 8
    ) {
        precondition(stride == 1 || stride == 2, "stride must be 1 or 2")
        self.stride = stride
        self.useDepthInput = useDepthInput

        let dimDecoder = dimsDecoder[0]
        let effectiveDimIn = useDepthInput ? dimIn : dimIn - 1
        let kernelSize = (stride == 1) ? 1 : stride

        // @ModuleInfo var (no key) — must be assigned before super.init()
        decoder = MultiresConvDecoder(dimsEncoder: dimsEncoderFeatures, dimsDecoder: dimsDecoder)
        fusion = FeatureFusionBlock2d(dimIn: dimDecoder, dimOut: dimDecoder, upsample: false)
        super.init()

        // @ModuleInfo(key:) — must be assigned after super.init()
        imageEncoder = SkipConvBackbone(dimIn: effectiveDimIn, dimOut: dimDecoder,
                                        kernelSize: kernelSize, stride: stride)
        textureHead = GNHead(dimDecoder: dimDecoder, dimOut: dimOut, numGroups: numGroups)
        geometryHead = GNHead(dimDecoder: dimDecoder, dimOut: dimOut, numGroups: numGroups)
    }

    /// featureInput: NCHW [B, C, H, W]  (image/depth features from initializer)
    /// encodings: list of NHWC feature maps from monodepth encoder
    func callAsFunction(_ featureInput: MLXArray, encodings: [MLXArray]) -> ImageFeatures {
        var features = decoder(encodings)  // NHWC
        // featureInput is NCHW [B,C,H,W]; Conv2d expects NHWC
        let skipFeatures = imageEncoder(featureInput.transposed(0, 2, 3, 1))
        features = fusion(features, skipFeatures)
        return ImageFeatures(
            geometryFeatures: geometryHead(features),
            textureFeatures: textureHead(features)
        )
    }
}

extension GaussianDPT: GaussianFeatureDecoding {}
