@preconcurrency import MLX
@preconcurrency import MLXNN

// MARK: - Monodepth head
// Python: nn.Sequential with indices 0=Conv2d, 1=ConvTransposed2d, 2=Conv2d, 3=ReLU, 4=Conv2d, 5=ReLU

// Python: nn.Sequential(Conv2d[0], ConvT[1], Conv2d[2], ReLU[3], Conv2d[4])
// @ModuleInfo var (no key); assign before super.init().
final class MonodepthHead: Module {
    @ModuleInfo var conv0: Conv2d
    @ModuleInfo var deconv1: ConvTransposed2d
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo var conv4: Conv2d

    init(dimDecoder: Int, lastDims: (Int, Int)) {
        conv0 = Conv2d(inputChannels: dimDecoder, outputChannels: dimDecoder / 2,
                       kernelSize: 3, padding: 1)
        deconv1 = ConvTransposed2d(inputChannels: dimDecoder / 2,
                                   outputChannels: dimDecoder / 2,
                                   kernelSize: 2, stride: 2)
        conv2 = Conv2d(inputChannels: dimDecoder / 2, outputChannels: lastDims.0,
                       kernelSize: 3, padding: 1)
        conv4 = Conv2d(inputChannels: lastDims.0, outputChannels: lastDims.1, kernelSize: 1)
        super.init()
    }

    /// x: NHWC [B, H, W, C] → NHWC [B, H', W', lastDims.1]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv0(x)
        out = deconv1(out)
        out = conv2(out)
        out = relu(out)
        out = conv4(out)
        out = relu(out)
        return out
    }
}

// MARK: - MonodepthDPT

final class MonodepthDPT: Module {
    @ModuleInfo var encoder: SlidingPyramidNetwork
    @ModuleInfo var decoder: MultiresConvDecoder
    @ModuleInfo var head: MonodepthHead

    // Normaliser has no weights; applied before encoder, not stored in safetensors.
    private let normalizer = AffineNormalizer(inputRange: (0, 1), outputRange: (-1, 1))

    init(encoder: SlidingPyramidNetwork, decoder: MultiresConvDecoder, head: MonodepthHead) {
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        super.init()
    }

    /// image: [B, 3, H, W] NCHW in [0,1]
    /// Returns (disparity NHWC, encoderFeatures [NHWC], decoderFeatures NHWC)
    func forward(_ image: MLXArray) -> (MLXArray, [MLXArray], MLXArray) {
        // Normalise in NCHW space
        let normalised = normalizer(image)
        let encoderFeatures = encoder(normalised)  // 5 NHWC maps
        let decoderFeatures = decoder(encoderFeatures)  // NHWC
        let disparity = head(decoderFeatures)  // NHWC
        return (disparity, encoderFeatures, decoderFeatures)
    }
}

// MARK: - MonodepthAdaptor (MonodepthWithEncodingAdaptor)

final class MonodepthAdaptor: Module {
    @ModuleInfo(key: "monodepth_predictor") var monodepthPredictor: MonodepthDPT
    let returnEncoderFeatures: Bool
    let numMonodepthLayers: Int
    let sortingMonodepth: Bool

    init(
        monodepthPredictor: MonodepthDPT,
        returnEncoderFeatures: Bool = true,
        numMonodepthLayers: Int = 2,
        sortingMonodepth: Bool = false
    ) {
        self.returnEncoderFeatures = returnEncoderFeatures
        self.numMonodepthLayers = numMonodepthLayers
        self.sortingMonodepth = sortingMonodepth
        super.init()
        self.monodepthPredictor = monodepthPredictor
    }

    /// Returns MonodepthOutput with disparity in NCHW format
    func callAsFunction(_ image: MLXArray) -> MonodepthOutput {
        let (disparity, encFeatures, decFeatures) = monodepthPredictor.forward(image)

        var sortedDisparity = disparity
        if numMonodepthLayers == 2 && sortingMonodepth {
            // disparity is NHWC [B, H, W, 2]; sort along channel axis
            let first = disparity.max(axis: -1, keepDims: true)
            let second = disparity.min(axis: -1, keepDims: true)
            sortedDisparity = concatenated([first, second], axis: -1)
        }
        // Convert NHWC [B,H,W,C] → NCHW [B,C,H,W] for downstream initializer
        sortedDisparity = sortedDisparity.transposed(0, 3, 1, 2)

        var outputFeatures: [MLXArray] = []
        if returnEncoderFeatures {
            outputFeatures = encFeatures
        }

        return MonodepthOutput(
            disparity: sortedDisparity,
            encoderFeatures: encFeatures,
            decoderFeatures: decFeatures,
            outputFeatures: outputFeatures
        )
    }
}

extension MonodepthAdaptor: MonodepthPredicting {}
