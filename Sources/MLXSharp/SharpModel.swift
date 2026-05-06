@preconcurrency import MLX
@preconcurrency import MLXNN
import Foundation

// MARK: - SharpModel
// Top-level Module matching the Python RGBGaussianPredictor parameter tree.
// @ModuleInfo var (no key) is used here because the init receives pre-built modules;
// top-level key remapping (snake_case ↔ camelCase) is handled in ModelLoader.

final class SharpModel: Module {
    @ModuleInfo var monodepthModel: MonodepthAdaptor
    @ModuleInfo var featureModel: GaussianDPT
    @ModuleInfo var predictionHead: DirectPredictionHead
    @ModuleInfo var depthAlignment: DepthAlignmentModule

    init(
        monodepthModel: MonodepthAdaptor,
        featureModel: GaussianDPT,
        predictionHead: DirectPredictionHead,
        depthAlignment: DepthAlignmentModule
    ) {
        self.monodepthModel = monodepthModel
        self.featureModel = featureModel
        self.predictionHead = predictionHead
        self.depthAlignment = depthAlignment
        super.init()
    }
}

// MARK: - Factory for default dinov2l16_384 + sharp_2572gikvuh config

/// Encoder output dims for dinov2l16_384 patch encoder.
/// SPN outputs: [latent0(256), latent1(256), level0(512), level1(1024), level2_fused(1024)]
private let spnDimsEncoder: [Int] = [256, 256, 512, 1024, 1024]

func makeDefaultSharpModel() -> SharpModel {
    // --- Monodepth encoder (SlidingPyramidNetwork) ---
    let monodepthEncoder = SlidingPyramidNetwork(
        dimsEncoder: spnDimsEncoder,
        embedDim: 1024,
        lowresEmbedDim: 1024,
        intermediateIds: [5, 11],
        imgSize: 384,
        patchSize: 16,
        depth: 24,
        numHeads: 16
    )

    // --- Monodepth decoder (MultiresConvDecoder) ---
    // dims_decoder[0] == dims_encoder[0] = 256 → conv0IsIdentity = true
    let monodepthDecoder = MultiresConvDecoder(
        dimsEncoder: spnDimsEncoder,
        dimsDecoder: [256, 256, 256, 256, 256]
    )

    // --- Monodepth head ---
    let monodepthHead = MonodepthHead(dimDecoder: 256, lastDims: (32, 2))

    // --- Monodepth DPT and adaptor ---
    let monodepthDPT = MonodepthDPT(
        encoder: monodepthEncoder,
        decoder: monodepthDecoder,
        head: monodepthHead
    )
    let monodepthAdaptor = MonodepthAdaptor(
        monodepthPredictor: monodepthDPT,
        returnEncoderFeatures: true,
        numMonodepthLayers: 2,
        sortingMonodepth: false
    )

    // --- Gaussian feature decoder ---
    // uses monodepth ENCODER features (5 maps of spnDimsEncoder)
    let gaussianDPT = GaussianDPT(
        dimsEncoderFeatures: spnDimsEncoder,
        dimsDecoder: [128, 128, 128, 128, 128],
        dimIn: 5,   // 3 (RGB) + 2 (2-layer disparity)
        dimOut: 32,
        stride: 2,
        useDepthInput: true
    )

    // --- Prediction head ---
    let predHead = DirectPredictionHead(featureDim: 32, numLayers: 2)

    // --- Depth alignment ---
    let alignment = LearnedAlignment(
        dimIn: 2,
        steps: 4,
        stride: 1,
        baseWidth: 16,
        numGroups: 4
    )
    let depthAlignModule = DepthAlignmentModule(scaleMapEstimator: alignment)

    return SharpModel(
        monodepthModel: monodepthAdaptor,
        featureModel: gaussianDPT,
        predictionHead: predHead,
        depthAlignment: depthAlignModule
    )
}
