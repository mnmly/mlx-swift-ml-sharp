@preconcurrency import MLX

public struct MonodepthOutput {
    public var disparity: MLXArray
    public var encoderFeatures: [MLXArray]
    public var decoderFeatures: MLXArray?
    public var outputFeatures: [MLXArray]
    public var intermediateFeatures: [MLXArray]

    public init(
        disparity: MLXArray,
        encoderFeatures: [MLXArray] = [],
        decoderFeatures: MLXArray? = nil,
        outputFeatures: [MLXArray] = [],
        intermediateFeatures: [MLXArray] = []
    ) {
        self.disparity = disparity
        self.encoderFeatures = encoderFeatures
        self.decoderFeatures = decoderFeatures
        self.outputFeatures = outputFeatures
        self.intermediateFeatures = intermediateFeatures
    }
}

public struct ImageFeatures {
    public var geometryFeatures: MLXArray
    public var textureFeatures: MLXArray

    public init(geometryFeatures: MLXArray, textureFeatures: MLXArray) {
        self.geometryFeatures = geometryFeatures
        self.textureFeatures = textureFeatures
    }
}

public protocol MonodepthPredicting {
    func callAsFunction(_ image: MLXArray) -> MonodepthOutput
}

public protocol GaussianFeatureDecoding {
    func callAsFunction(_ featureInput: MLXArray, encodings: [MLXArray]) -> ImageFeatures
}

public protocol DeltaPredicting {
    func callAsFunction(_ imageFeatures: ImageFeatures) -> MLXArray
}

public protocol DepthAligning {
    func callAsFunction(_ monodepth: MLXArray, _ depth: MLXArray?, _ depthDecoderFeatures: MLXArray?) -> (MLXArray, MLXArray)
}

public struct IdentityDepthAlignment: DepthAligning {
    public init() {}

    public func callAsFunction(_ monodepth: MLXArray, _ depth: MLXArray?, _ depthDecoderFeatures: MLXArray?) -> (MLXArray, MLXArray) {
        (monodepth, MLXArray.ones(monodepth.shape))
    }
}

public struct RGBGaussianPredictor {
    public let initializer: any GaussianInitializing
    public let monodepthModel: any MonodepthPredicting
    public let featureModel: any GaussianFeatureDecoding
    public let predictionHead: any DeltaPredicting
    public let gaussianComposer: GaussianComposer
    public let depthAlignment: any DepthAligning

    public init(
        initializer: any GaussianInitializing,
        monodepthModel: any MonodepthPredicting,
        featureModel: any GaussianFeatureDecoding,
        predictionHead: any DeltaPredicting,
        gaussianComposer: GaussianComposer,
        depthAlignment: any DepthAligning = IdentityDepthAlignment()
    ) {
        self.initializer = initializer
        self.monodepthModel = monodepthModel
        self.featureModel = featureModel
        self.predictionHead = predictionHead
        self.gaussianComposer = gaussianComposer
        self.depthAlignment = depthAlignment
    }

    public func callAsFunction(
        _ image: MLXArray,
        disparityFactor: MLXArray,
        depth: MLXArray? = nil
    ) -> Gaussians3D {
        let monodepthOutput = monodepthModel(image)
        let disparityScale = disparityFactor
            .expandedDimensions(axis: 1)
            .expandedDimensions(axis: 2)
            .expandedDimensions(axis: 3)
        let monodepth = disparityScale / clip(monodepthOutput.disparity, min: 1e-4, max: 1e4)
        let aligned = depthAlignment(monodepth, depth, monodepthOutput.decoderFeatures)
        let initOutput = initializer(image, aligned.0)
        let imageFeatures = featureModel(initOutput.featureInput, encodings: monodepthOutput.outputFeatures)
        let deltaValues = predictionHead(imageFeatures)
        return gaussianComposer(
            delta: deltaValues,
            baseValues: initOutput.gaussianBaseValues,
            globalScale: initOutput.globalScale
        )
    }
}
