@preconcurrency import MLX

public struct Gaussians3D {
    public var meanVectors: MLXArray
    public var singularValues: MLXArray
    public var quaternions: MLXArray
    public var colors: MLXArray
    public var opacities: MLXArray

    public init(
        meanVectors: MLXArray,
        singularValues: MLXArray,
        quaternions: MLXArray,
        colors: MLXArray,
        opacities: MLXArray
    ) {
        self.meanVectors = meanVectors
        self.singularValues = singularValues
        self.quaternions = quaternions
        self.colors = colors
        self.opacities = opacities
    }
}

public struct SceneMetaData {
    public var focalLengthPixels: Float
    public var resolutionPixels: (Int, Int)
    public var colorSpace: ColorSpace

    public init(focalLengthPixels: Float, resolutionPixels: (Int, Int), colorSpace: ColorSpace) {
        self.focalLengthPixels = focalLengthPixels
        self.resolutionPixels = resolutionPixels
        self.colorSpace = colorSpace
    }
}

public struct GaussianBaseValues {
    public var meanXNDC: MLXArray
    public var meanYNDC: MLXArray
    public var meanInverseZNDC: MLXArray
    public var scales: MLXArray
    public var quaternions: MLXArray
    public var colors: MLXArray
    public var opacities: MLXArray

    public init(
        meanXNDC: MLXArray,
        meanYNDC: MLXArray,
        meanInverseZNDC: MLXArray,
        scales: MLXArray,
        quaternions: MLXArray,
        colors: MLXArray,
        opacities: MLXArray
    ) {
        self.meanXNDC = meanXNDC
        self.meanYNDC = meanYNDC
        self.meanInverseZNDC = meanInverseZNDC
        self.scales = scales
        self.quaternions = quaternions
        self.colors = colors
        self.opacities = opacities
    }
}

public struct InitializerOutput {
    public var gaussianBaseValues: GaussianBaseValues
    public var featureInput: MLXArray
    public var globalScale: MLXArray?

    public init(gaussianBaseValues: GaussianBaseValues, featureInput: MLXArray, globalScale: MLXArray?) {
        self.gaussianBaseValues = gaussianBaseValues
        self.featureInput = featureInput
        self.globalScale = globalScale
    }
}
