import Foundation

public typealias DimsDecoder = (Int, Int, Int, Int, Int)

public enum ColorSpace: String {
    case sRGB
    case linearRGB
}

public enum ActivationType: String {
    case linear
    case exp
    case sigmoid
    case softplus
}

public enum ColorInitOption: String {
    case none
    case firstLayer = "first_layer"
    case allLayers = "all_layers"
}

public enum DepthInitOption: String {
    case surfaceMin = "surface_min"
    case surfaceMax = "surface_max"
    case baseDepth = "base_depth"
    case linearDisparity = "linear_disparity"
}

public struct AlignmentParams {
    public var kernelSize: Int = 16
    public var stride: Int = 1
    public var frozen: Bool = false
    public var steps: Int = 4
    public var activationType: ActivationType = .exp
    public var depthDecoderFeatures: Bool = false
    public var baseWidth: Int = 16

    public init() {}
}

public struct DeltaFactor {
    public var xy: Float = 0.001
    public var z: Float = 0.001
    public var color: Float = 0.1
    public var opacity: Float = 1.0
    public var scale: Float = 1.0
    public var quaternion: Float = 1.0

    public init() {}
}

public struct InitializerParams {
    public var scaleFactor: Float = 1.0
    public var disparityFactor: Float = 1.0
    public var stride: Int = 2
    public var numLayers: Int = 2
    public var firstLayerDepthOption: DepthInitOption = .surfaceMin
    public var restLayerDepthOption: DepthInitOption = .surfaceMin
    public var colorOption: ColorInitOption = .allLayers
    public var baseDepth: Float = 10.0
    public var featureInputStopGrad: Bool = false
    public var normalizeDepth: Bool = true

    public init() {}
}

public struct MonodepthParams {
    public var patchEncoderPreset: String = "dinov2l16_384"
    public var imageEncoderPreset: String = "dinov2l16_384"
    public var checkpointURI: String?
    public var unfreezePatchEncoder: Bool = false
    public var unfreezeImageEncoder: Bool = false
    public var unfreezeDecoder: Bool = false
    public var unfreezeHead: Bool = false
    public var unfreezeNormLayers: Bool = false
    public var gradCheckpointing: Bool = false
    public var usePatchOverlap: Bool = true
    public var dimsDecoder: DimsDecoder = (256, 256, 256, 256, 256)

    public init() {}
}

public struct MonodepthAdaptorParams {
    public var encoderFeatures: Bool = true
    public var decoderFeatures: Bool = false

    public init() {}
}

public struct GaussianDecoderParams {
    public var dimIn: Int = 5
    public var dimOut: Int = 32
    public var stride: Int = 2
    public var patchEncoderPreset: String = "dinov2l16_384"
    public var imageEncoderPreset: String = "dinov2l16_384"
    public var dimsDecoder: DimsDecoder = (128, 128, 128, 128, 128)
    public var useDepthInput: Bool = true
    public var gradCheckpointing: Bool = false

    public init() {}
}

public struct PredictorParams {
    public var initializer: InitializerParams = .init()
    public var monodepth: MonodepthParams = .init()
    public var monodepthAdaptor: MonodepthAdaptorParams = .init()
    public var gaussianDecoder: GaussianDecoderParams = .init()
    public var depthAlignment: AlignmentParams = .init()
    public var deltaFactor: DeltaFactor = .init()
    public var maxScale: Float = 10.0
    public var minScale: Float = 0.0
    public var colorActivationType: ActivationType = .sigmoid
    public var opacityActivationType: ActivationType = .sigmoid
    public var colorSpace: ColorSpace = .linearRGB
    public var lowPassFilterEps: Float = 1e-2
    public var numMonodepthLayers: Int = 2
    public var sortingMonodepth: Bool = false
    public var baseScaleOnPredictedMean: Bool = true

    public init() {}
}
