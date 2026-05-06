@preconcurrency import MLX
@preconcurrency import MLXNN

// MARK: - DirectPredictionHead
// Python keys: geometry_prediction_head, texture_prediction_head
// @ModuleInfo var (no key) here; ModelLoader remaps the Python keys to Swift names.

final class DirectPredictionHead: Module {
    @ModuleInfo var geomHead: Conv2d
    @ModuleInfo var textHead: Conv2d
    let numLayers: Int

    init(featureDim: Int, numLayers nl: Int) {
        numLayers = nl
        geomHead = Conv2d(inputChannels: featureDim, outputChannels: 3 * nl, kernelSize: 1)
        textHead = Conv2d(inputChannels: featureDim, outputChannels: 11 * nl, kernelSize: 1)
        super.init()
    }

    /// imageFeatures: geometry/texture NHWC
    /// Returns delta: [B, 14, numLayers, H, W]
    func callAsFunction(_ imageFeatures: ImageFeatures) -> MLXArray {
        let geomRaw = geomHead(imageFeatures.geometryFeatures)  // [B, H, W, 3*numLayers]
        let textRaw = textHead(imageFeatures.textureFeatures)   // [B, H, W, 11*numLayers]
        let B = geomRaw.shape[0], H = geomRaw.shape[1], W = geomRaw.shape[2]
        let geom = geomRaw.reshaped([B, H, W, 3, numLayers])
        let text = textRaw.reshaped([B, H, W, 11, numLayers])
        let combined = concatenated([geom, text], axis: 3)  // [B, H, W, 14, numLayers]
        return combined.transposed(0, 3, 4, 1, 2)            // [B, 14, numLayers, H, W]
    }
}

extension DirectPredictionHead: DeltaPredicting {}
