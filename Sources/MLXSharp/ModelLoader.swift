@preconcurrency import MLX
@preconcurrency import MLXNN
import Foundation

// MARK: - Weight loading

/// Load a SharpModel from a safetensors file and construct an RGBGaussianPredictor.
public func loadSharpModel(from url: URL) throws -> RGBGaussianPredictor {
    var weights = try loadArrays(url: url)

    // Python → Swift key remapping
    weights = remapKeys(weights)

    let model = makeDefaultSharpModel()
    let params = ModuleParameters.unflattened(weights)
    try model.update(parameters: params, verify: .none)
    eval(model)

    return makePredictor(from: model)
}

/// Create an `RGBGaussianPredictor` with random (untrained) weights for benchmarking.
public func makeRandomPredictor(params: PredictorParams = .init()) -> RGBGaussianPredictor {
    makePredictor(from: makeDefaultSharpModel(), params: params)
}

/// Build an `RGBGaussianPredictor` from a loaded `SharpModel` plus stateless helpers.
func makePredictor(
    from sharpModel: SharpModel,
    params: PredictorParams = .init()
) -> RGBGaussianPredictor {
    let initParams = params.initializer
    let compParams = params

    let initializer = createInitializer(params: initParams)

    let df = compParams.deltaFactor
    var deltaFactor = DeltaFactor()
    deltaFactor.xy = df.xy
    deltaFactor.z = df.z
    deltaFactor.color = df.color
    deltaFactor.opacity = df.opacity
    deltaFactor.scale = df.scale
    deltaFactor.quaternion = df.quaternion

    let composer = GaussianComposer(
        deltaFactor: deltaFactor,
        minScale: compParams.minScale,
        maxScale: compParams.maxScale,
        colorActivationType: compParams.colorActivationType,
        opacityActivationType: compParams.opacityActivationType,
        colorSpace: compParams.colorSpace,
        baseScaleOnPredictedMean: compParams.baseScaleOnPredictedMean,
        scaleFactor: initParams.stride
    )

    return RGBGaussianPredictor(
        initializer: initializer,
        monodepthModel: sharpModel.monodepthModel,
        featureModel: sharpModel.featureModel,
        predictionHead: sharpModel.predictionHead,
        gaussianComposer: composer,
        depthAlignment: sharpModel.depthAlignment
    )
}

// MARK: - Key remapping

/// Apply all Python → Swift key transformations.
private func remapKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var result = [String: MLXArray](minimumCapacity: weights.count)
    for (key, value) in weights {
        result[remapKey(key)] = value
    }
    return result
}

/// Transform a single weight key from Python naming to Swift parameter-tree naming.
private func remapKey(_ key: String) -> String {
    var k = key

    // ── Top-level module names (snake_case → camelCase) ──────────────────────
    k = replacePrefix(k, from: "monodepth_model.", to: "monodepthModel.")
    k = replacePrefix(k, from: "feature_model.", to: "featureModel.")
    k = replacePrefix(k, from: "prediction_head.", to: "predictionHead.")
    k = replacePrefix(k, from: "depth_alignment.", to: "depthAlignment.")

    // ── prediction_head subkeys ───────────────────────────────────────────────
    k = k.replacingOccurrences(of: ".geometry_prediction_head.", with: ".geomHead.")
    k = k.replacingOccurrences(of: ".texture_prediction_head.", with: ".textHead.")

    // ── monodepth decoder convs: Python starts at convs.1 (identity at 0) ────
    k = remapMonodepthDecoderConvs(k)

    // ── nn.Sequential integer indices → Swift property names ─────────────────
    k = remapSequentialIndices(k)

    return k
}

/// Walk the dot-separated key components and replace Python nn.Sequential integer
/// indices with the corresponding Swift property names, using parent context to
/// disambiguate modules that share the same index range.
private func remapSequentialIndices(_ key: String) -> String {
    let parts = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
    var out: [String] = []
    out.reserveCapacity(parts.count)

    for part in parts {
        guard let n = Int(part) else {
            out.append(part)
            continue
        }

        let parent = out.last ?? ""
        let grandparent = out.count >= 2 ? out[out.count - 2] : ""

        let mapped: String
        // MonodepthHead: nn.Sequential(Conv2d[0], ConvT[1], Conv2d[2], ReLU[3], Conv2d[4])
        if parent == "head" && grandparent == "monodepth_predictor" {
            mapped = [0: "conv0", 1: "deconv1", 2: "conv2", 4: "conv4"][n] ?? part

        // GNHead: nn.Sequential(GNResBlock[0], GNResBlock[1], ReLU[2], Conv2d[3])
        } else if parent == "geometry_head" || parent == "texture_head" {
            mapped = [0: "res1", 1: "res2", 3: "conv"][n] ?? part

        // UNetConvIn: nn.Sequential(Conv2d[0], GroupNorm[1], ReLU[2])
        } else if parent == "conv_in" {
            mapped = [0: "conv", 1: "norm"][n] ?? part

        // UNetConvOut: nn.Sequential(GN[0], ReLU[1], Conv2d[2], GN[3], ReLU[4])
        // Only inside .decoder.conv_out (not scale_map_estimator.conv_out which is plain Conv2d)
        } else if parent == "conv_out" && grandparent == "decoder" {
            mapped = [0: "norm1", 2: "conv", 3: "norm2"][n] ?? part

        // FFBResidualSequence: nn.Sequential(ReLU[0], Conv2d[1], ReLU[2], Conv2d[3])
        // Only under resnet1.residual or resnet2.residual
        } else if parent == "residual" && (grandparent == "resnet1" || grandparent == "resnet2") {
            mapped = [1: "conv1", 3: "conv2"][n] ?? part

        // GNResidualSequence: nn.Sequential(GN[0], ReLU[1], Conv2d[2], GN[3], ReLU[4], Conv2d[5])
        } else if parent == "residual" {
            mapped = [0: "norm1", 2: "conv1", 3: "norm2", 5: "conv2"][n] ?? part

        // UNetEncoderStep / UNetDecoderStep (parent is an array index, grandparent is convs_down/up)
        } else if Int(parent) != nil && (grandparent == "convs_down" || grandparent == "convs_up") {
            mapped = [1: "res1", 2: "res2"][n] ?? part

        // ProjectUpsampleBlock: nn.Sequential(Conv2d[0], ConvT[1], ConvT[2]?, ConvT[3]?)
        } else if parent.hasPrefix("upsample") {
            mapped = [0: "proj", 1: "up1", 2: "up2", 3: "up3"][n] ?? part

        } else {
            mapped = part  // pure array index — keep as-is
        }

        out.append(mapped)
    }

    return out.joined(separator: ".")
}

private func replacePrefix(_ s: String, from prefix: String, to replacement: String) -> String {
    s.hasPrefix(prefix) ? replacement + s.dropFirst(prefix.count) : s
}

/// Remap `*.monodepthModel.monodepth_predictor.decoder.convs.N` → `convs.(N-1)`.
/// Python stores convs starting at index 1 (index 0 is identity); shift to 0-based for Swift array.
private func remapMonodepthDecoderConvs(_ key: String) -> String {
    let marker = "monodepthModel.monodepth_predictor.decoder.convs."
    guard key.contains(marker) else { return key }

    guard let range = key.range(of: marker) else { return key }
    let afterMarker = String(key[range.upperBound...])
    let dotIdx = afterMarker.firstIndex(of: ".") ?? afterMarker.endIndex
    guard let n = Int(afterMarker[afterMarker.startIndex..<dotIdx]), n >= 1 else { return key }
    let suffix = String(afterMarker[dotIdx...])
    return String(key[key.startIndex..<range.lowerBound]) + marker + "\(n - 1)" + suffix
}
