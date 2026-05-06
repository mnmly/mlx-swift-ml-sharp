import Foundation
import MLX
import Testing
@testable import MLXSharp

// MARK: - Fixture decoding

private struct TensorData: Decodable {
    let shape: [Int]
    let values: [Float]

    func toMLXArray() -> MLXArray {
        MLXArray(values, shape)
    }
}

private struct BaseValuesData: Decodable {
    let mean_x_ndc: TensorData
    let mean_y_ndc: TensorData
    let mean_inverse_z_ndc: TensorData
    let scales: TensorData
    let quaternions: TensorData
    let colors: TensorData
    let opacities: TensorData
}

private struct GaussiansData: Decodable {
    let mean_vectors: TensorData
    let singular_values: TensorData
    let quaternions: TensorData
    let colors: TensorData
    let opacities: TensorData
}

private struct ExpectedData: Decodable {
    let feature_input: TensorData
    let global_scale: TensorData?
    let base_values: BaseValuesData
    let gaussians: GaussiansData
}

private struct DeltaFactorData: Decodable {
    let xy: Float
    let z: Float
    let color: Float
    let opacity: Float
    let scale: Float
    let quaternion: Float
}

private struct InitializerParamsData: Decodable {
    let scale_factor: Float
    let disparity_factor: Float
    let stride: Int
    let num_layers: Int
    let first_layer_depth_option: String
    let rest_layer_depth_option: String
    let color_option: String
    let base_depth: Float
    let feature_input_stop_grad: Bool
    let normalize_depth: Bool
}

private struct ComposerParamsData: Decodable {
    let delta_factor: DeltaFactorData
    let min_scale: Float
    let max_scale: Float
    let color_activation_type: String
    let opacity_activation_type: String
    let color_space: String
    let base_scale_on_predicted_mean: Bool
    let scale_factor: Int
}

private struct CaseData: Decodable {
    let name: String
    let initializer_params: InitializerParamsData
    let composer_params: ComposerParamsData
    let image: TensorData
    let depth: TensorData
    let delta: TensorData
    let expected: ExpectedData
}

private struct FixtureData: Decodable {
    let cases: [CaseData]
}

// MARK: - Helpers

private func depthOption(_ s: String) -> DepthInitOption {
    switch s {
    case "surface_min": return .surfaceMin
    case "surface_max": return .surfaceMax
    case "base_depth": return .baseDepth
    case "linear_disparity": return .linearDisparity
    default: fatalError("Unknown depth option: \(s)")
    }
}

private func colorOption(_ s: String) -> ColorInitOption {
    switch s {
    case "none": return .none
    case "first_layer": return .firstLayer
    case "all_layers": return .allLayers
    default: fatalError("Unknown color option: \(s)")
    }
}

private func activationType(_ s: String) -> ActivationType {
    switch s {
    case "sigmoid": return .sigmoid
    case "exp": return .exp
    case "softplus": return .softplus
    case "linear": return .linear
    default: fatalError("Unknown activation: \(s)")
    }
}

private func colorSpaceType(_ s: String) -> ColorSpace {
    switch s {
    case "sRGB": return .sRGB
    case "linearRGB": return .linearRGB
    default: fatalError("Unknown color space: \(s)")
    }
}

private func assertClose(
    _ actual: MLXArray,
    _ expected: MLXArray,
    atol: Float = 1e-4,
    label: String
) {
    eval(actual, expected)
    let diff = abs(actual - expected).max().item(Float.self)
    #expect(diff <= atol, "\(label): max abs diff \(diff) > \(atol)")
}

// MARK: - Parity tests

@Suite("Core Parity")
struct CoreParityTests {
    fileprivate static let fixture: FixtureData = {
        let url = Bundle.module.url(forResource: "core-parity", withExtension: "json",
                                    subdirectory: "Fixtures")!
        let data = try! Data(contentsOf: url)
        return try! JSONDecoder().decode(FixtureData.self, from: data)
    }()

    @Test("Initializer parity", arguments: CoreParityTests.fixture.cases.map(\.name))
    func initializerParity(caseName: String) throws {
        let c = CoreParityTests.fixture.cases.first { $0.name == caseName }!

        var params = InitializerParams()
        params.scaleFactor = c.initializer_params.scale_factor
        params.disparityFactor = c.initializer_params.disparity_factor
        params.stride = c.initializer_params.stride
        params.numLayers = c.initializer_params.num_layers
        params.firstLayerDepthOption = depthOption(c.initializer_params.first_layer_depth_option)
        params.restLayerDepthOption = depthOption(c.initializer_params.rest_layer_depth_option)
        params.colorOption = colorOption(c.initializer_params.color_option)
        params.baseDepth = c.initializer_params.base_depth
        params.featureInputStopGrad = c.initializer_params.feature_input_stop_grad
        params.normalizeDepth = c.initializer_params.normalize_depth

        let initializer = createInitializer(params: params)
        let image = c.image.toMLXArray()
        let depth = c.depth.toMLXArray()

        let output = initializer(image, depth)
        eval(output.featureInput,
             output.gaussianBaseValues.meanXNDC,
             output.gaussianBaseValues.meanYNDC,
             output.gaussianBaseValues.meanInverseZNDC,
             output.gaussianBaseValues.scales,
             output.gaussianBaseValues.quaternions,
             output.gaussianBaseValues.colors,
             output.gaussianBaseValues.opacities)

        assertClose(output.featureInput, c.expected.feature_input.toMLXArray(), atol: 1e-4,
                    label: "feature_input")
        assertClose(output.gaussianBaseValues.meanXNDC,
                    c.expected.base_values.mean_x_ndc.toMLXArray(), atol: 1e-4, label: "mean_x_ndc")
        assertClose(output.gaussianBaseValues.meanYNDC,
                    c.expected.base_values.mean_y_ndc.toMLXArray(), atol: 1e-4, label: "mean_y_ndc")
        assertClose(output.gaussianBaseValues.meanInverseZNDC,
                    c.expected.base_values.mean_inverse_z_ndc.toMLXArray(), atol: 1e-4,
                    label: "mean_inverse_z_ndc")
        assertClose(output.gaussianBaseValues.scales,
                    c.expected.base_values.scales.toMLXArray(), atol: 1e-4, label: "scales")
        assertClose(output.gaussianBaseValues.quaternions,
                    c.expected.base_values.quaternions.toMLXArray(), atol: 1e-4,
                    label: "quaternions")
        assertClose(output.gaussianBaseValues.colors,
                    c.expected.base_values.colors.toMLXArray(), atol: 1e-4, label: "colors")
        assertClose(output.gaussianBaseValues.opacities,
                    c.expected.base_values.opacities.toMLXArray(), atol: 1e-4, label: "opacities")

        if let expectedGlobalScale = c.expected.global_scale {
            #expect(output.globalScale != nil, "Expected globalScale to be non-nil")
            if let gs = output.globalScale {
                assertClose(gs, expectedGlobalScale.toMLXArray(), atol: 1e-4, label: "global_scale")
            }
        } else {
            #expect(output.globalScale == nil, "Expected globalScale to be nil")
        }
    }

    @Test("Composer parity", arguments: CoreParityTests.fixture.cases.map(\.name))
    func composerParity(caseName: String) throws {
        let c = CoreParityTests.fixture.cases.first { $0.name == caseName }!

        // Run initializer to get base values
        var iParams = InitializerParams()
        iParams.scaleFactor = c.initializer_params.scale_factor
        iParams.disparityFactor = c.initializer_params.disparity_factor
        iParams.stride = c.initializer_params.stride
        iParams.numLayers = c.initializer_params.num_layers
        iParams.firstLayerDepthOption = depthOption(c.initializer_params.first_layer_depth_option)
        iParams.restLayerDepthOption = depthOption(c.initializer_params.rest_layer_depth_option)
        iParams.colorOption = colorOption(c.initializer_params.color_option)
        iParams.baseDepth = c.initializer_params.base_depth
        iParams.featureInputStopGrad = c.initializer_params.feature_input_stop_grad
        iParams.normalizeDepth = c.initializer_params.normalize_depth

        let initOutput = createInitializer(params: iParams)(c.image.toMLXArray(), c.depth.toMLXArray())

        var df = DeltaFactor()
        df.xy = c.composer_params.delta_factor.xy
        df.z = c.composer_params.delta_factor.z
        df.color = c.composer_params.delta_factor.color
        df.opacity = c.composer_params.delta_factor.opacity
        df.scale = c.composer_params.delta_factor.scale
        df.quaternion = c.composer_params.delta_factor.quaternion

        let composer = GaussianComposer(
            deltaFactor: df,
            minScale: c.composer_params.min_scale,
            maxScale: c.composer_params.max_scale,
            colorActivationType: activationType(c.composer_params.color_activation_type),
            opacityActivationType: activationType(c.composer_params.opacity_activation_type),
            colorSpace: colorSpaceType(c.composer_params.color_space),
            baseScaleOnPredictedMean: c.composer_params.base_scale_on_predicted_mean,
            scaleFactor: c.composer_params.scale_factor
        )

        let delta = c.delta.toMLXArray()
        let gaussians = composer(
            delta: delta,
            baseValues: initOutput.gaussianBaseValues,
            globalScale: initOutput.globalScale,
            flattenOutput: true
        )
        eval(gaussians.meanVectors, gaussians.singularValues,
             gaussians.quaternions, gaussians.colors, gaussians.opacities)

        assertClose(gaussians.meanVectors, c.expected.gaussians.mean_vectors.toMLXArray(),
                    atol: 1e-4, label: "mean_vectors")
        assertClose(gaussians.singularValues, c.expected.gaussians.singular_values.toMLXArray(),
                    atol: 1e-4, label: "singular_values")
        assertClose(gaussians.quaternions, c.expected.gaussians.quaternions.toMLXArray(),
                    atol: 1e-4, label: "quaternions")
        assertClose(gaussians.colors, c.expected.gaussians.colors.toMLXArray(),
                    atol: 1e-4, label: "colors")
        assertClose(gaussians.opacities, c.expected.gaussians.opacities.toMLXArray(),
                    atol: 1e-4, label: "opacities")
    }
}
