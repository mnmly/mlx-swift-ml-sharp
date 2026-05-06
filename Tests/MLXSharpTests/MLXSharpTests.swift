import MLX
import Testing
@testable import MLXSharp

@Test func initializerProducesExpectedShapes() {
    let params = InitializerParams()
    let initializer = createInitializer(params: params)
    let image = MLXArray.ones([1, 3, 8, 8], dtype: .float32)
    let depth = MLXArray.ones([1, 2, 8, 8], dtype: .float32) * 2.0

    let output = initializer(image, depth)

    #expect(output.featureInput.shape == [1, 5, 8, 8])
    #expect(output.gaussianBaseValues.meanXNDC.shape == [1, 1, 2, 4, 4])
    #expect(output.gaussianBaseValues.colors.shape == [1, 3, 2, 4, 4])
    #expect(output.gaussianBaseValues.opacities.shape == [1, 2, 4, 4])
}

@Test func composerFlattensToPointCloudShape() {
    let params = PredictorParams()
    let initializer = createInitializer(params: params.initializer)
    let composer = GaussianComposer(
        deltaFactor: params.deltaFactor,
        minScale: params.minScale,
        maxScale: params.maxScale,
        colorActivationType: params.colorActivationType,
        opacityActivationType: params.opacityActivationType,
        colorSpace: params.colorSpace,
        baseScaleOnPredictedMean: params.baseScaleOnPredictedMean
    )

    let image = MLXArray.ones([1, 3, 8, 8], dtype: .float32) * 0.5
    let depth = MLXArray.ones([1, 2, 8, 8], dtype: .float32) * 3.0
    let initOutput = initializer(image, depth)
    let delta = MLXArray.zeros([1, 14, 2, 4, 4], dtype: .float32)
    let gaussians = composer(delta: delta, baseValues: initOutput.gaussianBaseValues, globalScale: initOutput.globalScale)

    #expect(gaussians.meanVectors.shape == [1, 32, 3])
    #expect(gaussians.singularValues.shape == [1, 32, 3])
    #expect(gaussians.quaternions.shape == [1, 32, 4])
    #expect(gaussians.colors.shape == [1, 32, 3])
    #expect(gaussians.opacities.shape == [1, 32])
}
