@preconcurrency import MLX
@preconcurrency import MLXNN

public struct GaussianComposer {
    public let deltaFactor: DeltaFactor
    public let minScale: Float
    public let maxScale: Float
    public let colorActivationType: ActivationType
    public let opacityActivationType: ActivationType
    public let colorSpace: ColorSpace
    public let scaleFactor: Int
    public let baseScaleOnPredictedMean: Bool

    public init(
        deltaFactor: DeltaFactor,
        minScale: Float,
        maxScale: Float,
        colorActivationType: ActivationType,
        opacityActivationType: ActivationType,
        colorSpace: ColorSpace,
        baseScaleOnPredictedMean: Bool,
        scaleFactor: Int = 1
    ) {
        self.deltaFactor = deltaFactor
        self.minScale = minScale
        self.maxScale = maxScale
        self.colorActivationType = colorActivationType
        self.opacityActivationType = opacityActivationType
        self.colorSpace = colorSpace
        self.scaleFactor = scaleFactor
        self.baseScaleOnPredictedMean = baseScaleOnPredictedMean
    }

    public func callAsFunction(
        delta: MLXArray,
        baseValues: GaussianBaseValues,
        globalScale: MLXArray? = nil,
        flattenOutput: Bool = true
    ) -> Gaussians3D {
        let actualScaleFactor = baseValues.meanXNDC.shape[4] / delta.shape[4]
        let upsampledDelta = (scaleFactor != 1 && actualScaleFactor != 1)
            ? SharpTensorOps.upsampleNearest5D(delta, scaleFactor: scaleFactor)
            : delta

        var meanVectors = forwardMean(baseValues: baseValues, delta: upsampledDelta)
        let baseScales = baseScaleOnPredictedMean
            ? (baseValues.scales * baseValues.meanInverseZNDC * meanVectors[0..., 2 ..< 3, 0..., 0..., 0...])
            : baseValues.scales
        var singularValues = scaleActivation(
            base: baseScales,
            learnedDelta: upsampledDelta[0..., 3 ..< 6, 0..., 0..., 0...]
        )
        var quaternions = quaternionActivation(
            base: baseValues.quaternions,
            learnedDelta: upsampledDelta[0..., 6 ..< 10, 0..., 0..., 0...]
        )
        var colors = colorActivation(
            base: baseValues.colors,
            learnedDelta: upsampledDelta[0..., 10 ..< 13, 0..., 0..., 0...]
        )
        var opacities = opacityActivation(
            base: baseValues.opacities,
            learnedDelta: upsampledDelta[0..., 13, 0..., 0..., 0...]
        )

        if flattenOutput {
            let batch = meanVectors.shape[0]
            meanVectors = meanVectors.transposed(0, 2, 3, 4, 1).reshaped([batch, -1, 3])
            singularValues = singularValues.transposed(0, 2, 3, 4, 1).reshaped([batch, -1, 3])
            quaternions = quaternions.transposed(0, 2, 3, 4, 1).reshaped([batch, -1, 4])
            colors = colors.transposed(0, 2, 3, 4, 1).reshaped([batch, -1, 3])
            opacities = opacities.reshaped([batch, -1])
        }

        if let globalScale {
            let meanScale = globalScale.expandedDimensions(axis: 1).expandedDimensions(axis: 2)
            meanVectors = meanVectors * meanScale
            singularValues = singularValues * meanScale
        }

        return Gaussians3D(
            meanVectors: meanVectors,
            singularValues: singularValues,
            quaternions: quaternions,
            colors: colors,
            opacities: opacities
        )
    }

    private func forwardMean(baseValues: GaussianBaseValues, delta: MLXArray) -> MLXArray {
        let base = concatenated(
            [baseValues.meanXNDC, baseValues.meanYNDC, baseValues.meanInverseZNDC],
            axis: 1
        )
        let factors = MLXArray([deltaFactor.xy, deltaFactor.xy, deltaFactor.z], [1, 3, 1, 1, 1])
        return meanActivation(base: base, learnedDelta: factors * delta[0..., 0 ..< 3, 0..., 0..., 0...])
    }

    private func meanActivation(base: MLXArray, learnedDelta: MLXArray) -> MLXArray {
        let xx = base[0..., 0 ..< 1, 0..., 0..., 0...] + learnedDelta[0..., 0 ..< 1, 0..., 0..., 0...]
        let yy = base[0..., 1 ..< 2, 0..., 0..., 0...] + learnedDelta[0..., 1 ..< 2, 0..., 0..., 0...]
        let a = base[0..., 2 ..< 3, 0..., 0..., 0...]
        let b = learnedDelta[0..., 2 ..< 3, 0..., 0..., 0...]
        let inverseZZ = MLXNN.softplus(SharpMath.inverseSoftplus(a) + b)
        let zz = 1.0 / (inverseZZ + 1e-3)
        return concatenated([zz * xx, zz * yy, zz], axis: 1)
    }

    private func scaleActivation(base: MLXArray, learnedDelta: MLXArray) -> MLXArray {
        let constants = scaleActivationConstants()
        let scaled = sigmoid(constants.a * deltaFactor.scale * learnedDelta + constants.b)
        return base * ((maxScale - minScale) * scaled + minScale)
    }

    private func quaternionActivation(base: MLXArray, learnedDelta: MLXArray) -> MLXArray {
        base + deltaFactor.quaternion * learnedDelta
    }

    private func colorActivation(base: MLXArray, learnedDelta: MLXArray) -> MLXArray {
        var clampedBase = base
        switch colorActivationType {
        case .sigmoid:
            clampedBase = clip(base, min: 0.01, max: 0.99)
        case .exp, .softplus:
            clampedBase = clip(base, min: 0.01)
        case .linear:
            break
        }

        let activation = SharpMath.createActivationPair(colorActivationType)
        var colors = activation.forward(activation.inverse(clampedBase) + deltaFactor.color * learnedDelta)
        if colorSpace == .linearRGB {
            colors = SharpColorSpace.sRGBToLinearRGB(colors)
        }
        return colors
    }

    private func opacityActivation(base: MLXArray, learnedDelta: MLXArray) -> MLXArray {
        let activation = SharpMath.createActivationPair(opacityActivationType)
        return activation.forward(activation.inverse(base) + deltaFactor.opacity * learnedDelta)
    }

    private func scaleActivationConstants() -> (a: Float, b: Float) {
        let a = (maxScale - minScale) / (1 - minScale) / (maxScale - 1)
        let ratio = (1.0 - minScale) / (maxScale - minScale)
        let b = SharpMath.inverseSigmoid(MLXArray(ratio)).item(Float.self)
        return (a, b)
    }
}
