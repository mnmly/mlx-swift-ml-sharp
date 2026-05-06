@preconcurrency import MLX

public protocol GaussianInitializing {
    func callAsFunction(_ image: MLXArray, _ depth: MLXArray) -> InitializerOutput
}

public func createInitializer(params: InitializerParams) -> MultiLayerInitializer {
    MultiLayerInitializer(params: params)
}

public struct MultiLayerInitializer: GaussianInitializing {
    public let params: InitializerParams

    public init(params: InitializerParams) {
        self.params = params
    }

    public func prepareFeatureInput(image: MLXArray, depth: MLXArray) -> MLXArray {
        let normalizedDisparity = params.disparityFactor / depth
        return 2.0 * concatenated([image, normalizedDisparity], axis: 1) - 1.0
    }

    public func callAsFunction(_ image: MLXArray, _ depth: MLXArray) -> InitializerOutput {
        let batchSize = depth.shape[0]
        let imageHeight = depth.shape[2]
        let imageWidth = depth.shape[3]
        let baseHeight = imageHeight / params.stride
        let baseWidth = imageWidth / params.stride

        var workingDepth = depth
        var globalScale: MLXArray?
        if params.normalizeDepth {
            let pair = rescaleDepth(depth)
            workingDepth = pair.depth
            globalScale = 1.0 / pair.factor
        }

        func createDisparityLayers(_ layerCount: Int = 1) -> MLXArray {
            let disparity = MLXArray.linspace(1.0 / params.baseDepth, 0.0, count: layerCount + 1)
            let trimmed = disparity[..<layerCount]
            let base = trimmed.reshaped([1, 1, layerCount, 1, 1])
            return broadcast(base, to: [batchSize, 1, layerCount, baseHeight, baseWidth])
        }

        func createSurfaceLayer(_ sourceDepth: MLXArray, pooling: DepthInitOption) -> MLXArray {
            let disparity = 1.0 / sourceDepth
            let pooled: MLXArray
            switch pooling {
            case .surfaceMin:
                pooled = SharpTensorOps.maxPool2dNCHW(disparity, kernelSize: params.stride, stride: params.stride)
            case .surfaceMax:
                pooled = SharpTensorOps.minPool2dNCHW(disparity, kernelSize: params.stride, stride: params.stride)
            default:
                fatalError("Invalid surface pooling mode \(pooling)")
            }
            return pooled.expandedDimensions(axis: 2)
        }

        let firstDisparity: MLXArray
        switch params.firstLayerDepthOption {
        case .surfaceMin, .surfaceMax:
            firstDisparity = createSurfaceLayer(workingDepth[0..., 0 ..< 1, 0..., 0...], pooling: params.firstLayerDepthOption)
        case .baseDepth, .linearDisparity:
            firstDisparity = createDisparityLayers(1)
        }

        let disparity: MLXArray
        if params.numLayers == 1 {
            disparity = firstDisparity
        } else {
            let followingDepth = workingDepth.shape[1] == 1
                ? workingDepth
                : workingDepth[0..., 1..., 0..., 0...]

            let followingDisparity: MLXArray
            switch params.restLayerDepthOption {
            case .surfaceMin, .surfaceMax:
                followingDisparity = createSurfaceLayer(followingDepth, pooling: params.restLayerDepthOption)
            case .baseDepth:
                let pieces = Array(repeating: createDisparityLayers(1), count: params.numLayers - 1)
                followingDisparity = concatenated(pieces, axis: 2)
            case .linearDisparity:
                followingDisparity = createDisparityLayers(params.numLayers - 1)
            }

            disparity = concatenated([firstDisparity, followingDisparity], axis: 2)
        }

        let baseXY = createBaseXY(
            batchSize: batchSize,
            imageHeight: imageHeight,
            imageWidth: imageWidth,
            stride: params.stride,
            numLayers: params.numLayers
        )
        let disparityScaleFactor = 2.0 * params.scaleFactor * Float(params.stride) / Float(imageWidth)
        let baseScales = createBaseScale(disparity: disparity, disparityScaleFactor: disparityScaleFactor)
        let baseQuaternions = broadcast(
            MLXArray([Float(1.0), Float(0.0), Float(0.0), Float(0.0)], [1, 4, 1, 1, 1]),
            to: [batchSize, 4, 1, 1, 1]
        )
        let baseOpacities = broadcast(
            MLXArray([Float(min(Float(1.0) / Float(params.numLayers), 0.5))], [1, 1, 1, 1]),
            to: [batchSize, params.numLayers, baseHeight, baseWidth]
        )

        let pooledImage = SharpTensorOps.avgPool2dNCHW(image, kernelSize: params.stride, stride: params.stride)
        let baseColors: MLXArray
        switch params.colorOption {
        case .none:
            baseColors = full(
                [batchSize, 3, params.numLayers, baseHeight, baseWidth],
                values: MLXArray(0.5),
                type: Float.self
            )
        case .firstLayer:
            let gray = full(
                [batchSize, 3, params.numLayers, baseHeight, baseWidth],
                values: MLXArray(0.5),
                type: Float.self
            )
            var layers = [MLXArray]()
            layers.append(pooledImage.expandedDimensions(axis: 2))
            if params.numLayers > 1 {
                layers.append(gray[0..., 0..., 1..., 0..., 0...])
            }
            baseColors = concatenated(layers, axis: 2)
        case .allLayers:
            let expanded = pooledImage.expandedDimensions(axis: 2)
            baseColors = broadcast(expanded, to: [batchSize, 3, params.numLayers, baseHeight, baseWidth])
        }

        let baseValues = GaussianBaseValues(
            meanXNDC: baseXY.x,
            meanYNDC: baseXY.y,
            meanInverseZNDC: disparity,
            scales: baseScales,
            quaternions: baseQuaternions,
            colors: baseColors,
            opacities: baseOpacities
        )

        return InitializerOutput(
            gaussianBaseValues: baseValues,
            featureInput: prepareFeatureInput(image: image, depth: workingDepth),
            globalScale: globalScale
        )
    }

    private func createBaseXY(
        batchSize: Int,
        imageHeight: Int,
        imageWidth: Int,
        stride: Int,
        numLayers: Int
    ) -> (x: MLXArray, y: MLXArray) {
        let xx = (arange(Double(stride) * 0.5, Double(imageWidth), step: Double(stride), dtype: .float32) * (2.0 / Float(imageWidth))) - 1.0
        let yy = (arange(Double(stride) * 0.5, Double(imageHeight), step: Double(stride), dtype: .float32) * (2.0 / Float(imageHeight))) - 1.0
        let grid = meshGrid([xx, yy], indexing: .xy)
        let xBase = grid[0].transposed(1, 0).expandedDimensions(axis: 0).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let yBase = grid[1].transposed(1, 0).expandedDimensions(axis: 0).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let shape = [batchSize, 1, numLayers, imageHeight / stride, imageWidth / stride]
        return (broadcast(xBase, to: shape), broadcast(yBase, to: shape))
    }

    private func createBaseScale(disparity: MLXArray, disparityScaleFactor: Float) -> MLXArray {
        (1.0 / disparity) * disparityScaleFactor
    }

    private func rescaleDepth(_ depth: MLXArray, depthMin: Float = 1.0, depthMax: Float = 1e2)
        -> (depth: MLXArray, factor: MLXArray)
    {
        let batchSize = depth.shape[0]
        let flattened = depth.reshaped([batchSize, -1])
        let currentDepthMin = flattened.min(axis: 1)
        let factor = depthMin / (currentDepthMin + 1e-6)
        let expanded = factor.expandedDimensions(axis: 1).expandedDimensions(axis: 2).expandedDimensions(axis: 3)
        return (clip(depth * expanded, max: depthMax), factor)
    }
}
