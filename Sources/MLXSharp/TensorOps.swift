@preconcurrency import MLX
@preconcurrency import MLXNN

enum SharpTensorOps {
    static func nchwToNhwc(_ x: MLXArray) -> MLXArray {
        x.transposed(0, 2, 3, 1)
    }

    static func nhwcToNchw(_ x: MLXArray) -> MLXArray {
        x.transposed(0, 3, 1, 2)
    }

    static func avgPool2dNCHW(_ x: MLXArray, kernelSize: Int, stride: Int) -> MLXArray {
        let pool = AvgPool2d(kernelSize: IntOrPair(kernelSize), stride: IntOrPair(stride))
        return nhwcToNchw(pool(nchwToNhwc(x)))
    }

    static func maxPool2dNCHW(_ x: MLXArray, kernelSize: Int, stride: Int) -> MLXArray {
        let pool = MaxPool2d(kernelSize: IntOrPair(kernelSize), stride: IntOrPair(stride))
        return nhwcToNchw(pool(nchwToNhwc(x)))
    }

    static func minPool2dNCHW(_ x: MLXArray, kernelSize: Int, stride: Int) -> MLXArray {
        -maxPool2dNCHW(-x, kernelSize: kernelSize, stride: stride)
    }

    static func repeatAxis(_ x: MLXArray, count: Int, axis: Int) -> MLXArray {
        repeated(x, count: count, axis: axis)
    }

    static func upsampleNearest5D(_ x: MLXArray, scaleFactor: Int) -> MLXArray {
        let upH = repeatAxis(x, count: scaleFactor, axis: 3)
        return repeatAxis(upH, count: scaleFactor, axis: 4)
    }
}
