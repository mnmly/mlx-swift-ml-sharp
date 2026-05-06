@preconcurrency import MLX
@preconcurrency import MLXNN

public struct ActivationPair {
    public let forward: (MLXArray) -> MLXArray
    public let inverse: (MLXArray) -> MLXArray
}

public enum SharpMath {
    public static func createActivationPair(_ type: ActivationType) -> ActivationPair {
        switch type {
        case .linear:
            return ActivationPair(forward: { $0 }, inverse: { $0 })
        case .exp:
            return ActivationPair(forward: { MLX.exp($0) }, inverse: { MLX.log($0) })
        case .sigmoid:
            return ActivationPair(forward: sigmoid(_:), inverse: inverseSigmoid(_:))
        case .softplus:
            return ActivationPair(forward: softplus(_:), inverse: { SharpMath.inverseSoftplus($0) })
        }
    }

    public static func inverseSigmoid(_ x: MLXArray) -> MLXArray {
        log(x / (1.0 - x))
    }

    public static func inverseSoftplus(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let clipped = clip(x, min: eps)
        return log(exp(clipped) - 1.0)
    }
}
