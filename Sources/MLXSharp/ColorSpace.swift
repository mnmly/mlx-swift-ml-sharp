@preconcurrency import MLX

public enum SharpColorSpace {
    public static func sRGBToLinearRGB(_ sRGB: MLXArray) -> MLXArray {
        let threshold: Float = 0.04045
        let lower = sRGB / 12.92
        let upper = pow((sRGB + 0.055) / 1.055, 2.4)
        return which(sRGB .<= threshold, lower, upper)
    }

    public static func linearRGBTosRGB(_ linearRGB: MLXArray) -> MLXArray {
        let threshold: Float = 0.0031308
        let lower = linearRGB * 12.92
        let upper = 1.055 * pow(linearRGB, 1.0 / 2.4) - 0.055
        return which(linearRGB .<= threshold, lower, upper)
    }
}
