import Foundation
import MLX
import Testing
@testable import MLXSharp

// MARK: - Weight loading smoke tests
// Requires sharp_2572gikvuh.safetensors at the package root.
// Skips gracefully if the file is absent.

private let weightsURL: URL? = {
    // Resolve relative to this source file's directory → walk up to package root
    let src = URL(fileURLWithPath: #filePath)
    let packageRoot = src
        .deletingLastPathComponent()  // WeightLoadingTests.swift → MLXSharpTests/
        .deletingLastPathComponent()  // MLXSharpTests/ → Tests/
        .deletingLastPathComponent()  // Tests/ → package root
    let candidate = packageRoot.appendingPathComponent("sharp_2572gikvuh.safetensors")
    return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
}()

@Suite("Weight Loading")
struct WeightLoadingTests {

    @Test("Model loads without error")
    func modelLoads() throws {
        guard let url = weightsURL else {
            print("Skipping: sharp_2572gikvuh.safetensors not found")
            return
        }
        _ = try loadSharpModel(from: url)
    }

    @Test("Forward pass produces correct shapes")
    func forwardPassShapes() throws {
        guard let url = weightsURL else {
            print("Skipping: sharp_2572gikvuh.safetensors not found")
            return
        }

        let predictor = try loadSharpModel(from: url)

        // SPN sliding pyramid requires each level's feature map to be 2× the next.
        // enc4=48×48, enc3=96×96, ... → input must be 1536×1536 (= 4 × imgSize).
        // NCHW [1, 3, 1536, 1536] in [0, 1]
        let H = 1536, W = 1536
        let image = MLXArray.ones([1, 3, H, W], dtype: .float32) * 0.5
        let disparityFactor = MLXArray([1.0] as [Float])

        let gaussians = predictor(image, disparityFactor: disparityFactor)
        eval(gaussians.meanVectors, gaussians.singularValues,
             gaussians.quaternions, gaussians.colors, gaussians.opacities)

        // stride=2, numLayers=2 → (H/2)×(W/2)×2 = 384×384×2 = 294912 Gaussians
        let N = (H / 2) * (W / 2) * 2
        #expect(gaussians.meanVectors.shape == [1, N, 3])
        #expect(gaussians.singularValues.shape == [1, N, 3])
        #expect(gaussians.quaternions.shape == [1, N, 4])
        #expect(gaussians.colors.shape == [1, N, 3])
        #expect(gaussians.opacities.shape == [1, N])
    }

    @Test("Output contains no NaNs or Infs")
    func outputNumerics() throws {
        guard let url = weightsURL else {
            print("Skipping: sharp_2572gikvuh.safetensors not found")
            return
        }

        let predictor = try loadSharpModel(from: url)
        // Model requires ≥1536×1536 input (= 4 × imgSize=384) so the sliding pyramid
        // produces enc[3] at 2× the spatial size of enc[4].
        let image = MLXArray.ones([1, 3, 1536, 1536], dtype: .float32) * 0.5
        let disparityFactor = MLXArray([1.0] as [Float])

        let gaussians = predictor(image, disparityFactor: disparityFactor)

        for (name, arr) in [
            ("meanVectors", gaussians.meanVectors),
            ("singularValues", gaussians.singularValues),
            ("quaternions", gaussians.quaternions),
            ("colors", gaussians.colors),
            ("opacities", gaussians.opacities),
        ] {
            eval(arr)
            let hasNaN = MLX.isNaN(arr).any().item(Bool.self)
            let hasInf = MLX.isInf(arr).any().item(Bool.self)
            #expect(!hasNaN, "\(name) contains NaN")
            #expect(!hasInf, "\(name) contains Inf")
        }
    }
}
