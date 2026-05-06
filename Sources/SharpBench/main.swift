import Foundation
import MLX
import MLXSharp

// MARK: - Result types (JSON-serialisable, matches RFDETRBench schema)

struct BenchmarkResult: Codable {
    let label: String
    let iterations: Int
    let warmup: Int
    let dtype: String
    let inputShape: [Int]
    let stageStatsMs: [String: StageStats]
    let totalStatsMs: StageStats
}

struct StageStats: Codable {
    let mean: Double
    let median: Double
    let min: Double
    let max: Double
    let stddev: Double
}

// MARK: - CLI config

struct BenchConfig {
    var weights: String? = nil
    var imageSize: (Int, Int) = (1536, 1536)  // H, W
    var iterations: Int = 10
    var warmup: Int = 3
    var dtype: DType = .float32
    var label: String = "swift-mlx"
    var focalLengthPixels: Float = 0.0  // 0 = imageWidth * 0.58
    var captureURL: URL? = nil          // if set, write a .gputrace after warmup
    var captureStage: String = "monodepth"  // which stage to capture
}

func parseArgs(_ args: [String]) throws -> BenchConfig {
    var cfg = BenchConfig()
    var i = 0
    while i < args.count {
        switch args[i] {
        case "--weights":
            i += 1; cfg.weights = args[i]
        case "--size":
            i += 1
            let parts = args[i].split(separator: "x").compactMap { Int($0) }
            guard parts.count == 2 else { throw BenchError.badArg("--size expects HxW") }
            cfg.imageSize = (parts[0], parts[1])
        case "--iterations":
            i += 1; cfg.iterations = Int(args[i]) ?? cfg.iterations
        case "--warmup":
            i += 1; cfg.warmup = Int(args[i]) ?? cfg.warmup
        case "--dtype":
            i += 1; cfg.dtype = try parseDType(args[i])
        case "--label":
            i += 1; cfg.label = args[i]
        case "--focal":
            i += 1; cfg.focalLengthPixels = Float(args[i]) ?? 0
        case "--capture":
            i += 1; cfg.captureURL = URL(fileURLWithPath: args[i])
        case "--capture-stage":
            i += 1; cfg.captureStage = args[i]
        default:
            throw BenchError.badArg("unknown argument: \(args[i])")
        }
        i += 1
    }
    return cfg
}

func parseDType(_ raw: String) throws -> DType {
    switch raw.lowercased() {
    case "float16", "fp16": return .float16
    case "float32", "fp32": return .float32
    default: throw BenchError.badArg("unsupported dtype: \(raw)")
    }
}

enum BenchError: Error, LocalizedError {
    case badArg(String)
    var errorDescription: String? {
        if case .badArg(let m) = self { return m }
        return nil
    }
}

// MARK: - Stage timing helpers

private func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
private func elapsedMs(since t: UInt64) -> Double { Double(nowNs() - t) / 1_000_000 }

private func stats(_ samples: [Double]) -> StageStats {
    let sorted = samples.sorted()
    let n = Double(samples.count)
    let mean = samples.reduce(0, +) / n
    let variance = samples.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / n
    let mid = sorted.count / 2
    let median: Double = sorted.count.isMultiple(of: 2)
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid]
    return StageStats(mean: mean, median: median,
                      min: sorted.first ?? 0, max: sorted.last ?? 0,
                      stddev: sqrt(variance))
}

// MARK: - Benchmark runner

func runBenchmark(predictor: RGBGaussianPredictor, cfg: BenchConfig) -> BenchmarkResult {
    let (H, W) = cfg.imageSize
    let fov = cfg.focalLengthPixels == 0 ? Float(W) * 0.58 : cfg.focalLengthPixels
    let disparityFactor = MLXArray([fov / Float(W)], [1])
    let image = MLXArray.ones([1, 3, H, W], dtype: cfg.dtype) * 0.5

    // Grab typed references to run stages individually
    let monodepthModel = predictor.monodepthModel
    let featureModel = predictor.featureModel
    let predictionHead = predictor.predictionHead
    let composer = predictor.gaussianComposer
    let initializer = predictor.initializer
    let depthAlignment = predictor.depthAlignment

    var stageSamples: [String: [Double]] = [
        "monodepth": [], "init_align": [], "feature_model": [],
        "prediction_head": [], "composer": []
    ]
    var totalSamples: [Double] = []

    // Capture only the first post-warmup iteration.
    let captureStep = cfg.warmup

    for step in 0..<(cfg.warmup + cfg.iterations) {
        // For each stage, start capture on captureStep if that stage is selected.
        func startCapture(for stage: String) {
            guard step == captureStep, stage == cfg.captureStage,
                  let url = cfg.captureURL else { return }
            GPU.startCapture(url: url)
        }
        func stopCapture(for stage: String) {
            guard step == captureStep, stage == cfg.captureStage,
                  let url = cfg.captureURL else { return }
            GPU.stopCapture(url: url)
            fputs("Metal trace written to \(url.path)\n", stderr)
        }

        let t0 = nowNs()

        // Stage 1 — monodepth
        let t1 = nowNs()
        startCapture(for: "monodepth")
        let monoOut = monodepthModel(image)
        eval(monoOut.disparity)
        stopCapture(for: "monodepth")
        let dtMono = elapsedMs(since: t1)

        // Stage 2 — disparity scaling + depth alignment + initializer
        let t2 = nowNs()
        let dScale = disparityFactor
            .expandedDimensions(axis: 1)
            .expandedDimensions(axis: 2)
            .expandedDimensions(axis: 3)
        let monodepth = dScale / clip(monoOut.disparity, min: 1e-4, max: 1e4)
        let (alignedDepth, _) = depthAlignment(monodepth, nil, monoOut.decoderFeatures)
        let initOut = initializer(image, alignedDepth)
        startCapture(for: "init_align")
        eval(initOut.featureInput)
        stopCapture(for: "init_align")
        let dtInit = elapsedMs(since: t2)

        // Stage 3 — feature model (GaussianDPT)
        let t3 = nowNs()
        let imageFeatures = featureModel(initOut.featureInput, encodings: monoOut.outputFeatures)
        startCapture(for: "feature_model")
        eval(imageFeatures.geometryFeatures, imageFeatures.textureFeatures)
        stopCapture(for: "feature_model")
        let dtFeat = elapsedMs(since: t3)

        // Stage 4 — prediction head
        let t4 = nowNs()
        let delta = predictionHead(imageFeatures)
        startCapture(for: "prediction_head")
        eval(delta)
        stopCapture(for: "prediction_head")
        let dtHead = elapsedMs(since: t4)

        // Stage 5 — composer
        let t5 = nowNs()
        let gaussians = composer(
            delta: delta,
            baseValues: initOut.gaussianBaseValues,
            globalScale: initOut.globalScale
        )
        startCapture(for: "composer")
        eval(gaussians.meanVectors, gaussians.singularValues,
             gaussians.quaternions, gaussians.colors, gaussians.opacities)
        stopCapture(for: "composer")
        let dtComp = elapsedMs(since: t5)

        let total = elapsedMs(since: t0)

        if step >= cfg.warmup {
            stageSamples["monodepth"]!.append(dtMono)
            stageSamples["init_align"]!.append(dtInit)
            stageSamples["feature_model"]!.append(dtFeat)
            stageSamples["prediction_head"]!.append(dtHead)
            stageSamples["composer"]!.append(dtComp)
            totalSamples.append(total)
        }
    }

    return BenchmarkResult(
        label: cfg.label,
        iterations: cfg.iterations,
        warmup: cfg.warmup,
        dtype: String(describing: cfg.dtype),
        inputShape: [1, 3, H, W],
        stageStatsMs: stageSamples.mapValues(stats),
        totalStatsMs: stats(totalSamples)
    )
}

// MARK: - Entry point

do {
    let cfg = try parseArgs(Array(CommandLine.arguments.dropFirst()))

    let predictor: RGBGaussianPredictor
    if let path = cfg.weights {
        let url = URL(fileURLWithPath: path)
        predictor = try loadSharpModel(from: url)
        fputs("Loaded weights from \(path)\n", stderr)
    } else {
        // Random-weight predictor for timing without real weights
        predictor = makeRandomPredictor()
        fputs("No --weights provided; using random initialisation.\n", stderr)
    }

    let result = runBenchmark(predictor: predictor, cfg: cfg)

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(result)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
} catch {
    fputs("error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
