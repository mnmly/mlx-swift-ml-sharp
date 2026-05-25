import CoreGraphics
import ImageIO
import MLX
import SwiftUI
@preconcurrency import MLXSharp

// MARK: - ModelRunner

/// Owns the pipeline and runs heavy inference off the @MainActor.
/// Storing SharpPipeline here means it never crosses an isolation boundary.
private actor ModelRunner {
    private var pipeline: SharpPipeline?

    func loadPipelineIfNeeded(from url: URL) throws {
        guard pipeline == nil else { return }
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        pipeline = try SharpPipeline.fromPretrained(url)
    }

    func invalidatePipeline() {
        pipeline = nil
    }

    func runInference(cgImage: CGImage, focalLength: Float) -> (data: Data, gaussianCount: Int) {
        guard let p = pipeline else { fatalError("Pipeline not loaded") }
        let prediction = p(cgImage, focalLengthPixels: focalLength)
        let n = prediction.gaussians.meanVectors.shape[1]
        let data = PLYExporter.export(prediction)
        return (data, n)
    }
}

// MARK: - AppViewModel

@Observable
@MainActor
final class AppViewModel {
    var weightsURL: URL?
    var imageURL: URL?
    var previewImage: CGImage?
    var plyData: Data?
    var isWorking = false
    var status = "Select a .safetensors weights file and an image."
    var gaussianCount: Int?

    var canProcess: Bool { weightsURL != nil && cgImage != nil && !isWorking }
    var canExport: Bool { plyData != nil && !isWorking }

    private var cgImage: CGImage?
    /// EXIF-derived focal length in pixels, or nil if EXIF was missing.
    private var exifFocalPixels: Float?
    private let runner = ModelRunner()

    func setWeightsURL(_ url: URL) {
        weightsURL = url
        Task { await runner.invalidatePipeline() }
        plyData = nil
        gaussianCount = nil
        updateStatus()
    }

    func setImageURL(_ url: URL) {
        imageURL = url
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }

        // Read bytes eagerly while we hold the security scope; decoding from
        // in-memory Data ensures the CGImage doesn't lazy-reopen the file
        // later (which would fail once the picker's scope is released).
        guard let data = try? Data(contentsOf: url),
              let src = CGImageSourceCreateWithData(data as CFData, nil),
              let img = CGImageSourceCreateImageAtIndex(
                src, 0,
                [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        else {
            status = "Could not read image."
            return
        }
        cgImage = img
        // Resolve focal length once from EXIF (with the same fallback chain as
        // the upstream Python pipeline). Falls back to 30mm if EXIF is absent.
        exifFocalPixels = SharpCameraIntrinsics.focalLengthPixels(
            from: src,
            imageWidth: img.width,
            imageHeight: img.height
        )
        previewImage = img
        plyData = nil
        gaussianCount = nil
        updateStatus()
    }

    func process() {
        guard let wURL = weightsURL, let cg = cgImage else { return }
        isWorking = true
        let focal = exifFocalPixels ?? SharpCameraIntrinsics.focalLengthPixels(
            focalMM: 30,
            imageWidth: cg.width,
            imageHeight: cg.height
        )

        // Task inherits @MainActor isolation; await hops into ModelRunner's executor.
        Task {
            do {
                status = "Loading model weights…"
                try await runner.loadPipelineIfNeeded(from: wURL)

                status = "Running inference…"
                let (data, n) = await runner.runInference(cgImage: cg, focalLength: focal)

                plyData = data
                gaussianCount = n
                status = "Done — \(n) Gaussians exported."
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
            isWorking = false
        }
    }

    private func updateStatus() {
        switch (weightsURL, cgImage) {
        case (nil, nil): status = "Select a .safetensors weights file and an image."
        case (nil, _):   status = "Select a .safetensors weights file."
        case (_, nil):   status = "Select an image."
        default:         status = "Ready — tap Process to run inference."
        }
    }
}
