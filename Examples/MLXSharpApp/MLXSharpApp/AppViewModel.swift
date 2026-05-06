import AppKit
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
    var previewImage: NSImage?
    var plyData: Data?
    var isWorking = false
    var status = "Select a .safetensors weights file and an image."
    var gaussianCount: Int?

    var canProcess: Bool { weightsURL != nil && cgImage != nil && !isWorking }
    var canExport: Bool { plyData != nil && !isWorking }

    private var cgImage: CGImage?
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

        guard
            let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            status = "Could not read image."
            return
        }
        cgImage = img
        previewImage = NSImage(cgImage: img, size: .zero)
        plyData = nil
        gaussianCount = nil
        updateStatus()
    }

    func process() {
        guard let wURL = weightsURL, let cg = cgImage else { return }
        isWorking = true
        let focal = Float(cg.width) * 0.58

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
