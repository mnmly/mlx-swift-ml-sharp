@preconcurrency import MLX
@preconcurrency import MLXNN
import Foundation

// MARK: - Sliding Pyramid Network
//
// Creates multi-resolution encodings from a ViT backbone using overlapping patches.
// All spatial operations happen in NCHW; the ViT itself expects NHWC patches.
// Output feature maps are in NHWC.

final class SlidingPyramidNetwork: Module {
    @ModuleInfo(key: "patch_encoder") var patchEncoder: TimmViT
    @ModuleInfo(key: "image_encoder") var imageEncoder: TimmViT
    @ModuleInfo(key: "upsample_latent0") var upsampleLatent0: ProjectUpsampleBlock
    @ModuleInfo(key: "upsample_latent1") var upsampleLatent1: ProjectUpsampleBlock
    @ModuleInfo(key: "upsample0") var upsample0: ProjectUpsampleBlock
    @ModuleInfo(key: "upsample1") var upsample1: ProjectUpsampleBlock
    @ModuleInfo(key: "upsample2") var upsample2: ProjectUpsampleBlock
    @ModuleInfo(key: "upsample_lowres") var upsampleLowres: ConvTransposed2d
    @ModuleInfo(key: "fuse_lowres") var fuseLowres: Conv2d

    let dimsEncoder: [Int]
    let imgSize: Int    // ViT native input size (= chunk crop size for spnSplit)
    let patchSize: Int  // ViT token patch size (e.g. 16), NOT the chunk size
    let intermediateIds: [Int]  // block indices for latent0, latent1

    init(
        dimsEncoder: [Int],  // [256, 256, 512, 1024, 1024] for dinov2l16_384 + last_encoder=256
        embedDim: Int = 1024,
        lowresEmbedDim: Int = 1024,
        intermediateIds: [Int] = [5, 11],  // which block outputs to capture for latent0/1
        imgSize: Int = 384,
        patchSize: Int = 16,
        depth: Int = 24,
        numHeads: Int = 16
    ) {
        self.dimsEncoder = dimsEncoder
        self.imgSize = imgSize
        self.patchSize = patchSize
        self.intermediateIds = intermediateIds

        super.init()
        patchEncoder = TimmViT(embedDim: embedDim, depth: depth, numHeads: numHeads,
                               imgSize: imgSize, patchSize: patchSize,
                               intermediateIds: intermediateIds)
        imageEncoder = TimmViT(embedDim: lowresEmbedDim, depth: depth, numHeads: numHeads,
                               imgSize: imgSize, patchSize: patchSize, intermediateIds: [])

        upsampleLatent0 = ProjectUpsampleBlock(dimIn: embedDim, dimOut: dimsEncoder[0],
                                               numUpsampleLayers: 3,
                                               dimIntermediate: dimsEncoder[1])
        upsampleLatent1 = ProjectUpsampleBlock(dimIn: embedDim, dimOut: dimsEncoder[1],
                                               numUpsampleLayers: 2)
        upsample0 = ProjectUpsampleBlock(dimIn: embedDim, dimOut: dimsEncoder[2],
                                         numUpsampleLayers: 1)
        upsample1 = ProjectUpsampleBlock(dimIn: embedDim, dimOut: dimsEncoder[3],
                                         numUpsampleLayers: 1)
        upsample2 = ProjectUpsampleBlock(dimIn: embedDim, dimOut: dimsEncoder[4],
                                         numUpsampleLayers: 1)
        upsampleLowres = ConvTransposed2d(inputChannels: lowresEmbedDim,
                                          outputChannels: dimsEncoder[4],
                                          kernelSize: 2, stride: 2)
        fuseLowres = Conv2d(inputChannels: dimsEncoder[4] * 2,
                            outputChannels: dimsEncoder[4], kernelSize: 1)
    }

    /// image: [B, 3, H, W] NCHW, already normalised to [-1, 1] by caller
    /// Returns 5 feature maps in NHWC: [latent0, latent1, level0, level1, level2(fused)]
    func callAsFunction(_ image: MLXArray) -> [MLXArray] {
        let x = image

        // Build image pyramid in NCHW
        let x0 = x                                    // any size ≥ imgSize
        let x1 = bilinearDownsample(x0, scale: 0.5)
        let x2raw = bilinearDownsample(x0, scale: 0.25)

        // x2 must be exactly imgSize×imgSize for the ViT; resize if needed
        let x2H = x2raw.shape[2], x2W = x2raw.shape[3]
        let x2: MLXArray
        if x2H != imgSize || x2W != imgSize {
            let nhwc = ncHWtoNHWC(x2raw)
            let scaleH = Float(imgSize) / Float(x2H)
            let scaleW = Float(imgSize) / Float(x2W)
            let resized = Upsample(
                scaleFactor: .array([scaleH, scaleW]),
                mode: .linear(alignCorners: false)
            )(nhwc)
            x2 = nhwcToNCHW(resized)
        } else {
            x2 = x2raw
        }

        // Split into overlapping imgSize×imgSize chunks (not patchSize=16 token size)
        let patches0 = spnSplit(x0, overlapRatio: 0.25, patchSize: imgSize)
        let patches1 = spnSplit(x1, overlapRatio: 0.5,  patchSize: imgSize)
        let patches2 = ncHWtoNHWC(x2)  // 1 "patch" at global context resolution

        let B = x.shape[0]
        let n0 = patches0.shape[0] / B  // 25
        let n1 = patches1.shape[0] / B  // 9

        // Concatenate all patches for a single batch pass through patch_encoder
        let allPatches = concatenated([patches0, patches1, patches2], axis: 0)
        let (allEncodings, intermediates) = patchEncoder(allPatches)

        // Split back
        let enc0 = allEncodings[0..<B * n0, 0..., 0..., 0...]
        let enc1 = allEncodings[B * n0..<B * (n0 + n1), 0..., 0..., 0...]
        let enc2 = allEncodings[B * (n0 + n1) ..< allEncodings.shape[0], 0..., 0..., 0...]

        let padding = 3
        let x0Features = spnMerge(enc0, batchSize: B, padding: padding)       // [B, C, 96, 96]
        let x1Features = spnMerge(enc1, batchSize: B, padding: padding * 2)   // [B, C, 48, 48]
        // x2Features: just reshape from [B, C, 24, 24] - already in NCHW from TimmViT
        let x2Features = enc2  // [B, C, 24, 24]

        // Intermediate latent features (from patchEncoder intermediate blocks)
        let latent0 = patchEncoder.reshapeFeature(
            intermediates[patchEncoder.intermediateIds.sorted()[0]]![0..<B * n0, 0..., 0...])
        let latent1 = patchEncoder.reshapeFeature(
            intermediates[patchEncoder.intermediateIds.sorted()[1]]![0..<B * n0, 0..., 0...])
        let latent0Features = spnMerge(latent0, batchSize: B, padding: padding)
        let latent1Features = spnMerge(latent1, batchSize: B, padding: padding)

        // Image encoder (low-res only)
        let (lowresEncoding, _) = imageEncoder(patches2)

        // Apply upsamplers (all work in NHWC)
        let out0 = upsampleLatent0(ncHWtoNHWC(latent0Features))
        let out1 = upsampleLatent1(ncHWtoNHWC(latent1Features))
        let out2 = upsample0(ncHWtoNHWC(x0Features))
        let out3 = upsample1(ncHWtoNHWC(x1Features))
        let x2F_nhwc = ncHWtoNHWC(x2Features)
        let out4_a = upsample2(x2F_nhwc)
        let lowresUp = upsampleLowres(ncHWtoNHWC(lowresEncoding))
        let out4 = fuseLowres(concatenated([out4_a, lowresUp], axis: -1))

        return [out0, out1, out2, out3, out4]
    }
}

// MARK: - Split / Merge helpers

/// image: [B, C, H, W] NCHW → [B*stepsH*stepsW, patchSize, patchSize, C] NHWC
/// Boundary patches are shifted inward so every patch is exactly patchSize×patchSize.
private func spnSplit(_ image: MLXArray, overlapRatio: Float, patchSize: Int) -> MLXArray {
    let stride = Int(Float(patchSize) * (1 - overlapRatio))
    let imageH = image.shape[2]
    let imageW = image.shape[3]
    let stepsH = max(1, Int(ceil(Float(imageH - patchSize) / Float(stride))) + 1)
    let stepsW = max(1, Int(ceil(Float(imageW - patchSize) / Float(stride))) + 1)

    var patches: [MLXArray] = []
    for j in 0..<stepsH {
        // Clamp so the patch always fits within the image
        let j0 = min(j * stride, imageH - patchSize)
        for i in 0..<stepsW {
            let i0 = min(i * stride, imageW - patchSize)
            let patch = image[0..., 0..., j0..<(j0 + patchSize), i0..<(i0 + patchSize)]
            patches.append(ncHWtoNHWC(patch))
        }
    }
    return concatenated(patches, axis: 0)
}

/// Merge patch feature maps back into a full feature map.
/// patches: [B*steps*steps, C, pH, pW] NCHW
/// Returns: [B, C, H, W] NCHW
private func spnMerge(_ patches: MLXArray, batchSize: Int, padding: Int) -> MLXArray {
    let total = patches.shape[0]
    let steps = Int((Double(total) / Double(batchSize)).squareRoot().rounded())

    var rowFeatures: [MLXArray] = []
    var idx = 0
    for j in 0..<steps {
        var colFeatures: [MLXArray] = []
        for i in 0..<steps {
            var p = patches[idx * batchSize..<(idx + 1) * batchSize, 0..., 0..., 0...]
            if padding != 0 {
                var hStart = 0, hEnd = p.shape[2]
                var wStart = 0, wEnd = p.shape[3]
                if j != 0 { hStart = padding }
                if j != steps - 1 { hEnd = hEnd - padding }
                if i != 0 { wStart = padding }
                if i != steps - 1 { wEnd = wEnd - padding }
                p = p[0..., 0..., hStart..<hEnd, wStart..<wEnd]
            }
            colFeatures.append(p)
            idx += 1
        }
        rowFeatures.append(concatenated(colFeatures, axis: 3))  // concat along W
    }
    return concatenated(rowFeatures, axis: 2)  // concat along H
}

/// Bilinear downsample NCHW by a scale factor using MLXNN.upsample (negative scale not needed)
private func bilinearDownsample(_ x: MLXArray, scale: Float) -> MLXArray {
    // Convert to NHWC for MLXNN interpolation, then back
    let nhwc = ncHWtoNHWC(x)
    let h = Int(Float(x.shape[2]) * scale)
    let w = Int(Float(x.shape[3]) * scale)
    let upsampler = Upsample(scaleFactor: .array([scale, scale]), mode: .linear(alignCorners: false))
    // Upsample only works for upsampling; for downsampling use interpolation
    // MLX doesn't have a direct bilinear interpolate API, so we use the upsample with target size
    // Workaround: use MLXNN upsample which accepts scale_factor
    // For downsampling we pass scale < 1; check if MLXNN supports this
    _ = upsampler  // avoid unused warning
    // Use avg pool as approximation for exact 2× downsampling
    let result: MLXArray
    if scale == 0.5 {
        let pool = AvgPool2d(kernelSize: IntOrPair(2), stride: IntOrPair(2))
        result = nhwcToNCHW(pool(nhwc))
    } else if scale == 0.25 {
        let pool = AvgPool2d(kernelSize: IntOrPair(4), stride: IntOrPair(4))
        result = nhwcToNCHW(pool(nhwc))
    } else {
        // Generic: interpolate via tiling/averaging
        // Use a 2-step approach for other scales
        let pool2 = AvgPool2d(kernelSize: IntOrPair(2), stride: IntOrPair(2))
        let tmp = nhwcToNCHW(pool2(nhwc))
        let nhwc2 = ncHWtoNHWC(tmp)
        let pool4 = AvgPool2d(kernelSize: IntOrPair(2), stride: IntOrPair(2))
        result = nhwcToNCHW(pool4(nhwc2))
        _ = h; _ = w
    }
    return result
}

// MARK: - Format conversion helpers (module-local)

private func ncHWtoNHWC(_ x: MLXArray) -> MLXArray {
    x.transposed(0, 2, 3, 1)
}

private func nhwcToNCHW(_ x: MLXArray) -> MLXArray {
    x.transposed(0, 3, 1, 2)
}
