@preconcurrency import MLX
@preconcurrency import MLXNN

// MARK: - Layer Scale

final class LayerScale: Module {
    var gamma: MLXArray

    init(dim: Int, initValue: Float = 1e-5) {
        gamma = MLXArray.full([dim], values: MLXArray(initValue), type: Float.self)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

// MARK: - ViT Attention (fused QKV)

final class VitAttention: Module {
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        headDim = dim / numHeads
        scale = 1.0 / Float(headDim).squareRoot()
        super.init()
        qkv = Linear(dim, dim * 3)
        proj = Linear(dim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, N, _) = (x.shape[0], x.shape[1], x.shape[2])
        // QKV projection: [B, N, 3*dim]
        let qkvOut = qkv(x).reshaped([B, N, 3, numHeads, headDim])
        // [3, B, numHeads, N, headDim]
        let qkvT = qkvOut.transposed(2, 0, 3, 1, 4)
        let q = qkvT[0]  // [B, numHeads, N, headDim]
        let k = qkvT[1]
        let v = qkvT[2]

        // Scaled dot-product attention
        let attnWeights = softmax((q * scale).matmul(k.transposed(0, 1, 3, 2)), axis: -1)
        // [B, numHeads, N, headDim]
        let attnOut = attnWeights.matmul(v)
        // [B, N, dim]
        let merged = attnOut.transposed(0, 2, 1, 3).reshaped([B, N, numHeads * headDim])
        return proj(merged)
    }
}

// MARK: - ViT MLP

final class VitMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(dim: Int, hiddenDim: Int) {
        super.init()
        fc1 = Linear(dim, hiddenDim)
        fc2 = Linear(hiddenDim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

// MARK: - ViT Block

final class VitBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var attn: VitAttention
    @ModuleInfo var ls1: LayerScale
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var mlp: VitMLP
    @ModuleInfo var ls2: LayerScale

    init(dim: Int, numHeads: Int, mlpRatio: Float = 4.0, initValues: Float = 1e-5) {
        super.init()
        norm1 = LayerNorm(dimensions: dim)
        attn = VitAttention(dim: dim, numHeads: numHeads)
        ls1 = LayerScale(dim: dim, initValue: initValues)
        norm2 = LayerNorm(dimensions: dim)
        mlp = VitMLP(dim: dim, hiddenDim: Int(Float(dim) * mlpRatio))
        ls2 = LayerScale(dim: dim, initValue: initValues)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x1 = x + ls1(attn(norm1(x)))
        return x1 + ls2(mlp(norm2(x1)))
    }
}

// MARK: - Patch Embedding

final class PatchEmbed: Module {
    @ModuleInfo var proj: Conv2d

    init(inChans: Int = 3, embedDim: Int = 1024, patchSize: Int = 16) {
        super.init()
        // NHWC conv: kernelSize=patchSize, stride=patchSize → non-overlapping patches
        proj = Conv2d(inputChannels: inChans, outputChannels: embedDim,
                      kernelSize: IntOrPair(patchSize), stride: IntOrPair(patchSize), bias: true)
    }

    /// x: [B, H, W, C] NHWC → [B, numPatches, embedDim]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = proj(x)  // [B, pH, pW, embedDim]
        let B = out.shape[0]
        let seqLen = out.shape[1] * out.shape[2]
        return out.reshaped([B, seqLen, out.shape[3]])
    }
}

// MARK: - TimmViT

final class TimmViT: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    var pos_embed: MLXArray
    var cls_token: MLXArray
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var head: Linear
    @ModuleInfo var blocks: [VitBlock]

    let embedDim: Int
    let depth: Int
    let numHeads: Int
    let patchSize: Int
    let imgSize: Int
    // Indices of transformer blocks whose pre-norm outputs are captured.
    let intermediateIds: Set<Int>

    init(
        inChans: Int = 3, embedDim: Int = 1024, depth: Int = 24, numHeads: Int = 16,
        mlpRatio: Float = 4.0, initValues: Float = 1e-5, imgSize: Int = 384,
        patchSize: Int = 16, numClasses: Int = 21841,
        intermediateIds: [Int] = []
    ) {
        self.embedDim = embedDim
        self.depth = depth
        self.numHeads = numHeads
        self.patchSize = patchSize
        self.imgSize = imgSize
        self.intermediateIds = Set(intermediateIds)

        // Stored properties must be initialized before super.init
        let numPatches = (imgSize / patchSize) * (imgSize / patchSize)
        pos_embed = MLXArray.zeros([1, numPatches + 1, embedDim])
        cls_token = MLXArray.zeros([1, 1, embedDim])

        super.init()
        // @ModuleInfo properties must be assigned after super.init
        patchEmbed = PatchEmbed(inChans: inChans, embedDim: embedDim, patchSize: patchSize)
        norm = LayerNorm(dimensions: embedDim)
        head = Linear(embedDim, numClasses)
        blocks = (0..<depth).map { _ in
            VitBlock(dim: embedDim, numHeads: numHeads, mlpRatio: mlpRatio,
                     initValues: initValues)
        }
    }

    /// x: [B, H, W, C] NHWC
    /// Returns: (finalFeatures [B, embedDim, pH, pW], intermediateFeatures dict)
    func callAsFunction(_ x: MLXArray) -> (MLXArray, [Int: MLXArray]) {
        let B = x.shape[0]
        var tokens = patchEmbed(x)  // [B, N, embedDim]
        let clsExpanded = broadcast(cls_token, to: [B, 1, embedDim])
        tokens = concatenated([clsExpanded, tokens], axis: 1)  // [B, N+1, embedDim]
        tokens = tokens + pos_embed

        var intermediates: [Int: MLXArray] = [:]
        for (idx, block) in blocks.enumerated() {
            tokens = block(tokens)
            if intermediateIds.contains(idx) {
                intermediates[idx] = tokens
            }
        }
        tokens = norm(tokens)

        // Reshape to 2D spatial (discard cls token), then NHWC → [B, embedDim, pH, pW]
        let pH = imgSize / patchSize
        let pW = imgSize / patchSize
        let spatial = tokens[0..., 1..., 0...]  // discard cls token
        // [B, pH, pW, embedDim]
        let grid = spatial.reshaped([B, pH, pW, embedDim])
        // [B, embedDim, pH, pW] for consistency with NCHW downstream
        let nchw = grid.transposed(0, 3, 1, 2)
        return (nchw, intermediates)
    }

    /// Reshape intermediate feature [B, N+1, embedDim] → [B, embedDim, pH, pW]
    func reshapeFeature(_ tokens: MLXArray) -> MLXArray {
        let B = tokens.shape[0]
        let pH = imgSize / patchSize
        // Remove cls token
        let spatial = tokens[0..., 1..., 0...]
        let grid = spatial.reshaped([B, pH, pH, embedDim])
        return grid.transposed(0, 3, 1, 2)  // NCHW
    }
}
