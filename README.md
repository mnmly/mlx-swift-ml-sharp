# mlx-swift-ml-sharp

Swift / [mlx-swift](https://github.com/ml-explore/mlx-swift) port of Apple's
[`ml-sharp`](https://github.com/apple/ml-sharp): a feed-forward predictor that
turns a single RGB image into a 3D Gaussian Splatting scene.

End-to-end on an Apple-Silicon Mac:

```swift
let pipeline = try Sharp.fromPretrained(weightsURL)
let prediction = pipeline(cgImage, focalLengthPixels: fPx)
// prediction.gaussians has meanVectors, singularValues, quaternions, colors, opacities
```

## Status

Numerically equivalent to the upstream Python pipeline on the same input.
On the reference image (`B300081_web-700x525.jpg`, EXIF `FocalLength=24mm`)
every non-rotation field of the exported `.ply` matches Python within
float-precision noise:

| Field | Δmean (Swift − Python) | σ ratio |
|---|---|---|
| x, y, z | +0.004, +0.004, −0.043 | 0.99–1.00 |
| f_dc_0/1/2 (color) | ±0.0005 | 1.00 |
| scale_0/1/2 (log) | −0.017 to −0.019 | 1.00 |
| opacity | +0.053 | 1.01 |

Quaternions are unit-norm but their per-component sign distribution differs
because LAPACK `dgesvd_` and `torch.linalg.svd` pick different sign
conventions for U columns. The covariance ellipsoids reconstruct identically;
renderers are sign-invariant on quaternions.

## Requirements

- macOS 14 or later, Swift 6.0 toolchain.
- Apple Silicon recommended (mlx-swift uses Metal).

## Install

Swift Package Manager:

```swift
.package(url: "https://github.com/<your-fork>/mlx-swift-ml-sharp", from: "0.1.0")
```

Then depend on the `MLXSharp` product:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "MLXSharp", package: "mlx-swift-ml-sharp"),
    ]
)
```

## Quick start

```swift
import CoreGraphics
import MLXSharp

// 1. Load the pretrained model from a .safetensors file (see "Weight conversion").
let pipeline = try Sharp.fromPretrained(weightsURL)

// 2. Compute focal length in pixels at the original image resolution.
//    `SharpCameraIntrinsics` mirrors the EXIF fallback chain in
//    `python/ml-sharp/src/sharp/utils/io.py:load_rgb`: prefer
//    FocalLengthIn35mmFilm, else FocalLength (with a ×8.4 heuristic for
//    sub-10mm values), else `defaultFocalMM` (30 mm by convention).
let fPx = SharpCameraIntrinsics.focalLengthPixels(
    from: cgImageSource,
    imageWidth: cgImage.width,
    imageHeight: cgImage.height
)
// Or, if you already know the 35mm-equivalent focal length:
//   let fPx = SharpCameraIntrinsics.focalLengthPixels(
//       focalMM: 24, imageWidth: cgImage.width, imageHeight: cgImage.height)

// 3. Run inference. Returns gaussians + image metadata for downstream PLY export.
let prediction = pipeline(cgImage, focalLengthPixels: fPx)

print("predicted \(prediction.gaussians.meanVectors.shape[1]) gaussians")
```

`SharpPipeline.callAsFunction` is pure — `CGImage` in, `SharpPrediction` out.
Image preprocessing (read at native resolution, fp32 [0,1], bilinear resample
to the model's internal resolution with `align_corners=True`) is handled
internally and matches PyTorch's `F.interpolate` semantics.

## Public API

The library lives under one module, `MLXSharp`. Top-level surface:

### Loading

| Symbol | Purpose |
|---|---|
| `Sharp.fromPretrained(_:internalResolution:)` | Convenience factory mirroring `mlx-swift-*` libraries. |
| `SharpPipeline.fromPretrained(_:internalResolution:)` | Same, called on the type. |

Default `internalResolution` is `(1536, 1536)` — the resolution the model was
trained at. Don't change this unless you know what you're doing.

### Inference

```swift
public struct SharpPipeline {
    public func callAsFunction(_ image: CGImage, focalLengthPixels: Float) -> SharpPrediction
    public func preprocess(_ image: CGImage) -> MLXArray   // [1, 3, H, W] in [0, 1]
    public func predict(imageNCHW: MLXArray, focalLengthPixels: Float) -> Gaussians3D
}
```

`callAsFunction` is the path most callers want. The two component methods are
exposed for benchmarking and tests.

### Camera intrinsics

```swift
public enum SharpCameraIntrinsics {
    public static func focalLengthPixels(
        focalMM: Float, imageWidth: Int, imageHeight: Int
    ) -> Float
    public static func focalLengthPixels(
        from imageSource: CGImageSource,
        imageWidth: Int, imageHeight: Int,
        defaultFocalMM: Float = 30
    ) -> Float
    public static func focalLengthMM(from imageSource: CGImageSource) -> Float?
}
```

### Output

```swift
public struct SharpPrediction {
    public let gaussians: Gaussians3D
    public let processedWidth: Int       // model's internal resolution
    public let processedHeight: Int
    public let originalWidth: Int        // input CGImage dims
    public let originalHeight: Int
    public let focalLengthPixels: Float  // echoed for downstream unprojection
}

public struct Gaussians3D {
    public let meanVectors: MLXArray     // [B, N, 3]  in normalised device coords
    public let singularValues: MLXArray  // [B, N, 3]  per-axis scales (linear)
    public let quaternions: MLXArray     // [B, N, 4]  (w, x, y, z)
    public let colors: MLXArray          // [B, N, 3]  linearRGB in [0, 1]
    public let opacities: MLXArray       // [B, N]     in [0, 1]
}
```

To export to a Gaussian-Splat-compatible `.ply`, see
`Examples/MLXSharpApp/MLXSharpApp/PLYExporter.swift` — it implements the
unprojection from NDC + the SH₀ colour conversion in ~150 lines.

### Configuration

Default `PredictorParams` in `Sources/MLXSharp/Types.swift` mirrors the
upstream Python `PredictorParams` 1:1. You only need to construct one
manually if you're loading a non-standard checkpoint.

## Weight conversion

The model is published as PyTorch `.pt`. Convert it to MLX-friendly
`.safetensors` once with the shipped script (which uses `uv` to vendor its
own `torch` + `safetensors` deps):

```bash
uv run scripts/convert.py -i path/to/sharp_2572gikvuh.pt
# → writes path/to/sharp_2572gikvuh.safetensors
```

The conversion is purely a tensor-layout transform:

- `Conv2d` weights: `OIHW → OHWI` (MLX uses NHWC inputs).
- `ConvTranspose2d` weights: `IOHW → OHWI`.
- Everything else is copied as-is.

Output is float32 throughout. The `.pt` is loaded with
`torch.load(weights_only=True)`.

The default Apple checkpoint is `sharp_2572gikvuh.pt`, available at
<https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt>. Its use is
governed by `LICENSE_MODEL` (Apple Machine Learning Research Model License) —
read it before redistribution.

## Examples

- **`Examples/MLXSharpApp/`** — a SwiftUI macOS app that picks a `.safetensors`
  weights file and a `CGImage`, runs inference off the main actor, and saves
  a `.ply`. Includes EXIF-based focal-length extraction in `AppViewModel.swift`
  and the SVD-based unprojection / SH₀ colour conversion in `PLYExporter.swift`.
- **`Sources/SharpBench/`** — a CLI benchmark target (`swift run sharp-bench
  --weights path.safetensors --iterations 20 --warmup 3`) that times the
  pipeline stage-by-stage and emits a JSON report.

## License

The Swift port and original Python `ml-sharp` are both distributed under the
**Apple Sample Code License** — see [`LICENSE`](LICENSE).

The model weights themselves are governed by the **Apple Machine Learning
Research Model License** — see [`LICENSE_MODEL`](LICENSE_MODEL). Read it
carefully if you plan to redistribute the weights or use them in a product.

Third-party components (mlx-swift, etc.) are listed in
[`ACKNOWLEDGEMENTS`](ACKNOWLEDGEMENTS).

[`NOTICE`](NOTICE) explains the relationship between this Swift port and the
upstream Apple project and lists the port's contributions.

## Acknowledgements

This port reuses the architecture and trained weights of Apple's `ml-sharp`
project; all algorithmic and modelling credit belongs upstream. The
mlx-swift framework by Apple's MLX team makes the GPU side of this port
possible.
