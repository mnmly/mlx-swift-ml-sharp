import Accelerate
import Foundation
@preconcurrency import MLX
@preconcurrency import MLXSharp

enum PLYExporter {
    // DC spherical harmonics constant: sqrt(1/(4*pi))
    private static let C0: Float = 0.28209479177387814

    nonisolated static func export(_ prediction: SharpPrediction) -> Data {
        export(
            prediction.gaussians,
            focalLengthPixels: prediction.focalLengthPixels,
            originalWidth: prediction.originalWidth,
            originalHeight: prediction.originalHeight
        )
    }

    nonisolated static func export(
        _ gaussians: Gaussians3D,
        focalLengthPixels: Float,
        originalWidth: Int,
        originalHeight: Int
    ) -> Data {
        let pos  = gaussians.meanVectors[0].asArray(Float.self)    // [N*3]
        let sv   = gaussians.singularValues[0].asArray(Float.self)  // [N*3]
        let quat = gaussians.quaternions[0].asArray(Float.self)     // [N*4]
        let col  = gaussians.colors[0].asArray(Float.self)          // [N*3]
        let opa  = gaussians.opacities[0].asArray(Float.self)       // [N]
        let n    = opa.count

        let W = Float(originalWidth)
        let H = Float(originalHeight)
        // Unprojection scale factors: world = ndc * [W/(2f), H/(2f), 1]
        let sx = W / (2 * focalLengthPixels)
        let sy = H / (2 * focalLengthPixels)

        // Vertex layout: x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
        let stride = 14
        var verts = [Float](repeating: 0, count: n * stride)

        for i in 0..<n {
            let b = i * stride

            // Unproject position: x and y scale by sx/sy, z unchanged
            verts[b + 0] = pos[i * 3 + 0] * sx
            verts[b + 1] = pos[i * 3 + 1] * sy
            verts[b + 2] = pos[i * 3 + 2]

            // Color: linearRGB → sRGB → SH DC coefficient
            verts[b + 3] = (linearToSRGB(col[i * 3 + 0]) - 0.5) / C0
            verts[b + 4] = (linearToSRGB(col[i * 3 + 1]) - 0.5) / C0
            verts[b + 5] = (linearToSRGB(col[i * 3 + 2]) - 0.5) / C0

            // Opacity logit
            let o = max(1e-6, min(1 - 1e-6, opa[i]))
            verts[b + 6] = log(o / (1 - o))

            // Unproject covariance: transform N = A @ R @ diag(sv), SVD → new rotation + scales
            let qw = quat[i * 4 + 0], qx = quat[i * 4 + 1]
            let qy = quat[i * 4 + 2], qz = quat[i * 4 + 3]
            let sv0 = sv[i * 3 + 0], sv1 = sv[i * 3 + 1], sv2 = sv[i * 3 + 2]

            let R = quatToRotMat(qw: qw, qx: qx, qy: qy, qz: qz)

            // N[j,k] = scaleRow[j] * R[j,k] * sv[k]
            var N = [Float](repeating: 0, count: 9)
            N[0*3+0] = sx * R[0] * sv0;  N[0*3+1] = sx * R[1] * sv1;  N[0*3+2] = sx * R[2] * sv2
            N[1*3+0] = sy * R[3] * sv0;  N[1*3+1] = sy * R[4] * sv1;  N[1*3+2] = sy * R[5] * sv2
            N[2*3+0] =      R[6] * sv0;  N[2*3+1] =      R[7] * sv1;  N[2*3+2] =      R[8] * sv2

            let (U, newSV) = svd3x3(N)

            verts[b + 7]  = log(max(1e-10, newSV[0]))
            verts[b + 8]  = log(max(1e-10, newSV[1]))
            verts[b + 9]  = log(max(1e-10, newSV[2]))

            let (nqw, nqx, nqy, nqz) = rotMatToQuat(U)
            verts[b + 10] = nqw
            verts[b + 11] = nqx
            verts[b + 12] = nqy
            verts[b + 13] = nqz
        }

        // ── Header ───────────────────────────────────────────────────────────
        let header = """
            ply\n\
            format binary_little_endian 1.0\n\
            element vertex \(n)\n\
            property float x\n\
            property float y\n\
            property float z\n\
            property float f_dc_0\n\
            property float f_dc_1\n\
            property float f_dc_2\n\
            property float opacity\n\
            property float scale_0\n\
            property float scale_1\n\
            property float scale_2\n\
            property float rot_0\n\
            property float rot_1\n\
            property float rot_2\n\
            property float rot_3\n\
            element extrinsic 16\n\
            property float extrinsic\n\
            element intrinsic 9\n\
            property float intrinsic\n\
            element image_size 2\n\
            property uint image_size\n\
            element frame 2\n\
            property int frame\n\
            element disparity 2\n\
            property float disparity\n\
            element color_space 1\n\
            property uchar color_space\n\
            element version 3\n\
            property uchar version\n\
            end_header\n
            """
        guard let headerData = header.data(using: .ascii) else {
            fatalError("PLYExporter: failed to encode header")
        }

        // ── Vertex binary data ────────────────────────────────────────────────
        let vertexData = verts.withUnsafeBytes { Data($0) }

        // ── Extrinsic: identity 4×4 ───────────────────────────────────────────
        let eye4: [Float] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        let extrinsicData = eye4.withUnsafeBytes { Data($0) }

        // ── Intrinsic: 3×3 (row-major) ────────────────────────────────────────
        let cx = W * 0.5, cy = H * 0.5
        let intr: [Float] = [focalLengthPixels, 0, cx, 0, focalLengthPixels, cy, 0, 0, 1]
        let intrinsicData = intr.withUnsafeBytes { Data($0) }

        // ── Image size: [width, height] as uint32 ────────────────────────────
        var imgSize: [UInt32] = [UInt32(originalWidth), UInt32(originalHeight)]
        let imageSizeData = imgSize.withUnsafeBytes { Data($0) }

        // ── Frame: [1, N] as int32 ────────────────────────────────────────────
        var frameArr: [Int32] = [1, Int32(n)]
        let frameData = frameArr.withUnsafeBytes { Data($0) }

        // ── Disparity: 10th and 90th percentile of 1/z ───────────────────────
        var disparities = (0..<n).map { 1.0 / pos[$0 * 3 + 2] as Float }
        disparities.sort()
        let idx10 = max(0, min(n - 1, Int(Float(n - 1) * 0.1)))
        let idx90 = max(0, min(n - 1, Int(Float(n - 1) * 0.9)))
        var dispArr: [Float] = [disparities[idx10], disparities[idx90]]
        let disparityData = dispArr.withUnsafeBytes { Data($0) }

        // ── Color space: 0 = sRGB ────────────────────────────────────────────
        var csArr: [UInt8] = [0]
        let colorSpaceData = csArr.withUnsafeBytes { Data($0) }

        // ── Version: [1, 5, 0] ───────────────────────────────────────────────
        var verArr: [UInt8] = [1, 5, 0]
        let versionData = verArr.withUnsafeBytes { Data($0) }

        return headerData + vertexData + extrinsicData + intrinsicData
             + imageSizeData + frameData + disparityData + colorSpaceData + versionData
    }

    // MARK: - Color

    private static func linearToSRGB(_ c: Float) -> Float {
        if c <= 0.0031308 {
            return 12.92 * c
        } else {
            return 1.055 * pow(max(0, c), 1.0 / 2.4) - 0.055
        }
    }

    // MARK: - Rotation helpers (row-major 3×3 stored as [Float] of length 9)

    // Quaternion (w, x, y, z) → row-major rotation matrix
    private static func quatToRotMat(qw: Float, qx: Float, qy: Float, qz: Float) -> [Float] {
        let len = (qw*qw + qx*qx + qy*qy + qz*qz).squareRoot() + 1e-10
        let w = qw/len, x = qx/len, y = qy/len, z = qz/len
        return [
            1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
            2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
            2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)
        ]
    }

    // Row-major rotation matrix → quaternion (w, x, y, z) using Shepperd's method
    private static func rotMatToQuat(_ R: [Float]) -> (Float, Float, Float, Float) {
        let trace = R[0] + R[4] + R[8]
        var qw, qx, qy, qz: Float

        if trace > 0 {
            let s = 0.5 / (trace + 1).squareRoot()
            qw = 0.25 / s
            qx = (R[7] - R[5]) * s
            qy = (R[2] - R[6]) * s
            qz = (R[3] - R[1]) * s
        } else if R[0] > R[4] && R[0] > R[8] {
            let s = 2 * (1 + R[0] - R[4] - R[8]).squareRoot()
            qw = (R[7] - R[5]) / s
            qx = 0.25 * s
            qy = (R[1] + R[3]) / s
            qz = (R[2] + R[6]) / s
        } else if R[4] > R[8] {
            let s = 2 * (1 + R[4] - R[0] - R[8]).squareRoot()
            qw = (R[2] - R[6]) / s
            qx = (R[1] + R[3]) / s
            qy = 0.25 * s
            qz = (R[5] + R[7]) / s
        } else {
            let s = 2 * (1 + R[8] - R[0] - R[4]).squareRoot()
            qw = (R[3] - R[1]) / s
            qx = (R[2] + R[6]) / s
            qy = (R[5] + R[7]) / s
            qz = 0.25 * s
        }

        let len = (qw*qw + qx*qx + qy*qy + qz*qz).squareRoot() + 1e-10
        return (qw/len, qx/len, qy/len, qz/len)
    }

    private static func det3x3(_ R: [Float]) -> Float {
        R[0] * (R[4]*R[8] - R[5]*R[7])
      - R[1] * (R[3]*R[8] - R[5]*R[6])
      + R[2] * (R[3]*R[7] - R[4]*R[6])
    }

    // MARK: - 3×3 SVD via LAPACK dgesvd_ (fp64)
    // Mirrors Python's `covariance_matrices.to(torch.float64)` before SVD, so the
    // smallest singular value doesn't pick up fp32 rounding noise.
    // A_rm: row-major fp32 input. Returns U (row-major proper rotation) and s (singular values ≥ 0).
    private static func svd3x3(_ A_rm: [Float]) -> (U: [Float], s: [Float]) {
        // Row-major fp32 → column-major fp64 for LAPACK.
        var a: [Double] = [
            Double(A_rm[0]), Double(A_rm[3]), Double(A_rm[6]),
            Double(A_rm[1]), Double(A_rm[4]), Double(A_rm[7]),
            Double(A_rm[2]), Double(A_rm[5]), Double(A_rm[8])
        ]
        var jobu:  Int8 = Int8(UInt8(ascii: "A"))
        var jobvt: Int8 = Int8(UInt8(ascii: "N"))
        var m = Int32(3), n = Int32(3)
        var lda = Int32(3)
        var sD = [Double](repeating: 0, count: 3)
        var uD = [Double](repeating: 0, count: 9)
        var ldu = Int32(3)
        var vt = [Double](repeating: 0, count: 1)
        var ldvt = Int32(1)
        var lwork = Int32(32)
        var work = [Double](repeating: 0, count: 32)
        var info = Int32(0)

        dgesvd_(&jobu, &jobvt, &m, &n, &a, &lda, &sD, &uD, &ldu, &vt, &ldvt, &work, &lwork, &info)

        // Column-major fp64 U → row-major fp32.
        var U: [Float] = [
            Float(uD[0]), Float(uD[3]), Float(uD[6]),
            Float(uD[1]), Float(uD[4]), Float(uD[7]),
            Float(uD[2]), Float(uD[5]), Float(uD[8])
        ]
        if det3x3(U) < 0 {
            U[2] = -U[2]; U[5] = -U[5]; U[8] = -U[8]
        }
        return (U, [Float(sD[0]), Float(sD[1]), Float(sD[2])])
    }
}
