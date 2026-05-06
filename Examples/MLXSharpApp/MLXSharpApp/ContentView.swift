import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var vm = AppViewModel()

    var body: some View {
        VStack(spacing: 0) {
            // ── File pickers ──────────────────────────────────────────────
            VStack(spacing: 8) {
                FileRow(
                    icon: "doc.badge.gearshape",
                    label: vm.weightsURL?.lastPathComponent ?? "No weights file selected"
                ) {
                    pickWeights()
                } buttonLabel: {
                    Text("Select Weights…")
                }

                FileRow(
                    icon: "photo",
                    label: vm.imageURL?.lastPathComponent ?? "No image selected"
                ) {
                    pickImage()
                } buttonLabel: {
                    Text("Select Image…")
                }
            }
            .padding()

            Divider()

            // ── Image preview ─────────────────────────────────────────────
            Group {
                if let img = vm.previewImage {
                    Image(nsImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    ZStack {
                        Color.secondary.opacity(0.07)
                        Label("No image loaded", systemImage: "photo.badge.plus")
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: 320)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .padding(.horizontal)
            .padding(.top, 12)

            // ── Status + progress ─────────────────────────────────────────
            HStack(spacing: 8) {
                if vm.isWorking {
                    ProgressView().controlSize(.small)
                }
                Text(vm.status)
                    .foregroundStyle(.secondary)
                    .font(.caption)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.horizontal)
            .padding(.vertical, 8)

            Divider()

            // ── Action buttons ────────────────────────────────────────────
            HStack(spacing: 12) {
                Button("Process") { vm.process() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!vm.canProcess)

                Button("Export .ply…") { savePLY() }
                    .disabled(!vm.canExport)

                if let n = vm.gaussianCount {
                    Text("\(n) Gaussians")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }
            .padding()
        }
        .frame(minWidth: 520, minHeight: 520)
    }

    // MARK: - Panel helpers

    private func pickWeights() {
        let panel = NSOpenPanel()
        panel.title = "Select weights file"
        panel.allowedContentTypes = [UTType(filenameExtension: "safetensors") ?? .data]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        vm.setWeightsURL(url)
    }

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.title = "Select image"
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        vm.setImageURL(url)
    }

    private func savePLY() {
        guard let data = vm.plyData else { return }
        let panel = NSSavePanel()
        panel.title = "Export Gaussians"
        panel.allowedContentTypes = [UTType(filenameExtension: "ply") ?? .data]
        panel.nameFieldStringValue = "gaussians.ply"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try data.write(to: url)
            vm.status = "Saved to \(url.lastPathComponent)."
        } catch {
            vm.status = "Export failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - FileRow

private struct FileRow<ButtonLabel: View>: View {
    let icon: String
    let label: String
    let action: () -> Void
    @ViewBuilder let buttonLabel: () -> ButtonLabel

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundStyle(.secondary)
                .frame(width: 20)
            Text(label)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button(action: action, label: buttonLabel)
        }
    }
}

#Preview {
    ContentView()
}
