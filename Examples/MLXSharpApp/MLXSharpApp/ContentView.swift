import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var vm = AppViewModel()
    @State private var showWeightsImporter = false
    @State private var showImageImporter = false
    @State private var showPLYExporter = false

    private static let safetensorsType: UTType = UTType(filenameExtension: "safetensors") ?? .data

    var body: some View {
        VStack(spacing: 0) {
            // ── File pickers ──────────────────────────────────────────────
            VStack(spacing: 8) {
                FileRow(
                    icon: "doc.badge.gearshape",
                    label: vm.weightsURL?.lastPathComponent ?? "No weights file selected"
                ) {
                    showWeightsImporter = true
                } buttonLabel: {
                    Text("Select Weights…")
                }

                FileRow(
                    icon: "photo",
                    label: vm.imageURL?.lastPathComponent ?? "No image selected"
                ) {
                    showImageImporter = true
                } buttonLabel: {
                    Text("Select Image…")
                }
            }
            .padding()

            Divider()

            // ── Image preview ─────────────────────────────────────────────
            Group {
                if let img = vm.previewImage {
                    Image(decorative: img, scale: 1)
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

                Button("Export .ply…") { showPLYExporter = true }
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
        #if os(macOS)
        .frame(minWidth: 520, minHeight: 520)
        #endif
        .background {
            // Isolated host so this fileImporter doesn't collide with the
            // image one — SwiftUI silently drops one when two .fileImporter
                // modifiers sit on the same view chain.
            Color.clear
                .fileImporter(
                    isPresented: $showWeightsImporter,
                    allowedContentTypes: [Self.safetensorsType, .data],
                    allowsMultipleSelection: false
                ) { result in
                    if case let .success(urls) = result, let url = urls.first {
                        vm.setWeightsURL(url)
                    }
                }
        }
        .background {
            Color.clear
                .fileImporter(
                    isPresented: $showImageImporter,
                    allowedContentTypes: [.image],
                    allowsMultipleSelection: false
                ) { result in
                    if case let .success(urls) = result, let url = urls.first {
                        vm.setImageURL(url)
                    }
                }
        }
        .fileExporter(
            isPresented: $showPLYExporter,
            document: vm.plyData.map { PLYDocument(data: $0) },
            contentType: .ply,
            defaultFilename: "gaussians"
        ) { result in
            switch result {
            case .success(let url):
                vm.status = "Saved to \(url.lastPathComponent)."
            case .failure(let error):
                vm.status = "Export failed: \(error.localizedDescription)"
            }
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
