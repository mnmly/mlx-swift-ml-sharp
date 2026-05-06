import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    static let ply = UTType(filenameExtension: "ply") ?? .data
}

struct PLYDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.ply, .data] }

    let data: Data

    init(data: Data) { self.data = data }

    init(configuration: ReadConfiguration) throws {
        data = configuration.file.regularFileContents ?? Data()
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
