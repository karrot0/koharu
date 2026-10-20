import Foundation
import Vision

struct Recognition: Encodable {
    let text: String
}

/// Recognizes the text in one encoded image with Vision.
@_cdecl("koharu_apple_recognize_text")
public func koharuAppleRecognizeText(
    _ image: UnsafePointer<UInt8>,
    _ length: Int,
    _ languages: UnsafePointer<CChar>
) -> UnsafeMutablePointer<CChar>? {
    let data = Data(bytes: image, count: length)
    let languages = String(cString: languages).split(separator: ",").map(String.init)
    return respond(Result(catching: { try recognize(data, languages: languages) }))
}

private func recognize(_ data: Data, languages: [String]) throws -> Recognition {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    if !languages.isEmpty {
        request.recognitionLanguages = languages
    }
    try VNImageRequestHandler(data: data, options: [:]).perform([request])
    let observations = request.results ?? []

    // Vision's boxes are normalized with the origin at the bottom left. Columns of vertical text
    // read right to left; lines of horizontal text read top to bottom.
    let vertical = observations.contains { $0.boundingBox.height > $0.boundingBox.width * 1.5 }
    let ordered = observations.sorted { left, right in
        vertical
            ? left.boundingBox.midX > right.boundingBox.midX
            : left.boundingBox.midY > right.boundingBox.midY
    }
    let lines = ordered.compactMap { $0.topCandidates(1).first?.string }
    return Recognition(text: lines.joined(separator: "\n"))
}
