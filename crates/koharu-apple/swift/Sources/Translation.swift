import Foundation
import NaturalLanguage
import Translation

struct TranslationJob: Decodable {
    let texts: [String]
    let source: String?
    let target: String
}

struct Translated: Encodable {
    let texts: [String]
}

/// Translates a batch of texts on device with the Translation framework.
@_cdecl("koharu_apple_translate")
public func koharuAppleTranslate(_ request: UnsafePointer<CChar>) -> UnsafeMutablePointer<CChar>? {
    let json = Data(String(cString: request).utf8)
    let result = Result(catching: { try JSONDecoder().decode(TranslationJob.self, from: json) })
        .flatMap { job in runBlocking { try await translate(job) } }
    return respond(result)
}

private func translate(_ job: TranslationJob) async throws -> Translated {
    let target = Locale.Language(identifier: job.target)
    let source = Locale.Language(identifier: job.source ?? dominantLanguage(job.texts))
    let pair = "\(source.minimalIdentifier) to \(target.minimalIdentifier)"

    switch await LanguageAvailability().status(from: source, to: target) {
    case .installed:
        break
    case .supported:
        throw Failure(
            errorDescription:
                "Download the languages for \(pair) in Settings > Apps > Translate to translate on this device."
        )
    case .unsupported:
        throw Failure(errorDescription: "Apple Translation does not support \(pair).")
    @unknown default:
        break
    }

    let session = TranslationSession(installedSource: source, target: target)
    let requests = job.texts.enumerated().map { index, text in
        TranslationSession.Request(sourceText: text, clientIdentifier: String(index))
    }
    var texts = Array(repeating: "", count: job.texts.count)
    for response in try await session.translations(from: requests) {
        if let identifier = response.clientIdentifier,
            let index = Int(identifier),
            texts.indices.contains(index)
        {
            texts[index] = response.targetText
        }
    }
    return Translated(texts: texts)
}

private func dominantLanguage(_ texts: [String]) -> String {
    let recognizer = NLLanguageRecognizer()
    recognizer.processString(texts.joined(separator: "\n"))
    return recognizer.dominantLanguage?.rawValue ?? "ja"
}
