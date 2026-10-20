import Foundation

/// Every exported function returns `{"value": …}` or `{"error": "…"}` as a C string that Rust
/// releases with `koharu_apple_free`.
struct Response<Value: Encodable>: Encodable {
    let value: Value?
    let error: String?
}

func respond<Value: Encodable>(_ result: Result<Value, Error>) -> UnsafeMutablePointer<CChar>? {
    let response: Response<Value>
    switch result {
    case .success(let value):
        response = Response(value: value, error: nil)
    case .failure(let error):
        response = Response(
            value: nil,
            error: (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        )
    }
    guard let data = try? JSONEncoder().encode(response),
        let json = String(data: data, encoding: .utf8)
    else {
        return nil
    }
    return strdup(json)
}

/// Runs async work on a detached task and blocks until it finishes. Rust calls the exported
/// functions from worker threads, never from the main thread.
func runBlocking<Value>(_ operation: @escaping @Sendable () async throws -> Value) -> Result<
    Value, Error
> {
    let box = ResultBox<Value>()
    let semaphore = DispatchSemaphore(value: 0)
    Task.detached {
        do {
            box.result = .success(try await operation())
        } catch {
            box.result = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    return box.result!
}

private final class ResultBox<Value>: @unchecked Sendable {
    var result: Result<Value, Error>?
}

struct Failure: LocalizedError {
    let errorDescription: String?
}

@_cdecl("koharu_apple_free")
public func koharuAppleFree(_ pointer: UnsafeMutablePointer<CChar>?) {
    free(pointer)
}
