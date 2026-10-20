// Runs a compiled Core ML model once and writes each output as contiguous little-endian float32
// for `convert_koharu_layout.py verify`.
//
// Output memory is read only inside MLMultiArray's accessor and through its strides, as Koharu's
// iOS runner does. coremltools' predict hands NumPy the float32 output buffer directly, and reading
// it crashes on current macOS.
//
//     xcrun swiftc -O predict.swift -o predict
//     ./predict MODEL.mlmodelc pixel_values.f32 OUT_DIR

import CoreML
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 4 else {
    fatalError("usage: predict MODEL.mlmodelc INPUT.f32 OUT_DIR")
}

let configuration = MLModelConfiguration()
configuration.computeUnits = .all
let model = try MLModel(contentsOf: URL(fileURLWithPath: arguments[1]), configuration: configuration)

guard let inputShape = model.modelDescription.inputDescriptionsByName["pixel_values"]?
    .multiArrayConstraint?.shape
else {
    fatalError("the model has no pixel_values array input")
}
let input = try MLMultiArray(shape: inputShape, dataType: .float32)
let bytes = try Data(contentsOf: URL(fileURLWithPath: arguments[2]))
precondition(bytes.count == input.count * 4, "input has \(bytes.count) bytes, expected \(input.count * 4)")
input.withUnsafeMutableBytes { pointer, strides in
    let shape = inputShape.map(\.intValue)
    let contiguous = shape.indices.map { shape[($0 + 1)...].reduce(1, *) }
    precondition(strides == contiguous, "input strides \(strides) are not contiguous")
    bytes.copyBytes(to: pointer)
}

let output = try model.prediction(
    from: MLDictionaryFeatureProvider(dictionary: ["pixel_values": input])
)
let directory = URL(fileURLWithPath: arguments[3])
try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

var shapes: [String: [Int]] = [:]
for name in model.modelDescription.outputDescriptionsByName.keys.sorted() {
    guard let array = output.featureValue(for: name)?.multiArrayValue else {
        fatalError("\(name) is not an array output")
    }
    let shape = array.shape.map(\.intValue)
    var values = [Float](repeating: 0, count: array.count)
    array.withUnsafeBytes { raw in
        let strides = array.strides.map(\.intValue)
        for index in values.indices {
            var rest = index
            var offset = 0
            for axis in shape.indices.reversed() {
                offset += rest % shape[axis] * strides[axis]
                rest /= shape[axis]
            }
            switch array.dataType {
            case .float32:
                values[index] = raw.loadUnaligned(fromByteOffset: offset * 4, as: Float.self)
            case .float16:
                values[index] = Float(raw.loadUnaligned(fromByteOffset: offset * 2, as: Float16.self))
            default:
                fatalError("\(name) has unsupported data type \(array.dataType.rawValue)")
            }
        }
    }
    try values.withUnsafeBytes { try Data($0).write(to: directory.appendingPathComponent("\(name).f32")) }
    shapes[name] = shape
}
try JSONSerialization.data(withJSONObject: shapes)
    .write(to: directory.appendingPathComponent("shapes.json"))
