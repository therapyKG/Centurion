import Foundation

// MARK: - Tokenizer Protocol

/// A tokenizer that converts between text and integer token IDs.
public protocol Tokenizer: Sendable {
    var vocabSize: Int { get }
    func encode(_ text: String) -> [Int32]
    func decode(_ ids: [Int32]) -> String
}

// MARK: - CharTokenizer Conformance

extension CharTokenizer: Tokenizer {}

// MARK: - Type-Erased Tokenizer

/// A type-erased `Tokenizer` wrapper that stores `@Sendable` closures,
/// allowing `TextDataset` to hold any tokenizer while remaining `Sendable`.
public struct AnyTokenizer: Tokenizer, Sendable {
    public let vocabSize: Int
    private let _encode: @Sendable (String) -> [Int32]
    private let _decode: @Sendable ([Int32]) -> String

    public init<T: Tokenizer>(_ tokenizer: T) {
        self.vocabSize = tokenizer.vocabSize
        self._encode = { tokenizer.encode($0) }
        self._decode = { tokenizer.decode($0) }
    }

    public func encode(_ text: String) -> [Int32] {
        _encode(text)
    }

    public func decode(_ ids: [Int32]) -> String {
        _decode(ids)
    }
}
