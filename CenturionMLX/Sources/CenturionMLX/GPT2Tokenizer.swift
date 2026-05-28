import Foundation

// MARK: - GPT-2 BPE Tokenizer

/// A byte-level BPE tokenizer matching OpenAI's GPT-2 tokenizer.
/// Loads `encoder.json` and `vocab.bpe` from the bundle resources.
public struct GPT2Tokenizer: Tokenizer, Sendable {
    public let vocabSize: Int

    /// Token string → ID (e.g. "hello" → 31373)
    private let encoder: [String: Int32]
    /// ID → token string
    private let decoder: [Int32: String]
    /// BPE merge pairs with their rank (lower = higher priority)
    private let bpeRanks: [StringPair: Int]
    /// Byte value → unicode character used by GPT-2
    private let byteEncoder: [UInt8: Character]
    /// Unicode character → byte value (inverse of byteEncoder)
    private let byteDecoder: [Character: UInt8]
    /// Pre-tokenization regex
    private let pattern: Regex<Substring>

    // MARK: - Init

    public init() throws {
        guard let encoderURL = Bundle.module.url(forResource: "encoder", withExtension: "json", subdirectory: "Resources"),
              let bpeURL = Bundle.module.url(forResource: "vocab", withExtension: "bpe", subdirectory: "Resources")
        else {
            throw TokenizerError.missingResource
        }

        let encoderData = try Data(contentsOf: encoderURL)
        let rawEncoder = try JSONDecoder().decode([String: Int].self, from: encoderData)

        var enc: [String: Int32] = [:]
        var dec: [Int32: String] = [:]
        for (key, value) in rawEncoder {
            let id = Int32(value)
            enc[key] = id
            dec[id] = key
        }
        self.encoder = enc
        self.decoder = dec
        self.vocabSize = rawEncoder.count

        // Build byte ↔ unicode mapping
        let (be, bd) = Self.buildByteMapping()
        self.byteEncoder = be
        self.byteDecoder = bd

        // Parse BPE merges
        let bpeText = try String(contentsOf: bpeURL, encoding: .utf8)
        let lines = bpeText.split(separator: "\n")
        var ranks: [StringPair: Int] = [:]
        // Skip first line (#version: 0.2) and any trailing empty lines
        for (i, line) in lines.dropFirst().enumerated() {
            let parts = line.split(separator: " ")
            guard parts.count == 2 else { continue }
            ranks[StringPair(String(parts[0]), String(parts[1]))] = i
        }
        self.bpeRanks = ranks

        // GPT-2 pre-tokenization pattern
        self.pattern = try Regex(#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#)
    }

    // MARK: - Encode

    public func encode(_ text: String) -> [Int32] {
        var tokens: [Int32] = []
        let matches = text.matches(of: pattern)
        for match in matches {
            let word = String(match.output)
            // Convert each byte of the UTF-8 representation to the GPT-2 unicode character
            let bpeChars = word.utf8.map { byteEncoder[$0]! }
            let bpeString = String(bpeChars)
            // Apply BPE merges
            let merged = bpe(bpeString)
            for token in merged.split(separator: " ") {
                if let id = encoder[String(token)] {
                    tokens.append(id)
                }
            }
        }
        return tokens
    }

    // MARK: - Decode

    public func decode(_ ids: [Int32]) -> String {
        let tokenStrings = ids.compactMap { decoder[$0] }
        let joined = tokenStrings.joined()
        // Convert GPT-2 unicode characters back to bytes
        let bytes = joined.compactMap { byteDecoder[$0] }
        return String(bytes: bytes, encoding: .utf8) ?? String(tokenStrings.joined())
    }

    // MARK: - BPE Algorithm

    /// Apply BPE merges to a word (already converted to GPT-2 unicode chars).
    /// Returns space-separated BPE tokens.
    private func bpe(_ token: String) -> String {
        guard token.count >= 2 else { return token }

        var word = token.map { String($0) }

        while true {
            // Find the pair with the lowest rank
            var bestPair: StringPair?
            var bestRank = Int.max

            for i in 0..<(word.count - 1) {
                let pair = StringPair(word[i], word[i + 1])
                if let rank = bpeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = pair
                }
            }

            guard let pair = bestPair else { break }

            // Merge all occurrences of this pair
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == pair.first && word[i + 1] == pair.second {
                    newWord.append(pair.first + pair.second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord

            if word.count == 1 { break }
        }

        return word.joined(separator: " ")
    }

    // MARK: - Byte ↔ Unicode Mapping

    /// Build the GPT-2 byte-to-unicode mapping table.
    /// Printable ASCII and Latin-1 supplement bytes map to themselves as unicode;
    /// the remaining 68 bytes map to codepoints 256–323.
    private static func buildByteMapping() -> ([UInt8: Character], [Character: UInt8]) {
        var byteToUnicode: [UInt8: Character] = [:]

        // Ranges that map directly: 33–126, 161–172, 174–255
        var directBytes: [UInt8] = []
        directBytes.append(contentsOf: Array(33...126).map { UInt8($0) })
        directBytes.append(contentsOf: Array(161...172).map { UInt8($0) })
        directBytes.append(contentsOf: Array(174...255).map { UInt8($0) })

        for b in directBytes {
            byteToUnicode[b] = Character(Unicode.Scalar(UInt32(b))!)
        }

        // Remaining bytes get mapped to 256+
        var nextCodepoint: UInt32 = 256
        for b in 0...255 {
            let byte = UInt8(b)
            if byteToUnicode[byte] == nil {
                byteToUnicode[byte] = Character(Unicode.Scalar(nextCodepoint)!)
                nextCodepoint += 1
            }
        }

        // Build inverse mapping
        var unicodeToByte: [Character: UInt8] = [:]
        for (b, c) in byteToUnicode {
            unicodeToByte[c] = b
        }

        return (byteToUnicode, unicodeToByte)
    }
}

// MARK: - Supporting Types

/// A hashable pair of strings, used as dictionary key for BPE merge ranks.
struct StringPair: Hashable, Sendable {
    let first: String
    let second: String

    init(_ first: String, _ second: String) {
        self.first = first
        self.second = second
    }
}

/// Errors that can occur during tokenizer initialization.
public enum TokenizerError: Error, LocalizedError {
    case missingResource

    public var errorDescription: String? {
        switch self {
        case .missingResource:
            return "Could not find encoder.json or vocab.bpe in bundle resources."
        }
    }
}
