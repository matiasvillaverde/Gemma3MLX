import Foundation
import OSLog

/// Implementation of the tokenizer for Gemma3 models
class Gemma3Tokenizer {
    private static let logger = Logger(subsystem: "com.example.Gemma3MLX", category: "Tokenizer")

    // Token mapping structures
    private var tokenToId: [String: Int] = [:]
    private var idToToken: [Int: String] = [:]

    // Special tokens
    private var bosToken: Int = 0
    private var eosToken: Int = 0
    private var padToken: Int = 0
    private var unkToken: Int = 0
    private var imageToken: Int = 0
    private var boiToken: String = ""
    private var eoiToken: String = ""

    /// Initialize the tokenizer from the model directory
    /// - Parameter directory: Directory containing the tokenizer files
    init(directory: URL) throws {
        Self.logger.debug("Initializing tokenizer from \(directory.lastPathComponent, privacy: .public)")

        // Load vocabulary file
        let vocabURL = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            Self.logger.error("Tokenizer file not found at \(vocabURL.path, privacy: .public)")
            throw TokenizerError.fileNotFound(path: vocabURL.path)
        }

        let vocabData = try Data(contentsOf: vocabURL)

        // Parse the tokenizer JSON
        guard let json = try JSONSerialization.jsonObject(with: vocabData) as? [String: Any],
              let vocabulary = json["model"] as? [String: Any],
              let vocab = vocabulary["vocab"] as? [String: Any] else {
            Self.logger.error("Invalid tokenizer format")
            throw TokenizerError.invalidTokenizer
        }

        // Build the token mappings
        for (token, idValue) in vocab {
            if let id = idValue as? Int {
                tokenToId[token] = id
                idToToken[id] = token
            }
        }

        Self.logger.debug("Loaded \(self.tokenToId.count, privacy: .public) tokens")

        // Load special tokens
        try loadSpecialTokens(directory: directory)

        Self.logger.notice("Tokenizer initialized successfully")
    }

    /// Load special tokens from the special_tokens_map.json file
    private func loadSpecialTokens(directory: URL) throws {
        let specialTokensURL = directory.appendingPathComponent("special_tokens_map.json")
        guard FileManager.default.fileExists(atPath: specialTokensURL.path) else {
            Self.logger.error("Special tokens file not found at \(specialTokensURL.path, privacy: .public)")
            throw TokenizerError.fileNotFound(path: specialTokensURL.path)
        }

        let tokensData = try Data(contentsOf: specialTokensURL)
        let tokensJson = try JSONSerialization.jsonObject(with: tokensData) as? [String: Any]

        // Handle special tokens based on the structure in the JSON file
        if let bos = tokensJson?["bos_token"] as? [String: Any],
           let content = bos["content"] as? String,
           let id = tokenToId[content] {
            bosToken = id
            Self.logger.debug("BOS token: \(content, privacy: .public) -> \(id, privacy: .public)")
        } else if let bos = tokensJson?["bos_token"] as? String,
                  let id = tokenToId[bos] {
            bosToken = id
            Self.logger.debug("BOS token: \(bos, privacy: .public) -> \(id, privacy: .public)")
        }

        if let eos = tokensJson?["eos_token"] as? [String: Any],
           let content = eos["content"] as? String,
           let id = tokenToId[content] {
            eosToken = id
            Self.logger.debug("EOS token: \(content, privacy: .public) -> \(id, privacy: .public)")
        } else if let eos = tokensJson?["eos_token"] as? String,
                  let id = tokenToId[eos] {
            eosToken = id
            Self.logger.debug("EOS token: \(eos, privacy: .public) -> \(id, privacy: .public)")
        }

        if let pad = tokensJson?["pad_token"] as? [String: Any],
           let content = pad["content"] as? String,
           let id = tokenToId[content] {
            padToken = id
            Self.logger.debug("PAD token: \(content, privacy: .public) -> \(id, privacy: .public)")
        } else if let pad = tokensJson?["pad_token"] as? String,
                  let id = tokenToId[pad] {
            padToken = id
            Self.logger.debug("PAD token: \(pad, privacy: .public) -> \(id, privacy: .public)")
        }

        if let unk = tokensJson?["unk_token"] as? [String: Any],
           let content = unk["content"] as? String,
           let id = tokenToId[content] {
            unkToken = id
            Self.logger.debug("UNK token: \(content, privacy: .public) -> \(id, privacy: .public)")
        } else if let unk = tokensJson?["unk_token"] as? String,
                  let id = tokenToId[unk] {
            unkToken = id
            Self.logger.debug("UNK token: \(unk, privacy: .public) -> \(id, privacy: .public)")
        }

        // Handle image-related tokens
        if let img = tokensJson?["image_token"] as? String {
            if let id = tokenToId[img] {
                imageToken = id
                Self.logger.debug("Image token: \(img, privacy: .public) -> \(id, privacy: .public)")
            } else {
                // Check if we need to load from added_tokens.json
                try loadImageTokenFromAddedTokens(directory: directory)
            }
        }

        if let boi = tokensJson?["boi_token"] as? String {
            boiToken = boi
            Self.logger.debug("BOI token: \(boi, privacy: .public)")
        }

        if let eoi = tokensJson?["eoi_token"] as? String {
            eoiToken = eoi
            Self.logger.debug("EOI token: \(eoi, privacy: .public)")
        }

        Self.logger.debug("Special tokens loaded")
    }

    /// Load image token from added_tokens.json if needed
    private func loadImageTokenFromAddedTokens(directory: URL) throws {
        let addedTokensURL = directory.appendingPathComponent("added_tokens.json")
        guard FileManager.default.fileExists(atPath: addedTokensURL.path) else {
            Self.logger.debug("No added_tokens.json file found")
            return
        }

        let addedTokensData = try Data(contentsOf: addedTokensURL)
        if let addedTokens = try JSONSerialization.jsonObject(with: addedTokensData) as? [String: Any] {
            for (token, idValue) in addedTokens {
                if let id = idValue as? Int {
                    // Add to our mappings
                    tokenToId[token] = id
                    idToToken[id] = token

                    // If this is the image soft token
                    if token == "<image_soft_token>" || token.contains("image") {
                        imageToken = id
                        Self.logger.debug("Image token from added_tokens: \(token, privacy: .public) -> \(id, privacy: .public)")
                    }
                }
            }
        }
    }

    /// Encode a text prompt into token IDs
    /// - Parameters:
    ///   - text: The text to encode
    ///   - addBos: Whether to add a beginning of sequence token
    ///   - addEos: Whether to add an end of sequence token
    /// - Returns: Array of token IDs
    func encode(_ text: String, addBos: Bool = false, addEos: Bool = false) throws -> [Int] {
        Self.logger.debug("Encoding text of length \(text.count, privacy: .public)")

        // Simple implementation - in a real tokenizer this would use the model's tokenization rules
        // This is just a placeholder for testing
        var tokens: [Int] = []

        // Add BOS token if requested
        if addBos {
            tokens.append(bosToken)
        }

        // Split by whitespace as a simple tokenization strategy
        let words = text.split(separator: " ")
        for word in words {
            if let token = tokenToId[String(word)] {
                tokens.append(token)
            } else {
                // If the word isn't in our vocabulary, use the unknown token
                tokens.append(unkToken)
            }
        }

        // Add EOS token if requested
        if addEos {
            tokens.append(eosToken)
        }

        Self.logger.debug("Encoded to \(tokens.count, privacy: .public) tokens")
        return tokens
    }

    /// Decode token IDs back to text
    /// - Parameter ids: Array of token IDs to decode
    /// - Returns: Decoded text
    func decode(_ ids: [Int]) throws -> String? {
        Self.logger.trace("Decoding \(ids.count, privacy: .public) tokens")

        var text = ""
        for id in ids {
            if let token = idToToken[id] {
                text += token
                // Add space between tokens for simple whitespace tokenization
                // A real implementation would be more sophisticated
                text += " "
            }
        }

        return text.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Error types

    enum TokenizerError: Error {
        case fileNotFound(path: String)
        case invalidTokenizer
        case encodingFailed
        case decodingFailed
    }
}
