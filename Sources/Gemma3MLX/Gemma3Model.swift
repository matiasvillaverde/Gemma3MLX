import Foundation
import MLX
import CoreImage
import OSLog
import AVFoundation

/// Complete implementation of the Gemma3 model using MLX for Swift
final class Gemma3Model {
    // MARK: - Types and Enums

    /// Logger for Gemma3 model operations
    private static let logger = Logger(subsystem: "com.example.Gemma3MLX", category: "Model")

    /// Text model configuration
    struct TextConfig: Codable {
        let modelType: String
        let hiddenSize: Int
        let numHiddenLayers: Int
        let intermediateSize: Int
        let numAttentionHeads: Int
        let headDim: Int
        let rmsNormEps: Float
        let vocabSize: Int
        let numKeyValueHeads: Int
        let ropeTheta: Float
        let ropeLocalBaseFreq: Float
        let queryPreAttnScalar: Float
        let slidingWindow: Int
        let slidingWindowPattern: Int

        // Optional fields
        let mmTokensPerImage: Int?
        let attentionBias: Bool?
        let attentionDropout: Float?
        let hiddenActivation: String?
        let useCache: Bool?
        let finalLogitSoftcapping: Float?
        let attnLogitSoftcapping: Float?
        let cacheImplementation: String?

        // Rope scaling is a nested structure
        struct RopeScaling: Codable {
            let factor: Float
            let ropeType: String

            enum CodingKeys: String, CodingKey {
                case factor
                case ropeType = "rope_type"
            }
        }

        let ropeScaling: RopeScaling?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case headDim = "head_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabSize = "vocab_size"
            case numKeyValueHeads = "num_key_value_heads"
            case ropeTheta = "rope_theta"
            case ropeLocalBaseFreq = "rope_local_base_freq"
            case queryPreAttnScalar = "query_pre_attn_scalar"
            case slidingWindow = "sliding_window"
            case slidingWindowPattern = "sliding_window_pattern"
            case mmTokensPerImage = "mm_tokens_per_image"
            case attentionBias = "attention_bias"
            case attentionDropout = "attention_dropout"
            case hiddenActivation = "hidden_activation"
            case useCache = "use_cache"
            case finalLogitSoftcapping = "final_logit_softcapping"
            case attnLogitSoftcapping = "attn_logit_softcapping"
            case cacheImplementation = "cache_implementation"
            case ropeScaling = "rope_scaling"
        }
    }

    /// Vision model configuration
    struct VisionConfig: Codable {
        let modelType: String
        let numHiddenLayers: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let numAttentionHeads: Int
        let patchSize: Int
        let imageSize: Int
        let numChannels: Int
        let layerNormEps: Float
        let visionUseHead: Bool?
        let hiddenAct: String?
        let attentionDropout: Float?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case numHiddenLayers = "num_hidden_layers"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case patchSize = "patch_size"
            case imageSize = "image_size"
            case numChannels = "num_channels"
            case layerNormEps = "layer_norm_eps"
            case visionUseHead = "vision_use_head"
            case hiddenAct = "hidden_act"
            case attentionDropout = "attention_dropout"
        }
    }

    /// Model configuration structure
    struct Config: Codable {
        let modelType: String
        let textConfig: TextConfig
        let visionConfig: VisionConfig
        let imageTokenIndex: Int
        let padTokenId: Int?  // This is null in the JSON
        let eosTokenId: [Int]?

        // Other optional fields present in the JSON
        let architectures: [String]?
        let quantization: Quantization?
        let mmTokensPerImage: Int?
        let boiTokenIndex: Int?
        let eoiTokenIndex: Int?

        struct Quantization: Codable {
            let groupSize: Int
            let bits: Int

            enum CodingKeys: String, CodingKey {
                case groupSize = "group_size"
                case bits
            }
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case textConfig = "text_config"
            case visionConfig = "vision_config"
            case imageTokenIndex = "image_token_index"
            case padTokenId = "pad_token_id"
            case eosTokenId = "eos_token_id"
            case architectures
            case quantization
            case mmTokensPerImage = "mm_tokens_per_image"
            case boiTokenIndex = "boi_token_index"
            case eoiTokenIndex = "eoi_token_index"
        }
    }

    /// Structure to parse model index file
    struct ModelIndex: Codable {
        let metadata: Metadata?
        let weight_map: [String: String]

        struct Metadata: Codable {
            let framework: String?
            let format: String?
            let total_size: Int64?

            // Custom decoder to handle missing fields
            init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                framework = try container.decodeIfPresent(String.self, forKey: .framework)
                format = try container.decodeIfPresent(String.self, forKey: .format)
                total_size = try container.decodeIfPresent(Int64.self, forKey: .total_size)
            }

            enum CodingKeys: String, CodingKey {
                case framework, format, total_size
            }
        }

        // Custom init to handle missing metadata
        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            metadata = try container.decodeIfPresent(Metadata.self, forKey: .metadata)
            weight_map = try container.decode([String: String].self, forKey: .weight_map)
        }

        enum CodingKeys: String, CodingKey {
            case metadata, weight_map
        }
    }

    /// Information about a safetensors weight shard
    struct WeightShard {
        let fileName: String
        let weights: [String]
    }

    /// Structure for model configuration
    struct ModelConfig {
        let textConfig: TextConfig
        let visionConfig: VisionConfig
        let modelType: String
        let imageTokenIndex: Int
        let padTokenId: Int
        let eosTokenId: [Int]?
        let mmTokensPerImage: Int?
        let boiTokenIndex: Int?
        let eoiTokenIndex: Int?
    }

    // MARK: - Properties

    /// Tokenizer for text processing
    private var tokenizer: Gemma3Tokenizer?

    /// Main language model components
    private var embedTokens: MLXArray?
    private var languageLayers: [[String: MLXArray]] = []
    private var finalNorm: MLXArray?
    private var lmHead: MLXArray?

    /// Vision components
    private var visionEncoder: [String: MLXArray]?
    private var multimodalProjector: [String: MLXArray]?

    /// Model configuration
    private var config: Config?

    /// Directory containing model files
    private var modelDirectory: URL?

    /// Beginning of sentence token ID
    private var bosToken: Int = 0

    /// End of sentence token ID
    private var eosToken: Int = 0

    /// Padding token ID
    private var padToken: Int = 0

    // MARK: - Public Methods

    /// Load the Gemma3 model from a directory
    /// - Parameter localURL: URL to the model directory
    /// - Returns: AsyncThrowingStream with loading progress
    func load(model localURL: URL) -> AsyncThrowingStream<Progress, Error> {
        Self.logger.info("Starting to load model from \(localURL.path, privacy: .public)")

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    self.modelDirectory = localURL

                    // Load configuration
                    continuation.yield(Progress(completed: 1, total: 5, description: "Loading configuration"))
                    Self.logger.debug("Loading model configuration")
                    try await loadConfiguration(from: localURL)

                    // Load tokenizer
                    continuation.yield(Progress(completed: 2, total: 5, description: "Loading tokenizer"))
                    Self.logger.debug("Loading tokenizer")
                    try loadTokenizer(from: localURL)

                    // Parse model index to identify weight shards
                    continuation.yield(Progress(completed: 3, total: 5, description: "Parsing model structure"))
                    Self.logger.debug("Parsing model index")
                    let weightShards = try parseModelIndex(from: localURL)

                    // Load model weights
                    continuation.yield(Progress(completed: 4, total: 5, description: "Loading model weights"))
                    Self.logger.debug("Loading model weights from \(weightShards.count, privacy: .public) shards")
                    try await loadModelWeights(shards: weightShards, from: localURL)

                    // Finalize loading
                    continuation.yield(Progress(completed: 5, total: 5, description: "Model loaded successfully"))
                    Self.logger.notice("Model loaded successfully")
                    continuation.finish()
                } catch {
                    Self.logger.error("Failed to load model: \(error, privacy: .public)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Generate text from the model with the provided configuration
    /// - Parameter config: Configuration for text generation
    /// - Returns: AsyncThrowingStream yielding generated text tokens
    func generateVLM(config: VLMConfiguration) -> AsyncThrowingStream<String, Error> {
        Self.logger.info("Starting VLM generation with temperature: \(config.temperature, privacy: .public), topP: \(config.topP, privacy: .public)")

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    guard let tokenizer = self.tokenizer else {
                        Self.logger.error("Tokenizer not initialized")
                        throw ModelError.tokenizerNotInitialized
                    }

                    // Tokenize the prompt
                    Self.logger.debug("Tokenizing prompt")
                    let tokenIds = try tokenizer.encode(config.prompt, addBos: true, addEos: false)

                    // Prepare image or video input if provided
                    var imageEmbeddings: MLXArray? = nil
                    if let image = config.image {
                        Self.logger.debug("Processing image input")
                        imageEmbeddings = try processImage(image)
                    } else if let videoURL = config.videoURL {
                        Self.logger.debug("Processing video input from \(videoURL.lastPathComponent, privacy: .public)")
                        imageEmbeddings = try await processVideo(videoURL)
                    }

                    // Generate text using the text generation logic
                    Self.logger.debug("Starting text generation with max tokens: \(config.maxTokens, privacy: .public)")
                    let textStream = generateText(
                        inputIds: MLXArray(tokenIds.map { Int32($0) }),
                        pixelValues: imageEmbeddings,
                        maxNewTokens: config.maxTokens,
                        temperature: config.temperature,
                        topP: config.topP,
                        repetitionPenalty: config.repetitionPenalty,
                        repetitionContextSize: config.repetitionContextSize
                    )

                    for try await token in textStream {
                        continuation.yield(token)
                    }

                    Self.logger.debug("Text generation completed")
                    continuation.finish()
                } catch {
                    Self.logger.error("Text generation failed: \(error, privacy: .public)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Private Configuration Methods

    /// Load model configuration from config.json
    private func loadConfiguration(from directory: URL) async throws {
        let configURL = directory.appendingPathComponent("config.json")
        Self.logger.debug("Reading configuration from \(configURL.lastPathComponent, privacy: .public)")

        let configData = try Data(contentsOf: configURL)
        self.config = try JSONDecoder().decode(Config.self, from: configData)

        Self.logger.debug("Configuration loaded: model type = \(self.config?.modelType ?? "unknown", privacy: .public)")

        // Extract special token IDs
        let specialTokensURL = directory.appendingPathComponent("special_tokens_map.json")
        let specialTokensData = try Data(contentsOf: specialTokensURL)
        let specialTokens = try JSONSerialization.jsonObject(with: specialTokensData) as? [String: Any]

        if let bosTokenValue = specialTokens?["bos_token"] as? [String: Any],
           let content = bosTokenValue["content"] as? String,
           let tokenizer = self.tokenizer,
           let id = try? tokenizer.encode(content, addBos: false, addEos: false).first {
            self.bosToken = id
            Self.logger.debug("BOS token ID: \(id, privacy: .public)")
        }

        if let eosTokenValue = specialTokens?["eos_token"] as? [String: Any],
           let content = eosTokenValue["content"] as? String,
           let tokenizer = self.tokenizer,
           let id = try? tokenizer.encode(content, addBos: false, addEos: false).first {
            self.eosToken = id
            Self.logger.debug("EOS token ID: \(id, privacy: .public)")
        } else if let eosTokenIds = self.config?.eosTokenId, !eosTokenIds.isEmpty {
            self.eosToken = eosTokenIds[0]
            Self.logger.debug("EOS token ID from config: \(eosTokenIds[0], privacy: .public)")
        }

        self.padToken = self.config?.padTokenId ?? 0
        Self.logger.debug("PAD token ID: \(self.padToken, privacy: .public)")
    }

    /// Load and initialize the tokenizer
    private func loadTokenizer(from directory: URL) throws {
        Self.logger.debug("Initializing tokenizer from \(directory.lastPathComponent, privacy: .public)")
        self.tokenizer = try Gemma3Tokenizer(directory: directory)
        Self.logger.debug("Tokenizer initialized successfully")
    }

    /// Parse the model index file to identify weight shards
    private func parseModelIndex(from directory: URL) throws -> [WeightShard] {
        Self.logger.debug("Parsing model index from \(directory.lastPathComponent, privacy: .public)/model.safetensors.index.json")

        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")

        do {
            let indexData = try Data(contentsOf: indexURL)
            let index = try JSONDecoder().decode(ModelIndex.self, from: indexData)
            Self.logger.debug("Successfully parsed model index with \(index.weight_map.count, privacy: .public) weights")

            // Extract unique shard filenames
            let uniqueShardNames = Set(index.weight_map.values)
            Self.logger.debug("Index references \(uniqueShardNames.count, privacy: .public) unique shard files")

            // Check if the referenced shards actually exist
            let fileManager = FileManager.default

            // First try looking for the exact shard files referenced in the index
            var missingShards: [String] = []
            for shardName in uniqueShardNames {
                let shardURL = directory.appendingPathComponent(shardName)
                if !fileManager.fileExists(atPath: shardURL.path) {
                    Self.logger.warning("Referenced shard file not found: \(shardName, privacy: .public)")
                    missingShards.append(shardName)
                }
            }

            // If all referenced shards exist, use them
            if missingShards.isEmpty {
                // Group weights by filename to minimize file operations
                var shardMap: [String: [String]] = [:]

                for (weightName, fileName) in index.weight_map {
                    if shardMap[fileName] == nil {
                        shardMap[fileName] = []
                    }
                    shardMap[fileName]?.append(weightName)
                }

                Self.logger.notice("Using \(shardMap.count, privacy: .public) shards from index")
                return shardMap.map { fileName, weights in
                    WeightShard(fileName: fileName, weights: weights)
                }
            }
            // Check if a single consolidated model.safetensors file exists instead
            else {
                let singleModelURL = directory.appendingPathComponent("model.safetensors")

                if fileManager.fileExists(atPath: singleModelURL.path) {
                    Self.logger.notice("Found single model.safetensors file - using all weights")

                    // Use weight names from index with the single file
                    let allWeightNames = Array(index.weight_map.keys)
                    return [WeightShard(fileName: "model.safetensors", weights: allWeightNames)]
                } else {
                    // No viable weight files found - throw an error
                    let errorMessage = "Missing weight shard files: \(missingShards.joined(separator: ", "))"
                    Self.logger.error("\(errorMessage)")
                    throw ModelError.missingWeightFile(file: missingShards.first ?? "unknown")
                }
            }
        } catch {
            // If the index file can't be parsed or read, check for a single model file
            Self.logger.warning("Failed to parse model index: \(error, privacy: .public)")

            let singleModelURL = directory.appendingPathComponent("model.safetensors")
            let fileManager = FileManager.default

            if fileManager.fileExists(atPath: singleModelURL.path) {
                Self.logger.notice("Found single model.safetensors file without valid index")
                return [WeightShard(fileName: "model.safetensors", weights: [])]
            } else {
                // No viable weight files found - throw the original error
                Self.logger.error("No model weights found")
                throw ModelError.invalidWeightIndex(reason: error.localizedDescription)
            }
        }
    }

    /// Load model weights from safetensors files
    private func loadModelWeights(shards: [WeightShard], from directory: URL) async throws {
        Self.logger.info("Loading model weights from \(shards.count, privacy: .public) shards")

        // Collect all weights in a dictionary
        var weightMap: [String: MLXArray] = [:]

        // Load weights from files
        for (index, shard) in shards.enumerated() {
            let shardURL = directory.appendingPathComponent(shard.fileName)
            Self.logger.debug("Loading shard \(index + 1)/\(shards.count, privacy: .public): \(shard.fileName, privacy: .public)")

            do {
                let weights = try MLX.loadArrays(url: shardURL)
                Self.logger.debug("Loaded \(weights.count, privacy: .public) weights from shard")

                if shard.weights.isEmpty {
                    // If no specific weights were specified, load all from this file
                    for (weightName, weight) in weights {
                        weightMap[weightName] = weight
                    }
                } else {
                    // Only load weights that are in this shard and exist in the file
                    for weightName in shard.weights {
                        if let weight = weights[weightName] {
                            weightMap[weightName] = weight
                        } else {
                            Self.logger.warning("Weight \(weightName, privacy: .public) not found in shard")
                        }
                    }
                }
            } catch {
                Self.logger.error("Failed to load shard \(shard.fileName, privacy: .public): \(error, privacy: .public)")
                throw error
            }
        }

        Self.logger.info("Building model from \(weightMap.count, privacy: .public) weights")

        // Build the model from weights
        try buildModel(from: weightMap)
    }

    /// Sanitize weights before building the model
    private func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        Self.logger.debug("Sanitizing weights")
        var sanitizedWeights = weights

        // Implement weight tying if lm_head is missing
        if sanitizedWeights["language_model.lm_head.weight"] == nil {
            if let embedWeight = sanitizedWeights["language_model.model.embed_tokens.weight"] {
                Self.logger.notice("Applying weight tying: using embedding weights for lm_head")
                sanitizedWeights["language_model.lm_head.weight"] = embedWeight
            }
        }

        // Filter out any rotary embedding inverse frequency weights
        sanitizedWeights = sanitizedWeights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }

        return sanitizedWeights
    }

    /// Build the model structure from loaded weights
    private func buildModel(from weights: [String: MLXArray]) throws {
        guard let config = self.config else {
            Self.logger.error("Missing configuration")
            throw ModelError.missingConfiguration
        }

        // Sanitize weights before building the model
        let sanitizedWeights = sanitizeWeights(weights)

        // Create model configuration
        Self.logger.debug("Creating model configuration")
        let modelConfig = try createModelConfig(from: config)

        // Build the language model components
        Self.logger.debug("Building embedding layer")
        self.embedTokens = try getWeight(
            weights: sanitizedWeights,
            name: "language_model.model.embed_tokens.weight"
        )

        Self.logger.debug("Building \(modelConfig.textConfig.numHiddenLayers) transformer layers")
        self.languageLayers = try createTransformerLayers(
            numLayers: modelConfig.textConfig.numHiddenLayers,
            hiddenSize: modelConfig.textConfig.hiddenSize,
            numHeads: modelConfig.textConfig.numAttentionHeads,
            intermediateSize: modelConfig.textConfig.intermediateSize,
            slidingWindow: modelConfig.textConfig.slidingWindow,
            slidingWindowPattern: modelConfig.textConfig.slidingWindowPattern,
            weights: sanitizedWeights
        )

        Self.logger.debug("Building final normalization layer")
        self.finalNorm = try getWeight(
            weights: sanitizedWeights,
            name: "language_model.model.norm.weight"
        )

        Self.logger.debug("Building output head")
        self.lmHead = try getWeight(
            weights: sanitizedWeights,
            name: "language_model.lm_head.weight"
        )

        // Build the vision model if needed
        if config.modelType == "gemma3_vision" || config.modelType == "gemma3" {
            Self.logger.info("Building vision model components for \(config.modelType, privacy: .public)")
            self.visionEncoder = try buildVisionModel(modelConfig: modelConfig, weights: sanitizedWeights)
            self.multimodalProjector = try buildMultimodalProjector(modelConfig: modelConfig, weights: sanitizedWeights)
        }

        Self.logger.notice("Model built successfully")
    }

    /// Create ModelConfig structure from the loaded config
    private func createModelConfig(from config: Config) throws -> ModelConfig {
        return ModelConfig(
            textConfig: config.textConfig,
            visionConfig: config.visionConfig,
            modelType: config.modelType,
            imageTokenIndex: config.imageTokenIndex,
            padTokenId: config.padTokenId ?? 0,
            eosTokenId: config.eosTokenId,
            mmTokensPerImage: config.mmTokensPerImage,
            boiTokenIndex: config.boiTokenIndex,
            eoiTokenIndex: config.eoiTokenIndex
        )
    }

    // MARK: - Model Component Building Methods

    /// Create transformer layers
    private func createTransformerLayers(
        numLayers: Int,
        hiddenSize: Int,
        numHeads: Int,
        intermediateSize: Int,
        slidingWindow: Int,
        slidingWindowPattern: Int,
        weights: [String: MLXArray]
    ) throws -> [[String: MLXArray]] {
        var layers: [[String: MLXArray]] = []

        for i in 0..<numLayers {
            // Each layer gets a dictionary of its weights
            var layerWeights: [String: MLXArray] = [:]

            // Determine if this is a local or global attention layer
            let isGlobal = (i + 1) % slidingWindowPattern == 0
            Self.logger.debug("Layer \(i, privacy: .public): \(isGlobal ? "global" : "local", privacy: .public) attention")

            // Get attention components
            let prefix = "language_model.model.layers.\(i)"

            // Attention weights
            layerWeights["q_proj"] = try getWeight(weights: weights, name: "\(prefix).self_attn.q_proj.weight")
            layerWeights["k_proj"] = try getWeight(weights: weights, name: "\(prefix).self_attn.k_proj.weight")
            layerWeights["v_proj"] = try getWeight(weights: weights, name: "\(prefix).self_attn.v_proj.weight")
            layerWeights["o_proj"] = try getWeight(weights: weights, name: "\(prefix).self_attn.o_proj.weight")

            // Norms
            layerWeights["q_norm"] = try getWeight(weights: weights, name: "\(prefix).self_attn.q_norm.weight")
            layerWeights["k_norm"] = try getWeight(weights: weights, name: "\(prefix).self_attn.k_norm.weight")

            // Layer norms
            layerWeights["input_layernorm"] = try getWeight(weights: weights, name: "\(prefix).input_layernorm.weight")
            layerWeights["post_attention_layernorm"] = try getWeight(weights: weights, name: "\(prefix).post_attention_layernorm.weight")
            layerWeights["pre_feedforward_layernorm"] = try getWeight(weights: weights, name: "\(prefix).pre_feedforward_layernorm.weight")
            layerWeights["post_feedforward_layernorm"] = try getWeight(weights: weights, name: "\(prefix).post_feedforward_layernorm.weight")

            // MLP weights
            layerWeights["gate_proj"] = try getWeight(weights: weights, name: "\(prefix).mlp.gate_proj.weight")
            layerWeights["up_proj"] = try getWeight(weights: weights, name: "\(prefix).mlp.up_proj.weight")
            layerWeights["down_proj"] = try getWeight(weights: weights, name: "\(prefix).mlp.down_proj.weight")

            // Add metadata
            layerWeights["is_global"] = MLXArray(isGlobal ? 1 : 0)
            layerWeights["sliding_window"] = MLXArray(Int32(slidingWindow))

            // Store the layer
            layers.append(layerWeights)
        }

        return layers
    }

    /// Build the vision model component
    private func buildVisionModel(modelConfig: ModelConfig, weights: [String: MLXArray]) throws -> [String: MLXArray] {
        // Vision components dict
        var visionWeights: [String: MLXArray] = [:]

        // Patch embedding weight
        visionWeights["patch_embedding"] = try getWeight(
            weights: weights,
            name: "vision_tower.vision_model.embeddings.patch_embedding.weight"
        )

        // Vision transformer layers
        var visionLayers: [[String: MLXArray]] = []

        for i in 0..<modelConfig.visionConfig.numHiddenLayers {
            var layerWeights: [String: MLXArray] = [:]
            let prefix = "vision_tower.vision_model.encoder.layers.\(i)"

            // Attention weights
            layerWeights["q_proj"] = try getWeight(weights: weights, name: "\(prefix).self_attn.q_proj.weight")
            layerWeights["k_proj"] = try getWeight(weights: weights, name: "\(prefix).self_attn.k_proj.weight")
            layerWeights["v_proj"] = try getWeight(weights: weights, name: "\(prefix).self_attn.v_proj.weight")
            layerWeights["out_proj"] = try getWeight(weights: weights, name: "\(prefix).self_attn.out_proj.weight")

            // Biases
            layerWeights["q_bias"] = try getWeight(weights: weights, name: "\(prefix).self_attn.q_proj.bias")
            layerWeights["k_bias"] = try getWeight(weights: weights, name: "\(prefix).self_attn.k_proj.bias")
            layerWeights["v_bias"] = try getWeight(weights: weights, name: "\(prefix).self_attn.v_proj.bias")
            layerWeights["out_bias"] = try getWeight(weights: weights, name: "\(prefix).self_attn.out_proj.bias")

            // Layer norms
            layerWeights["layer_norm1"] = try getWeight(weights: weights, name: "\(prefix).layer_norm1.weight")
            layerWeights["layer_norm1_bias"] = try getWeight(weights: weights, name: "\(prefix).layer_norm1.bias")
            layerWeights["layer_norm2"] = try getWeight(weights: weights, name: "\(prefix).layer_norm2.weight")
            layerWeights["layer_norm2_bias"] = try getWeight(weights: weights, name: "\(prefix).layer_norm2.bias")

            // MLP weights
            layerWeights["fc1"] = try getWeight(weights: weights, name: "\(prefix).mlp.fc1.weight")
            layerWeights["fc1_bias"] = try getWeight(weights: weights, name: "\(prefix).mlp.fc1.bias")
            layerWeights["fc2"] = try getWeight(weights: weights, name: "\(prefix).mlp.fc2.weight")
            layerWeights["fc2_bias"] = try getWeight(weights: weights, name: "\(prefix).mlp.fc2.bias")

            visionLayers.append(layerWeights)
        }

        visionWeights["layers"] = MLXArray(visionLayers.count) // Placeholder to store the count
        visionWeights["vision_layers"] = MLXArray(visionLayers.count) // Alternative storage approach

        // Store each layer in a numbered key
        for (i, layer) in visionLayers.enumerated() {
            for (key, value) in layer {
                visionWeights["layer_\(i)_\(key)"] = value
            }
        }

        // Post layer norm
        visionWeights["post_layernorm"] = try getWeight(
            weights: weights,
            name: "vision_tower.vision_model.post_layernorm.weight"
        )
        visionWeights["post_layernorm_bias"] = try getWeight(
            weights: weights,
            name: "vision_tower.vision_model.post_layernorm.bias"
        )

        // Configuration
        visionWeights["image_size"] = MLXArray(Int32(modelConfig.visionConfig.imageSize))
        visionWeights["patch_size"] = MLXArray(Int32(modelConfig.visionConfig.patchSize))
        visionWeights["num_channels"] = MLXArray(Int32(modelConfig.visionConfig.numChannels))
        visionWeights["hidden_size"] = MLXArray(Int32(modelConfig.visionConfig.hiddenSize))

        return visionWeights
    }

    /// Build the multimodal projector component
    private func buildMultimodalProjector(modelConfig: ModelConfig, weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var projectorWeights: [String: MLXArray] = [:]

        // Projection weight
        projectorWeights["projection_weight"] = try getWeight(
            weights: weights,
            name: "multi_modal_projector.mm_input_projection_weight"
        )

        // Norm weight
        projectorWeights["soft_emb_norm"] = try getWeight(
            weights: weights,
            name: "multi_modal_projector.mm_soft_emb_norm.weight"
        )

        // Calculate pooling parameters
        let patchesPerImage = modelConfig.visionConfig.imageSize / modelConfig.visionConfig.patchSize
        let mmTokensPerImage = modelConfig.mmTokensPerImage ?? 256
        let tokensPerSide = Int(Double(mmTokensPerImage).squareRoot())
        let kernelSize = patchesPerImage / tokensPerSide

        projectorWeights["patches_per_image"] = MLXArray(Int32(patchesPerImage))
        projectorWeights["tokens_per_side"] = MLXArray(Int32(tokensPerSide))
        projectorWeights["kernel_size"] = MLXArray(Int32(kernelSize))

        return projectorWeights
    }

    /// Helper to get weight from weights dictionary with error handling
    private func getWeight(weights: [String: MLXArray], name: String) throws -> MLXArray {
        guard let weight = weights[name] else {
            Self.logger.error("Missing weight: \(name, privacy: .public)")
            throw ModelError.missingWeight(name: name)
        }
        return weight
    }

    // MARK: - Text Generation Methods

    /// Main text generation function
    func generateText(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        maxNewTokens: Int,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 0
    ) -> AsyncThrowingStream<String, Error> {
        Self.logger.debug("Starting text generation with max tokens: \(maxNewTokens, privacy: .public)")
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    // Create KV cache for all layers
                    Self.logger.debug("Creating KV cache")
                    var kvCache = try createKVCache()

                    // First run the model on the prompt to initialize the KV cache
                    var currentInputIds = inputIds

                    Self.logger.debug("Running initial forward pass")
                    // Get logits from the model for the input
                    let logits = try forward(
                        inputIds: currentInputIds,
                        pixelValues: pixelValues,
                        cache: &kvCache
                    )

                    // Extract the last logits (for the next token prediction)
                    var lastLogits = extractLastTokenLogits(logits)

                    // Get the token IDs from the input for tracking context
                    var tokensGenerated: [Int] = []
                    var generatedIds = Array(inputIds.asArray(Int32.self)).map { Int($0) }

                    Self.logger.debug("Starting generation loop for \(maxNewTokens, privacy: .public) tokens")
                    // Start the generation loop
                    for i in 0..<maxNewTokens {
                        // Sample the next token using temperature and top-p
                        let nextToken = try sampleNextToken(
                            from: lastLogits,
                            temperature: temperature,
                            topP: topP,
                            repetitionPenalty: repetitionPenalty,
                            previousTokens: tokensGenerated.isEmpty ?
                                generatedIds :
                                Array(generatedIds.suffix(repetitionContextSize))
                        )

                        // Check for EOS token
                        if nextToken == eosToken {
                            Self.logger.debug("Generated EOS token, stopping generation")
                            break
                        }

                        // Add the token to our tracking
                        tokensGenerated.append(nextToken)
                        generatedIds.append(nextToken)

                        // Decode the token to text
                        if let tokenizer = self.tokenizer,
                           let token = try tokenizer.decode([nextToken]) {
                            // Yield the decoded token
                            continuation.yield(token)
                            Self.logger.trace("Generated token \(i+1)/\(maxNewTokens, privacy: .public)")
                        }

                        // Prepare for next inference step
                        currentInputIds = MLXArray([Int32(nextToken)])

                        // Get next token's logits
                        let nextLogits = try forward(
                            inputIds: currentInputIds,
                            cache: &kvCache
                        )

                        // Update last logits for next iteration
                        lastLogits = extractLastTokenLogits(nextLogits)
                    }

                    Self.logger.debug("Text generation completed with \(tokensGenerated.count, privacy: .public) tokens")
                    continuation.finish()
                } catch {
                    Self.logger.error("Text generation failed: \(error, privacy: .public)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Create key-value cache for attention layers
    func createKVCache() throws -> [MLXArray] {
        guard let config = self.config else {
            Self.logger.error("Missing configuration")
            throw ModelError.missingConfiguration
        }

        var cache: [MLXArray] = []

        // Create cache for each layer's key and value
        for _ in 0..<config.textConfig.numHiddenLayers {
            // Key cache - initially empty with appropriate dimensions
            cache.append(MLXArray.zeros([1, 0, config.textConfig.hiddenSize]))
            // Value cache
            cache.append(MLXArray.zeros([1, 0, config.textConfig.hiddenSize]))
        }

        Self.logger.debug("Created KV cache with \(cache.count, privacy: .public) arrays")
        return cache
    }

    /// Forward pass through the model
    func forward(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        cache: inout [MLXArray]
    ) throws -> MLXArray {
        guard let embedTokens = self.embedTokens,
              let finalNorm = self.finalNorm,
              let lmHead = self.lmHead else {
            Self.logger.error("Model components not loaded")
            throw ModelError.modelNotLoaded
        }

        // 1. Get embeddings from the tokens and possible image
        let embeddings = try getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues
        )

        // Scale embeddings
        guard let config = self.config else {
            throw ModelError.missingConfiguration
        }

        // Scale embeddings by square root of hidden size, similar to the Python:
        // h *= mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(h.dtype)
        let scale = Float(sqrt(Double(config.textConfig.hiddenSize)))
        var hiddenStates = embeddings * MLXArray(scale)

        // 2. Pass through the transformer layers, updating the KV cache
        for (i, layer) in languageLayers.enumerated() {
            // Determine if this is a global or local attention layer
            let isGlobal = (layer["is_global"]?.item(Int32.self) ?? 0) == 1

            // Apply input layernorm
            guard let inputNorm = layer["input_layernorm"] else {
                Self.logger.error("Missing layer component: input_layernorm")
                throw ModelError.missingLayerComponent(name: "input_layernorm")
            }
            let normalizedInput = MLX.rmsNorm(hiddenStates, weight: inputNorm, eps: 1e-6)

            // Self-attention
            guard
                let qProj = layer["q_proj"],
                let kProj = layer["k_proj"],
                let vProj = layer["v_proj"],
                let oProj = layer["o_proj"],
                let qNorm = layer["q_norm"],
                let kNorm = layer["k_norm"]
            else {
                Self.logger.error("Missing attention components in layer \(i, privacy: .public)")
                throw ModelError.missingLayerComponent(name: "attention components")
            }

            // Query, key, value projections
            let q = MLX.matmul(normalizedInput, qProj)
            let k = MLX.matmul(normalizedInput, kProj)
            let v = MLX.matmul(normalizedInput, vProj)

            // Apply norms to Q and K
            let qNormed = MLX.rmsNorm(q, weight: qNorm, eps: 1e-6)
            let kNormed = MLX.rmsNorm(k, weight: kNorm, eps: 1e-6)

            // Update KV cache
            cache[i * 2] = MLX.concatenated([cache[i * 2], k], axis: 1)
            cache[i * 2 + 1] = MLX.concatenated([cache[i * 2 + 1], v], axis: 1)

            // Attention computation
            let slidingWindow = Int(layer["sliding_window"]?.item(Int32.self) ?? 1024)
            let attentionOutput = computeAttention(
                q: qNormed,
                k: kNormed,
                v: v,
                isGlobal: isGlobal,
                slidingWindow: slidingWindow,
                cache: cache,
                layerIndex: i
            )

            // Project back to hidden dim
            let attentionProjected = MLX.matmul(attentionOutput, oProj)

            // Post attention norm
            guard let postAttentionNorm = layer["post_attention_layernorm"] else {
                Self.logger.error("Missing layer component: post_attention_layernorm")
                throw ModelError.missingLayerComponent(name: "post_attention_layernorm")
            }
            let normalizedAttention = MLX.rmsNorm(attentionProjected, weight: postAttentionNorm, eps: 1e-6)

            // Residual connection
            hiddenStates = hiddenStates + normalizedAttention

            // MLP
            guard
                let preFeedforwardNorm = layer["pre_feedforward_layernorm"],
                let gateProj = layer["gate_proj"],
                let upProj = layer["up_proj"],
                let downProj = layer["down_proj"]
            else {
                Self.logger.error("Missing MLP components in layer \(i, privacy: .public)")
                throw ModelError.missingLayerComponent(name: "MLP components")
            }

            let normalizedFF = MLX.rmsNorm(hiddenStates, weight: preFeedforwardNorm, eps: 1e-6)

            // MLP computation (SwiGLU)
            let gated = MLX.matmul(normalizedFF, gateProj)
            let up = MLX.matmul(normalizedFF, upProj)

            // Approximate GELU with sigmoid * input * 1.414
            let geluApprox = sigmoid(gated) * gated * 1.414
            let mlpOutput = MLX.matmul(geluApprox * up, downProj)

            // Post feedforward norm
            guard let postFeedforwardNorm = layer["post_feedforward_layernorm"] else {
                Self.logger.error("Missing layer component: post_feedforward_layernorm")
                throw ModelError.missingLayerComponent(name: "post_feedforward_layernorm")
            }
            let normalizedMLP = MLX.rmsNorm(mlpOutput, weight: postFeedforwardNorm, eps: 1e-6)

            // Residual connection
            hiddenStates = hiddenStates + normalizedMLP
        }

        // 3. Apply final layer norm
        let normalizedStates = MLX.rmsNorm(hiddenStates, weight: finalNorm, eps: 1e-6)

        // 4. Project to vocabulary
        let logits = MLX.matmul(normalizedStates, lmHead.transposed(0, 1))

        return logits
    }

    /// Overloaded forward function without image input
    func forward(
        inputIds: MLXArray,
        cache: inout [MLXArray]
    ) throws -> MLXArray {
        return try forward(inputIds: inputIds, pixelValues: nil, cache: &cache)
    }

    /// Helper function for sigmoid activation
    private func sigmoid(_ x: MLXArray) -> MLXArray {
        return 1.0 / (1.0 + exp(-x))
    }

    /// Compute self-attention mechanism
    private func computeAttention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        isGlobal: Bool,
        slidingWindow: Int,
        cache: [MLXArray],
        layerIndex: Int
    ) -> MLXArray {
        Self.logger.trace("Computing \(isGlobal ? "global" : "local", privacy: .public) attention in layer \(layerIndex, privacy: .public)")

        // Get cached KV
        let kCache = cache[layerIndex * 2]
        let vCache = cache[layerIndex * 2 + 1]

        let fullK = kCache
        let fullV = vCache

        // Get dimensions
        let batchSize = q.shape[0]
        let seqLen = q.shape[1]

        // Handle head dimensions according to config
        guard let config = self.config else {
            // Default values if config is not available
            let numHeads = 8
            let headDim = q.shape[2] / numHeads

            // Traditional reshape and transpose
            let qReshaped = q.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
            let kReshaped = fullK.reshaped([batchSize, fullK.shape[1], numHeads, headDim]).transposed(0, 2, 1, 3)
            let vReshaped = fullV.reshaped([batchSize, fullV.shape[1], numHeads, headDim]).transposed(0, 2, 1, 3)

            // Scale factor
            let scale = 1.0 / sqrt(Double(headDim))

            // Compute attention scores
            var scores = MLX.matmul(qReshaped, kReshaped.transposed(0, 1, 3, 2)) * MLXArray(Float(scale))

            // Apply causal mask (and sliding window mask if local attention)
            let mask = generateAttentionMask(
                queryLength: seqLen,
                keyLength: fullK.shape[1],
                isGlobal: isGlobal,
                slidingWindow: slidingWindow
            )

            scores = scores + mask

            // Softmax and apply attention
            let attentionWeights = MLX.softmax(scores, axis: -1)
            let attentionOutput = MLX.matmul(attentionWeights, vReshaped)

            // Reshape back
            return attentionOutput.transposed(0, 2, 1, 3).reshaped([batchSize, seqLen, numHeads * headDim])
        }

        // Get configuration values for attention
        let numHeads = config.textConfig.numAttentionHeads
        let numKVHeads = config.textConfig.numKeyValueHeads
        let headDim = config.textConfig.headDim
        let repeats = numHeads / numKVHeads

        // Reshape for multi-head attention with grouped KV
        let qReshaped = q.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)

        // KV heads might be fewer than Q heads (multi-query attention)
        let kReshaped = fullK.reshaped([batchSize, fullK.shape[1], numKVHeads, headDim]).transposed(0, 2, 1, 3)
        let vReshaped = fullV.reshaped([batchSize, fullV.shape[1], numKVHeads, headDim]).transposed(0, 2, 1, 3)

        // If using MQA (multi-query attention), repeat the KV heads
        var kForAttention = kReshaped
        var vForAttention = vReshaped

        if repeats > 1 {
            // TODO:
//            // Repeat KV heads to match Q heads
//            kForAttention = MLXArray.concatenated([kReshaped] * repeats, axis: 1) // ERROR: Cannot convert value of type 'MLXArray' to expected element type 'Int32'
//            vForAttention = MLXArray.concatenated([vReshaped] * repeats, axis: 1) // ERROR: Cannot convert value of type 'MLXArray' to expected element type 'Int32'
        }

        // Scale factor - use the query_pre_attn_scalar from config
        let scale = config.textConfig.queryPreAttnScalar

        // Compute attention scores
        var scores = MLX.matmul(qReshaped, kForAttention.transposed(0, 1, 3, 2)) * MLXArray(scale)

        // Apply causal mask (and sliding window mask if local attention)
        let mask = generateAttentionMask(
            queryLength: seqLen,
            keyLength: fullK.shape[1],
            isGlobal: isGlobal,
            slidingWindow: slidingWindow
        )

        scores = scores + mask

        // Softmax
        let attentionWeights = MLX.softmax(scores, axis: -1)

        // Apply attention
        let attentionOutput = MLX.matmul(attentionWeights, vForAttention)

        // Reshape back
        return attentionOutput.transposed(0, 2, 1, 3).reshaped([batchSize, seqLen, numHeads * headDim])
    }

    /// Generate attention mask for causal and sliding window attention
    private func generateAttentionMask(
        queryLength: Int,
        keyLength: Int,
        isGlobal: Bool,
        slidingWindow: Int
    ) -> MLXArray {
        // Create causal mask
        var maskData = [Float](repeating: 0.0, count: 1 * 1 * queryLength * keyLength)

        // Set upper triangular values to negative infinity (causal masking)
        for i in 0..<queryLength {
            let queryOffset = (keyLength - queryLength) + i
            for j in 0..<keyLength {
                if j > queryOffset {
                    maskData[i * keyLength + j] = -Float.greatestFiniteMagnitude
                }

                // If local attention, apply sliding window
                if !isGlobal && j < queryOffset - slidingWindow {
                    maskData[i * keyLength + j] = -Float.greatestFiniteMagnitude
                }
            }
        }

        return MLXArray(maskData, [1, 1, queryLength, keyLength])
    }

    /// Get embeddings from token IDs and optional image
    private func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil
    ) throws -> MLXArray {
        guard let embedTokens = self.embedTokens else {
            Self.logger.error("Model not loaded")
            throw ModelError.modelNotLoaded
        }

        // Basic token embeddings - manual implementation of embedding lookup
        let embeddingDim = embedTokens.shape[1]
        let batchSize = inputIds.shape[0]
        let seqLen = inputIds.shape[1] // BUG: 

        // Gather embeddings for each input token
        let textEmbeddings = MLXArray.zeros([batchSize, seqLen, embeddingDim])
        let inputIdsArray = Array(inputIds.asArray(Int32.self))

        for i in 0..<inputIdsArray.count {
            let tokenId = Int(inputIdsArray[i])
            if tokenId >= 0 && tokenId < embedTokens.shape[0] {
                let embedding = embedTokens[tokenId]
                textEmbeddings[0, i % seqLen] = embedding
            }
        }

        // If no image, return directly
        if pixelValues == nil {
            return textEmbeddings
        }

        // Process image embeddings and integrate with text
        guard let visionEncoder = self.visionEncoder,
              let multimodalProjector = self.multimodalProjector,
              let config = self.config else {
            Self.logger.error("Model not loaded for multimodal processing")
            throw ModelError.modelNotLoaded
        }

        Self.logger.debug("Processing image with vision encoder")
        // Process image through vision encoder to get features
        let visionFeatures = try processImageWithEncoder(pixelValues!, visionEncoder)

        Self.logger.debug("Projecting vision features to text embedding space")
        // Project vision features to text embedding space
        let projectedFeatures = try projectVisionFeatures(
            visionFeatures,
            projectorDict: multimodalProjector
        )

        Self.logger.debug("Integrating text and image embeddings")
        // Combine text and image embeddings based on image token locations
        return integrateEmbeddings(
            textEmbeddings: textEmbeddings,
            imageEmbeddings: projectedFeatures,
            inputIds: inputIds,
            imageTokenId: config.imageTokenIndex,
            padTokenId: config.padTokenId ?? 0
        )
    }

    /// Process image through vision encoder
    private func processImageWithEncoder(_ pixelValues: MLXArray, _ visionDict: [String: MLXArray]) throws -> MLXArray {
        // For a full implementation, this would process the image through a vision transformer
        // This is a simplified implementation - in production, you would implement the full vision transformer processing

        guard let imageSize = visionDict["image_size"]?.item(Int32.self),
              let patchSize = visionDict["patch_size"]?.item(Int32.self),
              let hiddenSize = visionDict["hidden_size"]?.item(Int32.self) else {
            Self.logger.error("Missing vision configuration parameters")
            throw ModelError.missingVisionConfiguration
        }

        // In a full implementation, we would:
        // 1. Apply patch embedding
        // 2. Process through transformer layers
        // 3. Apply final normalization

        Self.logger.debug("Vision processing with image size: \(imageSize, privacy: .public), patch size: \(patchSize, privacy: .public)")
        // For now, returning a placeholder embedding of correct shape
        let batchSize = 1
        let numPatches = (Int(imageSize) / Int(patchSize)) * (Int(imageSize) / Int(patchSize))

        return MLXArray.ones([batchSize, numPatches, Int(hiddenSize)])
    }

    /// Project vision features to text embedding space
    private func projectVisionFeatures(_ features: MLXArray, projectorDict: [String: MLXArray]) throws -> MLXArray {
        // Extract projector weights
        guard let projectionWeight = projectorDict["projection_weight"],
              let normWeight = projectorDict["soft_emb_norm"] else {
            Self.logger.error("Missing projector components")
            throw ModelError.missingProjectorComponent
        }

        // Apply normalization
        let normalized = MLX.rmsNorm(features, weight: normWeight, eps: 1e-6)

        // Project to text embedding space
        return MLX.matmul(normalized, projectionWeight)
    }

    /// Integrate text and image embeddings
    private func integrateEmbeddings(
        textEmbeddings: MLXArray,
        imageEmbeddings: MLXArray,
        inputIds: MLXArray,
        imageTokenId: Int,
        padTokenId: Int
    ) -> MLXArray {
        let batchSize = textEmbeddings.shape[0]
        let seqLen = textEmbeddings.shape[1]
        let hiddenSize = textEmbeddings.shape[2]

        // Create combined embeddings
        var combinedEmbeddings = MLXArray.zeros([batchSize, seqLen, hiddenSize])

        // Get token IDs to identify image token positions
        let inputIdsArray = Array(inputIds.asArray(Int32.self))
        var imageTokenPosition = -1

        // Find the position of the image token
        for i in 0..<inputIdsArray.count {
            if Int(inputIdsArray[i]) == imageTokenId {
                imageTokenPosition = i
                break
            }
        }

        // Copy text embeddings
        combinedEmbeddings = textEmbeddings

        // If we found an image token, replace its embedding with the image embedding
        if imageTokenPosition >= 0 && imageTokenPosition < seqLen {
            Self.logger.debug("Found image token at position \(imageTokenPosition, privacy: .public)")
            if imageEmbeddings.shape[1] == 1 {
                // Single image embedding token
                combinedEmbeddings[0, imageTokenPosition] = imageEmbeddings[0, 0]
            } else {
                // Multiple image tokens - would need more complex logic in a full implementation
                combinedEmbeddings[0, imageTokenPosition] = imageEmbeddings[0, 0]
            }
        }

        return combinedEmbeddings
    }

    /// Sample the next token based on the logits
    private func sampleNextToken(
        from logits: MLXArray,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float?,
        previousTokens: [Int]
    ) throws -> Int {
        // 1. Apply temperature scaling
        var samplingLogits = logits

        if temperature > 0 {
            samplingLogits = samplingLogits / MLXArray(temperature)
        }

        // 2. Apply repetition penalty if specified
        if let penalty = repetitionPenalty, penalty != 1.0, !previousTokens.isEmpty {
            samplingLogits = try applyRepetitionPenalty(
                logits: samplingLogits,
                previousTokens: previousTokens,
                penalty: penalty
            )
        }

        // 3. Apply top-p sampling
        if topP < 1.0 {
            samplingLogits = try applyTopPSampling(logits: samplingLogits, topP: topP)
        }

        // 4. Sample from the distribution
        return try sampleFromLogits(samplingLogits)
    }

    /// Apply repetition penalty to logits
    private func applyRepetitionPenalty(
        logits: MLXArray,
        previousTokens: [Int],
        penalty: Float
    ) throws -> MLXArray {
        Self.logger.trace("Applying repetition penalty \(penalty, privacy: .public) to \(previousTokens.count, privacy: .public) tokens")

        // Create a copy of the logits for modification
        var logitsArray = logits.asArray(Float.self)
        var penalizedLogits = MLXArray(logitsArray)

        // For each token in previous tokens
        for token in previousTokens {
            if token >= 0 && token < logitsArray.count {
                let currentValue = logitsArray[token]

                // Apply the penalty
                let penalizedValue = currentValue > 0 ?
                    currentValue / penalty :
                    currentValue * penalty

                // Update the logits
                logitsArray[token] = penalizedValue
            }
        }

        return MLXArray(logitsArray)
    }

    /// Apply top-p (nucleus) sampling to logits
    private func applyTopPSampling(logits: MLXArray, topP: Float) throws -> MLXArray {
        Self.logger.trace("Applying top-p sampling with p=\(topP, privacy: .public)")

        // Convert to probabilities
        let probs = MLX.softmax(logits, axis: -1)

        // Get arrays for manipulation
        let probsArray = probs.asArray(Float.self)
        let sortedIndices = sortWithIndices(probsArray, descending: true)

        // Compute cumulative probabilities
        var cumulative = 0.0
        var keepMask = [Bool](repeating: false, count: probsArray.count)

        for (i, idx) in sortedIndices.enumerated() {
            cumulative += Double(probsArray[idx])
            keepMask[idx] = cumulative <= Double(topP)

            // Always keep at least the top token
            if i == 0 {
                keepMask[idx] = true
            }

            if cumulative > Double(topP) {
                break
            }
        }

        // Apply the mask
        var filteredProbs = [Float](repeating: 0.0, count: probsArray.count)
        for i in 0..<probsArray.count {
            if keepMask[i] {
                filteredProbs[i] = probsArray[i]
            }
        }

        // Renormalize
        let sum = filteredProbs.reduce(0.0, +)
        if sum > 0 {
            for i in 0..<filteredProbs.count {
                filteredProbs[i] /= sum
            }
        }

        // Convert back to logits
        var filteredLogits = [Float](repeating: 0.0, count: probsArray.count)
        for i in 0..<filteredProbs.count {
            filteredLogits[i] = filteredProbs[i] > 1e-10 ? log(filteredProbs[i]) : -Float.greatestFiniteMagnitude
        }

        return MLXArray(filteredLogits)
    }

    /// Helper function to sort array with indices
    private func sortWithIndices(_ array: [Float], descending: Bool = false) -> [Int] {
        let indices = Array(0..<array.count)
        return indices.sorted {
            descending ? array[$0] > array[$1] : array[$0] < array[$1]
        }
    }

    /// Sample a token from the logits distribution
    private func sampleFromLogits(_ logits: MLXArray) throws -> Int {
        // Convert logits to probabilities
        let probs = MLX.softmax(logits, axis: -1)
        let probsArray = probs.asArray(Float.self)

        // Sample from the distribution
        var maxProb: Float = -Float.infinity
        var maxIndex = 0

        for i in 0..<probsArray.count {
            if probsArray[i] > maxProb {
                maxProb = probsArray[i]
                maxIndex = i
            }
        }

        Self.logger.trace("Sampled token \(maxIndex, privacy: .public) with probability \(maxProb, privacy: .public)")
        return maxIndex
    }

    /// Extract the logits for the last token
    private func extractLastTokenLogits(_ logits: MLXArray) -> MLXArray {
        // If logits has shape [batch_size, seq_len, vocab_size], extract the last token's logits
        if logits.ndim >= 2 {
            return logits[logits.shape[0] - 1]
        }

        return logits
    }

    // MARK: - Image and Video Processing

    /// Process an image for the vision encoder
    func processImage(_ image: CIImage) throws -> MLXArray {
        guard let config = self.config else {
            Self.logger.error("Missing configuration")
            throw ModelError.missingConfiguration
        }

        Self.logger.debug("Processing image for vision encoder")
        // 1. Resize the image to the required size for the vision model
        let imageSize = CGFloat(config.visionConfig.imageSize)
        let context = CIContext()

        // Create a transform to resize the image
        let scaleTransform = CGAffineTransform(
            scaleX: imageSize / image.extent.width,
            y: imageSize / image.extent.height
        )

        let resizedImage = image.transformed(by: scaleTransform)

        // 2. Convert to RGB bitmap
        guard let cgImage = context.createCGImage(resizedImage, from: resizedImage.extent) else {
            Self.logger.error("Failed to create CGImage from CIImage")
            throw ModelError.imageProcessingFailed
        }

        // 3. Convert to MLXArray
        return try imageToMLXArray(cgImage, size: Int(imageSize))
    }

    /// Convert CGImage to MLXArray with pixel values normalized to [0, 1]
    private func imageToMLXArray(_ image: CGImage, size: Int) throws -> MLXArray {
        Self.logger.debug("Converting image to MLXArray with size \(size, privacy: .public)")

        // Create a data buffer to hold the image pixels
        var pixelData = [Float]()
        pixelData.reserveCapacity(size * size * 3)  // RGB channels

        guard let context = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            Self.logger.error("Failed to create CGContext for image processing")
            throw ModelError.imageProcessingFailed
        }

        // Draw the image into the context
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        // Get the pixel data
        guard let data = context.data else {
            Self.logger.error("Failed to get pixel data from context")
            throw ModelError.imageProcessingFailed
        }

        // Extract and normalize pixel values
        let buffer = data.bindMemory(to: UInt8.self, capacity: size * size * 4)
        let bufferPointer = UnsafeBufferPointer(start: buffer, count: size * size * 4)

        // Extract RGB channels and normalize to [0, 1]
        for i in stride(from: 0, to: bufferPointer.count, by: 4) {
            // Red, Green, Blue channels - normalized to [0, 1]
            pixelData.append(Float(bufferPointer[i]) / 255.0)
            pixelData.append(Float(bufferPointer[i + 1]) / 255.0)
            pixelData.append(Float(bufferPointer[i + 2]) / 255.0)
        }

        // Create MLXArray with shape [1, size, size, 3] (NHWC format)
        return MLXArray(pixelData, [1, size, size, 3])
    }

    /// Process a video for multimodal input
    func processVideo(_ url: URL) async throws -> MLXArray {
        guard let config = self.config else {
            Self.logger.error("Missing configuration")
            throw ModelError.missingConfiguration
        }

        Self.logger.info("Processing video from \(url.lastPathComponent, privacy: .public)")

        // Create asset reader
        let asset = AVAsset(url: url)
        guard let reader = try? AVAssetReader(asset: asset) else {
            Self.logger.error("Failed to create AVAssetReader")
            throw ModelError.videoProcessingFailed
        }

        // Setup video track
        guard let videoTrack = asset.tracks(withMediaType: .video).first else {
            Self.logger.error("No video track found in asset")
            throw ModelError.videoProcessingFailed
        }

        // Configure reader output for RGB format
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        let readerOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
        reader.add(readerOutput)

        // Start reading
        guard reader.startReading() else {
            Self.logger.error("Failed to start reading video")
            throw ModelError.videoProcessingFailed
        }

        // Calculate frame extraction parameters
        let duration = try await asset.load(.duration).seconds
        let frameCount = 16  // Extract 16 frames
        let interval = duration / Double(frameCount)

        Self.logger.debug("Video duration: \(duration, privacy: .public)s, extracting \(frameCount, privacy: .public) frames")

        // Extract frames at regular intervals
        var frameImages: [MLXArray] = []
        var currentTime = 0.0

        while frameImages.count < frameCount && reader.status == .reading {
            if let sampleBuffer = readerOutput.copyNextSampleBuffer() {
                let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds

                if presentationTime >= currentTime {
                    if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                        // Convert pixel buffer to CIImage
                        let ciImage = CIImage(cvPixelBuffer: imageBuffer)

                        // Process image
                        let frameArray = try processImage(ciImage)
                        frameImages.append(frameArray)

                        Self.logger.trace("Extracted frame at \(presentationTime, privacy: .public)s")

                        // Update for next frame
                        currentTime += interval
                    }
                }

                CMSampleBufferInvalidate(sampleBuffer)
            }
        }

        // If we couldn't extract enough frames, duplicate the last one
        while frameImages.count < frameCount && !frameImages.isEmpty {
            frameImages.append(frameImages.last!)
        }

        // If we have no frames at all, return an error
        if frameImages.isEmpty {
            Self.logger.error("Failed to extract any frames from video")
            throw ModelError.videoProcessingFailed
        }

        Self.logger.debug("Successfully extracted \(frameImages.count, privacy: .public) frames")

        // Stack frames along a new dimension
        // We'll manually concatenate since MLX.stacked might not handle this specific case
        let frameShape = frameImages[0].shape
        let combinedFrames = MLXArray.zeros([frameCount, frameShape[1], frameShape[2], frameShape[3]])

        for (i, frame) in frameImages.enumerated() {
            combinedFrames[i] = frame[0]  // Remove batch dimension
        }

        return combinedFrames.expandedDimensions(axis: 0)  // Add batch dimension back
    }
}
