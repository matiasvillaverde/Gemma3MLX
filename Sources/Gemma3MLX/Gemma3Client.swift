import Foundation
import MLX
import CoreImage
import OSLog

/// Main client for interacting with Gemma3 models
public class Gemma3Client {
    private static let logger = Logger(subsystem: "com.example.Gemma3MLX", category: "Client")

    /// The underlying model implementation
    private let model: Gemma3Model

    /// Whether the model has been loaded
    private var isModelLoaded: Bool = false

    /// Initialize a new Gemma3 client
    public init() {
        Self.logger.debug("Initializing Gemma3Client")
        model = Gemma3Model()
    }

    /// Load the model from the specified directory
    /// - Parameter modelURL: Directory containing the model files
    /// - Returns: An async stream of progress updates
    public func load(model modelURL: URL) -> AsyncThrowingStream<Progress, Error> {
        Self.logger.info("Loading model from \(modelURL.path, privacy: .public)")

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    var progressStream = model.load(model: modelURL)

                    for try await progress in progressStream {
                        // Convert model's internal progress to public progress
                        let clientProgress = Progress(
                            completed: progress.completed,
                            total: progress.total,
                            description: progress.description
                        )
                        continuation.yield(clientProgress)
                    }

                    self.isModelLoaded = true
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
    /// - Parameter config: Configuration for generation
    /// - Returns: An async stream of generated text tokens
    public func generateVLM(config: VLMConfiguration) -> AsyncThrowingStream<String, Error> {
        Self.logger.info("Starting VLM generation with prompt: \(config.prompt, privacy: .public)")

        // Validate model is loaded
        guard isModelLoaded else {
            Self.logger.error("Model not loaded")
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: NSError(domain: "com.example.Gemma3MLX", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"]))
            }
        }
        
        return model.generateVLM(config: config)
    }
}
