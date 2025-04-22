import Foundation
import MLX
import CoreImage

/// Main client interface for the Gemma3 QAT model
public final class Gemma3Client {
    // MARK: - Properties
    
    /// Underlying model implementation
    private let model = Gemma3Model()
    
    /// Flag indicating if the model is loaded
    private var isModelLoaded = false
    
    // MARK: - Initialization
    
    public init() {}
    
    // MARK: - Public API
    
    /// Load the Gemma3 QAT model from a local URL
    /// - Parameter localURL: URL to the directory containing the model files
    /// - Returns: An async stream reporting progress during loading
    public func load(model localURL: URL) -> AsyncThrowingStream<Progress, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    var progressStream = self.model.load(model: localURL)
                    
                    // Forward progress updates
                    for try await progress in progressStream {
                        continuation.yield(progress)
                    }
                    
                    self.isModelLoaded = true
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    /// Generate text based on the provided configuration
    /// - Parameter config: Configuration for text generation
    /// - Returns: An async stream yielding generated text tokens
    public func generateVLM(config: VLMConfiguration) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    // Verify the model is loaded
                    guard isModelLoaded else {
                        throw Gemma3Error.modelNotLoaded
                    }
                    
                    // Start generation
                    var generationStream = self.model.generateVLM(config: config)
                    
                    // Forward generated text
                    for try await text in generationStream {
                        continuation.yield(text)
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
