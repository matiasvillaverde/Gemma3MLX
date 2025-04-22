import Foundation
import CoreImage

/// Configuration for VLM generation
public struct VLMConfiguration {
    /// Text prompt for the model
    public let prompt: String

    /// Optional image for visual input
    public let image: CIImage?

    /// Optional video URL for video input
    public let videoURL: URL?

    /// Maximum number of tokens to generate
    public let maxTokens: Int

    /// Temperature for sampling (higher = more random)
    public let temperature: Float

    /// Top-p sampling parameter (nucleus sampling)
    public let topP: Float

    /// Repetition penalty to reduce repeating tokens
    public let repetitionPenalty: Float

    /// Context size for repetition penalty
    public let repetitionContextSize: Int

    /// Initialize with text-only generation
    public init(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.1,
        repetitionContextSize: Int = 64
    ) {
        self.prompt = prompt
        self.image = nil
        self.videoURL = nil
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }

    /// Initialize with image input
    public init(
        prompt: String,
        image: CIImage,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.1,
        repetitionContextSize: Int = 64
    ) {
        self.prompt = prompt
        self.image = image
        self.videoURL = nil
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }

    /// Initialize with video input
    public init(
        prompt: String,
        videoURL: URL,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.1,
        repetitionContextSize: Int = 64
    ) {
        self.prompt = prompt
        self.image = nil
        self.videoURL = videoURL
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }
}
