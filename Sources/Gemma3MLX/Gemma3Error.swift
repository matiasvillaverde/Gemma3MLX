import Foundation

/// Custom errors for Gemma3 client
public enum Gemma3Error: Error, LocalizedError {
    case modelNotLoaded
    case invalidModelFormat
    case generationFailed
    case invalidGenerationConfig(reason: String)
    case imageLoadingFailed
    case videoLoadingFailed
    case tokenizationFailed(reason: String)
}

/// Custom errors for model operations
enum ModelError: Error, LocalizedError {
    case modelNotLoaded
    case missingConfiguration
    case tokenizerNotInitialized
    case invalidTokenizer
    case weightLoadingFailed
    case forwardPassFailed
    case imageProcessingFailed
    case videoProcessingFailed
    case missingWeight(name: String)
    case missingWeightFile(file: String)
    case invalidWeightIndex(reason: String)
    case custom(message: String)
    case missingLayerComponent(name: String)
    case missingVisionConfiguration
    case missingProjectorComponent
    case dimensionMismatch(expected: Int, found: Int)
}
