import Foundation

/// Custom errors for Gemma3 client
public enum Gemma3Error: Error, LocalizedError {
    case modelNotLoaded
    case invalidModelFormat
    case generationFailed
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "The Gemma3 model is not loaded. Call load(model:) first."
        case .invalidModelFormat:
            return "The model files are in an invalid format or corrupted."
        case .generationFailed:
            return "Text generation failed. Check the model and inputs."
        }
    }
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
}
