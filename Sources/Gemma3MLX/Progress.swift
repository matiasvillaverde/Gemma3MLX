import Foundation

/// Progress information for model loading
public struct Progress: Sendable {
    public let completed: Int
    public let total: Int
    public let description: String
    
    public var fractionCompleted: Double {
        guard total > 0 else { return 0.0 }
        return Double(completed) / Double(total)
    }
}
