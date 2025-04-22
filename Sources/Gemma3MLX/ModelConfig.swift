//
//  ModelConfig.swift
//  Gemma3MLX
//
//  Created by mati on 22/04/2025.
//


//
//  ModelConfig.swift
//  Gemma3MLX
//
//  Created on 22/04/2025.
//

import Foundation

/// Model configuration structure for Gemma3
struct ModelConfig {
    let textConfig: Gemma3Model.TextConfig
    let visionConfig: Gemma3Model.VisionConfig
    let modelType: String
    let vocabSize: Int
    let ignoreIndex: Int
    let imageTokenIndex: Int
    let hiddenSize: Int
    let padTokenId: Int
    let eosTokenId: [Int]?
}

// Model types supported by this implementation
enum ModelType: String {
    case gemma3 = "gemma3"
    case gemma3Vision = "gemma3_vision"
    case siglipVision = "siglip_vision_model"
}

// Attention types used in the model
enum AttentionType {
    case local(windowSize: Int)
    case global
}