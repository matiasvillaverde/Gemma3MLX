import XCTest
import CoreImage
import OSLog
@testable import Gemma3MLX

final class Gemma3MLXTests: XCTestCase {
    var client: Gemma3Client!
    var modelURL: URL!
    var imageURL: URL!
    
    override func setUpWithError() throws {
        client = Gemma3Client()
        
        // Get the URL to the test bundle resources
        let testBundle = Bundle.module
        
        // Get model path
        guard let modelURL = testBundle.url(forResource: "models", withExtension: nil) else {
            XCTFail("Could not find models directory")
            return
        }
        self.modelURL = modelURL
        
        // Get test image path
        guard let imageURL = testBundle.url(forResource: "test", withExtension: "png", subdirectory: "images") else {
            XCTFail("Could not find test image")
            return
        }
        self.imageURL = imageURL
        
        os_log("Setup completed, model path: %{public}@, image path: %{public}@", 
               log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "Setup"),
               type: .info,
               modelURL.path, 
               imageURL.path)
    }
    
    override func tearDownWithError() throws {
        client = nil
    }
    
    // Test 1: Load model and generate text
    func testModelLoadAndGenerate() async throws {
        os_log("Starting test: testModelLoadAndGenerate", 
               log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "TextGeneration"),
               type: .info)
        
        // Load the model
        let loadExpectation = expectation(description: "Model loading")
        
        Task {
            do {
                let progressStream = client.load(model: modelURL)
                for try await progress in progressStream {
                    os_log("Loading progress: %{public}f%% - %{public}@", 
                           log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "TextGeneration"),
                           type: .debug,
                           progress.fractionCompleted * 100, 
                           progress.description)
                }
                loadExpectation.fulfill()
            } catch {
                os_log("Model loading failed: %{public}@", 
                       log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "TextGeneration"),
                       type: .error,
                       error.localizedDescription)
                XCTFail("Model loading failed: \(error)")
            }
        }
        
        await fulfillment(of: [loadExpectation], timeout: 60.0)
        
        // Test text generation
        let generateExpectation = expectation(description: "Text generation")
        
        var generatedText = ""
        
        Task {
            do {
                os_log("Starting text generation with prompt: Who wrote the theory of relativity?", 
                       log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "TextGeneration"),
                       type: .info)
                
                let config = VLMConfiguration(
                    prompt: "Who wrote the theory of relativity?",
                    maxTokens: 100,
                    temperature: 0.7
                )
                
                let textStream = client.generateVLM(config: config)

                for try await text in textStream {
                    os_log("Generated token: %{public}@", 
                           log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "TextGeneration"),
                           type: .debug,
                           text)
                    generatedText += text
                }
                
                os_log("Full generated text: %{public}@", 
                       log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "TextGeneration"),
                       type: .info,
                       generatedText)
                generateExpectation.fulfill()
            } catch {
                os_log("Text generation failed: %{public}@", 
                       log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "TextGeneration"),
                       type: .error,
                       error.localizedDescription)
                XCTFail("Text generation failed: \(error)")
            }
        }
        
        await fulfillment(of: [generateExpectation], timeout: 30.0)
        
        // Assert that the text contains "Einstein"
        XCTAssertTrue(
            generatedText.contains("Einstein"),
            "Generated text should mention Einstein when asking about theory of relativity"
        )
    }
    
    // Test 2: Load model and generate text from image
    func testModelImageGenerate() async throws {
        os_log("Starting test: testModelImageGenerate", 
               log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ImageGeneration"),
               type: .info)
        
        // Load the model
        let loadExpectation = expectation(description: "Model loading")
        
        Task {
            do {
                let progressStream = client.load(model: modelURL)
                for try await progress in progressStream {
                    os_log("Loading progress: %{public}f%% - %{public}@", 
                           log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ImageGeneration"),
                           type: .debug,
                           progress.fractionCompleted * 100, 
                           progress.description)
                }
                loadExpectation.fulfill()
            } catch {
                os_log("Model loading failed: %{public}@", 
                       log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ImageGeneration"),
                       type: .error,
                       error.localizedDescription)
                XCTFail("Model loading failed: \(error)")
            }
        }
        
        await fulfillment(of: [loadExpectation], timeout: 60.0)
        
        // Load the test image
        guard let ciImage = CIImage(contentsOf: imageURL) else {
            os_log("Could not load test image", 
                   log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ImageGeneration"),
                   type: .error)
            XCTFail("Could not load test image")
            return
        }
        
        os_log("Successfully loaded test image", 
               log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ImageGeneration"),
               type: .info)
        
        // Test image-based text generation
        let generateExpectation = expectation(description: "Image-based generation")
        
        var generatedText = ""
        
        Task {
            do {
                os_log("Starting image-based generation with prompt: What can you see in this image?", 
                       log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ImageGeneration"),
                       type: .info)
                
                let config = VLMConfiguration(
                    prompt: "What can you see in this image?",
                    image: ciImage,
                    maxTokens: 100,
                    temperature: 0.7
                )
                
                let textStream = client.generateVLM(config: config)

                for try await text in textStream {
                    os_log("Generated token: %{public}@", 
                           log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ImageGeneration"),
                           type: .debug,
                           text)
                    generatedText += text
                }
                
                os_log("Full generated text: %{public}@", 
                       log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ImageGeneration"),
                       type: .info,
                       generatedText)
                generateExpectation.fulfill()
            } catch {
                os_log("Image-based generation failed: %{public}@", 
                       log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ImageGeneration"),
                       type: .error,
                       error.localizedDescription)
                XCTFail("Image-based generation failed: \(error)")
            }
        }
        
        await fulfillment(of: [generateExpectation], timeout: 30.0)
        
        // Assert that the text contains "cat"
        XCTAssertTrue(
            generatedText.contains("cat"),
            "Generated text should mention a cat when analyzing the test image"
        )
    }
    
    // Simple test that just checks if the model can be loaded
    func testModelCanLoad() async throws {
        os_log("Starting test: testModelCanLoad", 
               log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ModelLoading"),
               type: .info)
        
        let loadExpectation = expectation(description: "Basic model loading")
        
        Task {
            do {
                let progressStream = client.load(model: modelURL)
                for try await progress in progressStream {
                    os_log("Loading progress: %{public}f%% - %{public}@", 
                           log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ModelLoading"),
                           type: .debug,
                           progress.fractionCompleted * 100, 
                           progress.description)
                }
                loadExpectation.fulfill()
            } catch {
                os_log("Basic model loading failed: %{public}@", 
                       log: OSLog(subsystem: "com.gemma3.mlx.tests", category: "ModelLoading"),
                       type: .error,
                       error.localizedDescription)
                XCTFail("Basic model loading failed: \(error)")
            }
        }
        
        await fulfillment(of: [loadExpectation], timeout: 60.0)
        
        // If we get here, the model loaded successfully
        XCTAssertTrue(true, "Model loaded successfully")
    }
}
