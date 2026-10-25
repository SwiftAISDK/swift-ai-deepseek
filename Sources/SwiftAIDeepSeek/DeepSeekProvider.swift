import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

public struct DeepSeekProviderSettings: Sendable {
    public var apiKey: String?
    public var baseURL: String?
    public var headers: [String: String]
    public var queryParams: [String: String]
    public var supportsStructuredOutputs: Bool
    public var transport: any AITransport

    public init(apiKey: String? = nil, baseURL: String? = nil, headers: [String: String] = [:],
                queryParams: [String: String] = [:], transport: any AITransport = URLSessionTransport.shared, supportsStructuredOutputs: Bool = false) {
        self.apiKey = apiKey; self.baseURL = baseURL; self.headers = headers
        self.queryParams = queryParams; self.transport = transport; self.supportsStructuredOutputs = supportsStructuredOutputs
    }
}

public final class DeepSeekProvider: AIFileProvider, Sendable {
    public let providerID = "deepseek"
    public let supportedCapabilities: Set<ModelCapability> = [.language]
    private let config: DeepSeekModelConfiguration

    public init(configuration: DeepSeekProviderSettings = .init()) throws {
        let names = ["DEEPSEEK_API_KEY"]
        guard let key = configuration.apiKey ?? environmentValue(names) else {
            throw AIError.missingAPIKey(provider: "deepseek", environmentVariables: names)
        }
        var headers = configuration.headers
        headers["Authorization"] = headers["Authorization"] ?? "Bearer \(key)"
        headers = withUserAgentSuffix(headers, "ai-sdk-deepseek/3.0.73")
        let baseURL = withoutTrailingSlash(configuration.baseURL ?? "https://api.deepseek.com")
        let http = AIProviderHTTPClient(providerID: "deepseek", headers: headers, transport: configuration.transport,
            url: { _, path in try providerURL("\(baseURL)\(path)", queryParams: configuration.queryParams) })
        config = DeepSeekModelConfiguration(http: http, baseURL: baseURL, supportsStructuredOutputs: configuration.supportsStructuredOutputs)
    }

    public func files() -> any AIFileClient { DeepSeekFileClient(config: config) }

    public func chatModel(_ modelID: String) throws -> any LanguageModel { try languageModel(modelID) }

    public func languageModel(_ modelID: String) throws -> any LanguageModel {
        DeepSeekLanguageModel(modelID: modelID, config: config.withProviderID("deepseek.chat"))
    }

    public func embeddingModel(_ modelID: String) throws -> any EmbeddingModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .embedding, modelID: modelID)
    }

    public func imageModel(_ modelID: String) throws -> any ImageModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .image, modelID: modelID)
    }

    public func transcriptionModel(_ modelID: String) throws -> any TranscriptionModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .transcription, modelID: modelID)
    }

    public func speechModel(_ modelID: String) throws -> any SpeechModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .speech, modelID: modelID)
    }

    public func videoModel(_ modelID: String) throws -> any VideoModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .video, modelID: modelID)
    }

    public func rerankingModel(_ modelID: String) throws -> any RerankingModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .reranking, modelID: modelID)
    }
}

public extension AIProviders {
    static func deepSeek(configuration: DeepSeekProviderSettings) throws -> DeepSeekProvider {
        try DeepSeekProvider(configuration: configuration)
    }
}

/// Adapter configuration also used by Azure's hosted DeepSeek models.
public struct DeepSeekModelConfiguration: AIProviderHTTPConfiguration {
    public var http: AIProviderHTTPClient
    public var baseURL: String
    public var supportsThinking: Bool
    public var supportsStructuredOutputs: Bool
    public init(http: AIProviderHTTPClient, baseURL: String,
                supportsThinking: Bool = true, supportsStructuredOutputs: Bool = false) {
        self.http = http; self.baseURL = withoutTrailingSlash(baseURL)
        self.supportsThinking = supportsThinking; self.supportsStructuredOutputs = supportsStructuredOutputs
    }
}
