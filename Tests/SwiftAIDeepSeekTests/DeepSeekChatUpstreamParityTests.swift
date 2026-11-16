import Foundation
import Testing
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities
@testable import SwiftAIDeepSeek

// Translated from packages/deepseek/src/chat/deepseek-chat-language-model.test.ts,
// chat/deepseek-raw-usage.test.ts and the Azure-hosted DeepSeek cases in
// packages/azure/src/azure-openai-provider.test.ts (adapter owned by this package).

private let helloText = #"{"id":"chatcmpl-1","object":"chat.completion","created":1785880000,"model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"system_fingerprint":"fp_eaab8d114b_prod0820_fp8_kvcache"}"#

private func makeModel(_ modelID: String, transport: RecordingTransport, baseURL: String? = nil) throws -> any LanguageModel {
    try DeepSeekProvider(configuration: .init(apiKey: "test-api-key", baseURL: baseURL, transport: transport)).languageModel(modelID)
}

private func requestBody(_ transport: RecordingTransport) async throws -> JSONValue {
    try decodeJSONBody(try #require(await transport.requests().first?.body))
}

private func streamParts(_ model: any LanguageModel, _ request: LanguageModelRequest) async throws -> [LanguageStreamPart] {
    var parts: [LanguageStreamPart] = []
    for try await part in model.stream(request) { parts.append(part) }
    return parts
}

@Suite("DeepSeek chat upstream parity")
struct DeepSeekChatUpstreamParityTests {
    // it('should omit deprecated and ineffective sampling options in default V4 thinking mode')
    @Test func omitsDeprecatedAndIneffectiveSamplingInDefaultV4ThinkingMode() async throws {
        let transport = RecordingTransport(response: jsonResponse(helloText))
        let result = try await makeModel("deepseek-v4-flash", transport: transport).generate(LanguageModelRequest(
            messages: [.user("Hello")], temperature: 0.2, topP: 0.4, presencePenalty: 0.6, frequencyPenalty: 0.5
        ))
        #expect(try await requestBody(transport) == ["model": "deepseek-v4-flash", "messages": [["role": "user", "content": "Hello"]]])
        #expect(result.warnings == [
            AIWarning(type: "deprecated", setting: "frequencyPenalty", message: "frequencyPenalty is deprecated by DeepSeek and has been omitted. Remove frequencyPenalty from the request."),
            AIWarning(type: "deprecated", setting: "presencePenalty", message: "presencePenalty is deprecated by DeepSeek and has been omitted. Remove presencePenalty from the request."),
            AIWarning(type: "unsupported", feature: "temperature", message: "temperature has no effect when DeepSeek thinking is enabled. Set providerOptions.deepseek.thinking.type to 'disabled' to use temperature."),
            AIWarning(type: "unsupported", feature: "topP", message: "topP has no effect when DeepSeek thinking is enabled. Set providerOptions.deepseek.thinking.type to 'disabled' to use topP.")
        ])
    }

    // it('should preserve supported sampling options when V4 thinking is disabled')
    @Test func preservesSamplingWhenV4ThinkingIsDisabled() async throws {
        let transport = RecordingTransport(response: jsonResponse(helloText))
        let result = try await makeModel("deepseek-v4-flash", transport: transport).generate(LanguageModelRequest(
            messages: [.user("Hello")], temperature: 0.2, topP: 0.4, presencePenalty: 0.6, frequencyPenalty: 0.5,
            providerOptions: ["deepseek": ["thinking": ["type": "disabled"]]]
        ))
        #expect(try await requestBody(transport) == [
            "model": "deepseek-v4-flash", "messages": [["role": "user", "content": "Hello"]],
            "temperature": 0.2, "top_p": 0.4, "thinking": ["type": "disabled"]
        ])
        #expect(result.warnings.map(\.setting) == ["frequencyPenalty", "presencePenalty"])
        #expect(result.warnings.allSatisfy { $0.type == "deprecated" && $0.feature == nil })
    }

    // it('should send logprobs provider options') / it('should enable logprobs when topLogprobs is set')
    // plus the upstream `(logprobs === true || topLogprobs != null) && { logprobs: true }` rule.
    static let logprobCases: [(JSONValue, JSONValue)] = [
        (["logprobs": true, "topLogprobs": 1], ["logprobs": true, "top_logprobs": 1]),
        (["topLogprobs": 1], ["logprobs": true, "top_logprobs": 1]),
        (["logprobs": false], [:]),
        (["logprobs": false, "topLogprobs": 2], ["logprobs": true, "top_logprobs": 2])
    ]
    @Test(arguments: logprobCases)
    func sendsLogprobsOnlyWhenEnabled(_ options: JSONValue, _ expected: JSONValue) async throws {
        let transport = RecordingTransport(response: jsonResponse(helloText))
        _ = try await makeModel("deepseek-v4-flash", transport: transport).generate(LanguageModelRequest(
            messages: [.user("Hello")], providerOptions: ["deepseek": options]
        ))
        var body = try #require(try await requestBody(transport).objectValue)
        body.removeValue(forKey: "model"); body.removeValue(forKey: "messages")
        #expect(JSONValue.object(body) == expected)
    }

    // A thinking object without `type` does not select a mode (upstream `thinkingType != null`).
    @Test func thinkingWithoutTypeIsOmittedAndPortableReasoningStillApplies() async throws {
        let transport = RecordingTransport(response: jsonResponse(helloText))
        _ = try await makeModel("deepseek-reasoner", transport: transport).generate(LanguageModelRequest(
            messages: [.user("Hello")], providerOptions: ["deepseek": ["thinking": [:]]]
        ))
        #expect(try await requestBody(transport)["thinking"] == nil)

        let reasoningTransport = RecordingTransport(response: jsonResponse(helloText))
        _ = try await makeModel("deepseek-reasoner", transport: reasoningTransport).generate(LanguageModelRequest(
            messages: [.user("Hello")], reasoning: "none", providerOptions: ["deepseek": ["thinking": [:]]]
        ))
        #expect(try await requestBody(reasoningTransport)["thinking"] == ["type": "disabled"])
    }

    // it.each providerOptions reasoningEffort low/medium/xhigh (exact warning text)
    @Test(arguments: [("low", "low", false), ("medium", "high", true), ("xhigh", "max", true)])
    func mapsProviderReasoningEffort(_ input: String, _ output: String, _ warns: Bool) async throws {
        let transport = RecordingTransport(response: jsonResponse(helloText))
        let result = try await makeModel("deepseek-reasoner", transport: transport).generate(LanguageModelRequest(
            messages: [.user("Hello")], providerOptions: ["deepseek": ["reasoningEffort": .string(input)]]
        ))
        #expect(try await requestBody(transport)["reasoning_effort"] == .string(output))
        #expect(result.warnings == (warns ? [AIWarning(
            type: "compatibility", feature: "reasoningEffort",
            message: #"reasoningEffort "\#(input)" is not a canonical DeepSeek value. mapped to "\#(output)"."#
        )] : []))
    }

    // it('should map legacy providerOptions thinking.type=adaptive to enabled with a compatibility warning')
    @Test func mapsAdaptiveThinkingWithUpstreamWarningText() async throws {
        let transport = RecordingTransport(response: jsonResponse(helloText))
        let result = try await makeModel("deepseek-reasoner", transport: transport).generate(LanguageModelRequest(
            messages: [.user("Hello")], providerOptions: ["deepseek": ["thinking": ["type": "adaptive"]]]
        ))
        #expect(try await requestBody(transport)["thinking"] == ["type": "enabled"])
        #expect(result.warnings.contains(AIWarning(
            type: "compatibility", feature: "thinking.type",
            message: #"thinking.type "adaptive" is not a canonical DeepSeek value. mapped to "enabled"."#
        )))
    }

    // Upstream warning order: message conversion (JSON schema injection) first, then call settings.
    @Test func schemaInjectionWarningPrecedesCallSettingWarnings() async throws {
        let transport = RecordingTransport(response: jsonResponse(helloText))
        let result = try await makeModel("deepseek-chat", transport: transport).generate(LanguageModelRequest(
            messages: [.user("Hello")], topK: 1, seed: 2, responseFormat: .json(schema: ["type": "object"])
        ))
        #expect(result.warnings == [
            AIWarning(type: "compatibility", feature: "responseFormat JSON schema", message: "JSON response schema is injected into the system message."),
            AIWarning(type: "unsupported", feature: "topK"),
            AIWarning(type: "unsupported", feature: "seed")
        ])
    }

    // it('should extract tool call content') with deepseek-tool-call.json
    @Test func extractsToolCallContentInUpstreamOrder() async throws {
        let fixture = #"{"id":"7a630f5b-b7e6-4878-82f8-d77db164d42b","object":"chat.completion","created":1764665845,"model":"deepseek-reasoner","choices":[{"index":0,"message":{"role":"assistant","content":"","reasoning_content":"The user is asking for the weather in San Francisco.","tool_calls":[{"index":0,"id":"call_00_9V0vrf86Pc9aelHCJMZqnJBo","type":"function","function":{"name":"weather","arguments":"{\"location\": \"San Francisco\"}"}}]},"logprobs":null,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":339,"completion_tokens":92,"total_tokens":431,"prompt_tokens_details":{"cached_tokens":320},"completion_tokens_details":{"reasoning_tokens":48},"prompt_cache_hit_tokens":320,"prompt_cache_miss_tokens":19},"system_fingerprint":"fp_eaab8d114b_prod0820_fp8_kvcache"}"#
        let transport = RecordingTransport(response: jsonResponse(fixture))
        let result = try await makeModel("deepseek-reasoner", transport: transport).generate(LanguageModelRequest(
            messages: [.user("Hello")], tools: ["weather": ["type": "object", "properties": ["location": ["type": "string"]]]]
        ))
        #expect(result.content == [
            .reasoning("The user is asking for the weather in San Francisco."),
            .toolCall(result.toolCalls[0])
        ])
        #expect(result.toolCalls[0].id == "call_00_9V0vrf86Pc9aelHCJMZqnJBo")
        #expect(result.toolCalls[0].arguments == #"{"location": "San Francisco"}"#)
        #expect(result.finishReason == "tool-calls")
        #expect(result.providerMetadata == ["deepseek": [
            "promptCacheHitTokens": 320, "promptCacheMissTokens": 19, "responseObject": "chat.completion",
            "choiceIndex": 0, "messageRole": "assistant", "toolCallTypes": ["function"],
            "systemFingerprint": "fp_eaab8d114b_prod0820_fp8_kvcache"
        ]])
        #expect(result.usage?.inputTokens == 339)
        #expect(result.usage?.inputTokensNoCache == 19)
        #expect(result.usage?.outputTextTokens == 44)
        #expect(result.warnings.isEmpty)
    }

    // Upstream returns the parts it found; a null message content is not an error.
    @Test func nullMessageContentYieldsReasoningOnlyContent() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":"thinking"},"finish_reason":"length"}]}"#))
        let result = try await makeModel("deepseek-v4-flash", transport: transport).generate(LanguageModelRequest(messages: [.user("Hello")]))
        #expect(result.content == [.reasoning("thinking")])
        #expect(result.text == "")
        #expect(result.finishReason == "length")
        #expect(result.usage == TokenUsage())
    }

    // it.each([null, undefined])('should tolerate a %s system fingerprint')
    @Test(arguments: [#","system_fingerprint":null"#, ""])
    func toleratesMissingSystemFingerprint(_ fingerprint: String) async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"Hello"},"finish_reason":"stop"}]"# + fingerprint + "}"))
        let result = try await makeModel("deepseek-chat", transport: transport).generate(LanguageModelRequest(messages: [.user("Hello")]))
        let metadata = try #require(result.providerMetadata["deepseek"]?.objectValue)
        #expect(metadata["systemFingerprint"] == nil)
    }

    // deepseek-raw-usage.test.ts: 'preserves complete raw usage through doGenerate parsing'
    @Test func preservesCompleteRawUsageInGenerate() async throws {
        let usage = #"{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"prompt_cache_hit_tokens":20,"prompt_cache_miss_tokens":80,"prompt_tokens_details":{"cached_tokens":20,"provider_prompt_metadata":{"preserved":true}},"completion_tokens_details":{"reasoning_tokens":10,"provider_completion_metadata":["preserved",1]},"provider_usage_metadata":{"nested":{"preserved":true}}}"#
        let transport = RecordingTransport(response: jsonResponse(#"{"id":"chatcmpl-raw-usage","object":"chat.completion","created":1785880000,"model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":"# + usage + "}"))
        let result = try await makeModel("deepseek-v4-flash", transport: transport).generate(LanguageModelRequest(messages: [.user("Hello")]))
        #expect(result.usage?.rawValue == (try decodeJSONBody(Data(usage.utf8))))
        #expect(result.usage?.inputTokens == 100)
        #expect(result.usage?.inputTokensNoCache == 80)
        #expect(result.usage?.inputTokensCacheRead == 20)
        #expect(result.usage?.inputTokensCacheWrite == nil)
        #expect(result.usage?.outputTokens == 50)
        #expect(result.usage?.outputTextTokens == 40)
        #expect(result.usage?.outputReasoningTokens == 10)
    }

    // deepseek-raw-usage.test.ts: 'preserves complete raw usage from the final doStream usage event'
    @Test func streamKeepsFinalUsageEvent() async throws {
        let usage = #"{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"prompt_cache_hit_tokens":20,"prompt_cache_miss_tokens":80,"prompt_tokens_details":{"cached_tokens":20},"completion_tokens_details":{"reasoning_tokens":10}}"#
        let transport = RecordingTransport(response: sseResponse("""
        data: {"id":"chatcmpl-raw-usage","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"provider_usage_metadata":{"stage":"initial"}}}

        data: {"id":"chatcmpl-raw-usage","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":\(usage)}

        data: {"id":"chatcmpl-raw-usage","object":"chat.completion.chunk","choices":[{"index":0,"delta":{}}]}

        data: [DONE]

        """))
        let parts = try await streamParts(try makeModel("deepseek-v4-flash", transport: transport), LanguageModelRequest(messages: [.user("Hello")]))
        guard case let .finishMetadata(reason, finalUsage, metadata) = try #require(parts.last) else {
            Issue.record("missing finish"); return
        }
        #expect(reason == "stop")
        #expect(finalUsage?.rawValue == (try decodeJSONBody(Data(usage.utf8))))
        #expect(finalUsage?.inputTokensNoCache == 80)
        #expect(finalUsage?.outputTextTokens == 40)
        #expect(metadata["deepseek"]?["promptCacheHitTokens"] == 20)
        #expect(metadata["deepseek"]?["promptCacheMissTokens"] == 80)
    }

    // deepseek-raw-usage.test.ts: 'continues validating known usage field %s'
    @Test(arguments: [
        #"{"prompt_tokens":"100"}"#,
        #"{"prompt_tokens":1,"prompt_tokens_details":{"cached_tokens":"20"}}"#,
        #"{"completion_tokens":1,"completion_tokens_details":{"reasoning_tokens":"10"}}"#
    ])
    func rejectsInvalidKnownUsageFields(_ usage: String) async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":"# + usage + "}"))
        await #expect(throws: (any Error).self) {
            _ = try await makeModel("deepseek-v4-flash", transport: transport).generate(LanguageModelRequest(messages: [.user("Hello")]))
        }
    }

    // Stream metadata mirrors upstream: tool-call types from deltas ordered by index.
    @Test func streamCollectsToolCallTypesFromDeltas() async throws {
        let transport = RecordingTransport(response: sseResponse("""
        data: {"id":"s","object":"chat.completion.chunk","system_fingerprint":"fp_1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"b","arguments":"{}"}},{"index":0,"id":"call_a","type":"function","function":{"name":"a","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}

        data: [DONE]

        """))
        let parts = try await streamParts(try makeModel("deepseek-chat", transport: transport), LanguageModelRequest(
            messages: [.user("Hello")], tools: ["a": ["type": "object"], "b": ["type": "object"]]
        ))
        guard case let .finishMetadata(reason, _, metadata) = try #require(parts.last) else {
            Issue.record("missing finish"); return
        }
        #expect(reason == "tool-calls")
        #expect(metadata == ["deepseek": [
            "responseObject": "chat.completion.chunk", "choiceIndex": 0, "messageRole": "assistant",
            "toolCallTypes": ["function", "function"], "systemFingerprint": "fp_1"
        ]])
    }
}

@Suite("Azure-hosted DeepSeek adapter upstream parity")
struct DeepSeekAzureAdapterUpstreamParityTests {
    private func azureModel(_ modelID: String, transport: RecordingTransport) -> DeepSeekLanguageModel {
        let http = AIProviderHTTPClient(providerID: "azure.deepseek", headers: ["api-key": "azure-key"], transport: transport,
            url: { _, path in try providerURL("https://test-resource.openai.azure.com/openai/v1\(path)", queryParams: [:]) })
        return DeepSeekLanguageModel(modelID: modelID, config: DeepSeekModelConfiguration(
            http: http, baseURL: "https://test-resource.openai.azure.com/openai/v1",
            supportsThinking: false, supportsStructuredOutputs: true))
    }

    // it('should map top-level reasoning to Azure DeepSeek reasoning effort')
    @Test func mapsTopLevelReasoningToEffortWithoutThinking() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"index":0,"message":{"role":"assistant","content":"Hi","reasoning_content":"We need"},"finish_reason":"stop"}]}"#))
        let result = try await azureModel("deepseek-v4-pro", transport: transport).generate(LanguageModelRequest(
            messages: [.user("Hello")], reasoning: "high"
        ))
        #expect(try await requestBody(transport) == [
            "messages": [["content": "Hello", "role": "user"]], "model": "deepseek-v4-pro", "reasoning_effort": "high"
        ])
        #expect(result.content.map(\.typeName) == ["reasoning", "text"])
        #expect(result.providerMetadata["azure"]?["messageRole"] == "assistant")
        #expect(result.providerMetadata["deepseek"] == nil)
    }

    // it('should parse providerOptions from the azure namespace')
    @Test func parsesProviderOptionsFromAzureNamespace() async throws {
        let transport = RecordingTransport(response: jsonResponse(helloText))
        _ = try await azureModel("deepseek-v4-flash", transport: transport).generate(LanguageModelRequest(
            messages: [.user("Hello")], reasoning: "high", providerOptions: ["azure": ["reasoningEffort": "max"]]
        ))
        #expect(try await requestBody(transport) == [
            "messages": [["content": "Hello", "role": "user"]], "model": "deepseek-v4-flash", "reasoning_effort": "max"
        ])
    }

    // it('should send a json_schema response format for structured output')
    @Test func sendsJSONSchemaResponseFormat() async throws {
        let transport = RecordingTransport(response: jsonResponse(helloText))
        let schema: JSONValue = ["type": "object", "properties": ["sentiment": ["type": "string"]], "required": ["sentiment"], "additionalProperties": false]
        _ = try await azureModel("deepseek-v4-flash", transport: transport).generate(LanguageModelRequest(
            messages: [.user("Hello")], responseFormat: .json(schema: schema), reasoning: "high"
        ))
        #expect(try await requestBody(transport) == [
            "messages": [["content": "Hello", "role": "user"]], "model": "deepseek-v4-flash", "reasoning_effort": "high",
            "response_format": ["type": "json_schema", "json_schema": ["name": "response", "schema": schema, "strict": true]]
        ])
    }

    // it('should stream reasoning content') — request body only.
    @Test func streamRequestMapsReasoningEffort() async throws {
        let transport = RecordingTransport(response: sseResponse("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
        let parts = try await streamParts(azureModel("deepseek-v4-pro", transport: transport), LanguageModelRequest(messages: [.user("Hello")], reasoning: "high"))
        #expect(try await requestBody(transport) == [
            "messages": [["content": "Hello", "role": "user"]], "model": "deepseek-v4-pro", "reasoning_effort": "high",
            "stream": true, "stream_options": ["include_usage": true]
        ])
        guard case let .streamStart(warnings) = try #require(parts.first) else { Issue.record("missing start"); return }
        #expect(warnings.isEmpty)
    }
}

private extension AIResultContentPart {
    var typeName: String {
        switch self {
        case .reasoning: "reasoning"
        case .text: "text"
        case .toolCall: "tool-call"
        default: "other"
        }
    }
}

@Suite("DeepSeek Responses upstream parity")
struct DeepSeekResponsesUpstreamParityTests {
    // Upstream getArgs pushes input warnings, then tool warnings, then portable reasoning warnings.
    @Test func reasoningMappingWarningFollowsToolWarnings() throws {
        let request = LanguageModelRequest(messages: [.user("Hi")], reasoning: "minimal",
            tools: ["lookup": ["type": "object", "strict": true]])
        let prepared = try deepSeekResponsesPrepared(request, modelID: "deepseek-v4-pro")
        #expect(prepared.body["reasoning"] == ["effort": "low"])
        #expect(prepared.warnings.map(\.feature) == ["strict tool calls", "reasoning"])
    }

    // deepseekFilePartProviderOptions: `fileData: z.literal(true).optional()` rejects false.
    @Test func rejectsFileDataFalseLikeUpstreamSchema() throws {
        let request = LanguageModelRequest(messages: [AIMessage(role: .user, content: [
            .imageURL("https://example.com/a.png", providerMetadata: ["deepseek": ["fileData": false]])
        ])])
        #expect(throws: AIError.invalidArgument(argument: "fileData", message: "DeepSeek fileData must be true when provided.")) {
            _ = try deepSeekResponsesPrepared(request, modelID: "deepseek-v4-pro")
        }
    }
}
