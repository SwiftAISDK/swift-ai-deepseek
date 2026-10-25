@testable import SwiftAIDeepSeek
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities
import Testing
import SwiftAI


@Test func deepSeekLanguageStreamsReasoningAndIncludesUsageOptions() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"id":"ds-1","created":1780326500,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}

    data: {"id":"ds-1","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5,"prompt_cache_hit_tokens":1,"prompt_cache_miss_tokens":1}}

    data: [DONE]

    """, headers: ["x-deepseek": "stream"]))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-reasoner")

    var streamStartWarnings: [AIWarning]?
    var responseMetadata: AIResponseMetadata?
    var reasoningLifecycle: [String] = []
    var textLifecycle: [String] = []
    var finishReason: String?
    var totalTokens: Int?
    var providerMetadata: [String: JSONValue] = [:]
    for try await part in model.stream(LanguageModelRequest(
        messages: [.user("Hi")],
        providerOptions: ["deepseek": ["thinking": ["type": "enabled"], "reasoningEffort": "xhigh"]]
    )) {
        switch part {
        case let .streamStart(warnings):
            streamStartWarnings = warnings
        case let .responseMetadata(metadata):
            responseMetadata = metadata
        case let .reasoningStart(id, _):
            reasoningLifecycle.append("start:\(id)")
        case let .reasoningDeltaPart(id, delta, _):
            reasoningLifecycle.append("delta:\(id):\(delta)")
        case let .reasoningEnd(id, _):
            reasoningLifecycle.append("end:\(id)")
        case let .textStart(id, _):
            textLifecycle.append("start:\(id)")
        case let .textDeltaPart(id, delta, _):
            textLifecycle.append("delta:\(id):\(delta)")
        case let .textEnd(id, _), let .textEndWithCitations(id, _, _):
            textLifecycle.append("end:\(id)")
        case let .finishMetadata(reason, usage, metadata):
            finishReason = reason
            totalTokens = usage?.totalTokens
            providerMetadata = metadata
        default:
            break
        }
    }

    #expect(streamStartWarnings == [AIWarning(
        type: "compatibility",
        feature: "reasoningEffort",
        message: "reasoningEffort xhigh is not a canonical DeepSeek value and was mapped to max."
    )])
    #expect(responseMetadata?.id == "ds-1")
    #expect(responseMetadata?.modelID == "deepseek-reasoner")
    #expect(responseMetadata?.headers["x-deepseek"] == "stream")
    #expect(reasoningLifecycle == ["start:reasoning-0", "delta:reasoning-0:think", "end:reasoning-0"])
    #expect(textLifecycle == ["start:txt-0", "delta:txt-0:answer", "end:txt-0"])
    #expect(finishReason == "stop")
    #expect(totalTokens == 5)
    #expect(providerMetadata["deepseek"]?["promptCacheHitTokens"]?.intValue == 1)
    #expect(providerMetadata["deepseek"]?["promptCacheMissTokens"]?.intValue == 1)
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://api.deepseek.com/chat/completions")
    #expect(request.headers["authorization"] == "Bearer deepseek-key")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["stream"] == true)
    #expect(body["stream_options"]?["include_usage"]?.boolValue == true)
    #expect(body["thinking"]?["type"]?.stringValue == "enabled")
    #expect(body["reasoning_effort"]?.stringValue == "max")
}

@Test func deepSeekLanguageParsesToolCallsMetadataAndReasoning() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"id":"ds-generate","created":1780326500,"model":"deepseek-reasoner","choices":[{"message":{"role":"assistant","content":"","reasoning_content":"I should call weather.","tool_calls":[{"id":"call_weather","index":0,"type":"function","function":{"name":"weather","arguments":"{\"location\":\"San Francisco\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13,"prompt_cache_hit_tokens":2,"prompt_cache_miss_tokens":7,"completion_tokens_details":{"reasoning_tokens":1}}}"#, headers: ["x-deepseek": "generate"]))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-reasoner")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("Weather?")]))

    #expect(result.text == "")
    #expect(result.reasoning == "I should call weather.")
    #expect(result.finishReason == "tool-calls")
    #expect(result.usage?.totalTokens == 13)
    #expect(result.usage?.inputTokensNoCache == 7)
    #expect(result.usage?.inputTokensCacheRead == 2)
    #expect(result.usage?.outputTextTokens == 3)
    #expect(result.usage?.outputReasoningTokens == 1)
    #expect(result.responseMetadata.id == "ds-generate")
    #expect(result.responseMetadata.modelID == "deepseek-reasoner")
    #expect(result.responseMetadata.headers["x-deepseek"] == "generate")
    #expect(result.providerMetadata["deepseek"]?["promptCacheHitTokens"]?.intValue == 2)
    #expect(result.providerMetadata["deepseek"]?["promptCacheMissTokens"]?.intValue == 7)
    #expect(result.toolCalls.count == 1)
    #expect(result.toolCalls[0].id == "call_weather")
    #expect(result.toolCalls[0].name == "weather")
    #expect(try decodeJSONBody(Data(result.toolCalls[0].arguments.utf8))["location"]?.stringValue == "San Francisco")
}

@Test func deepSeekLanguageMapsMissingFinishReasonToOther() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"id":"ds-generate","model":"deepseek-chat","choices":[{"message":{"content":"ok"},"finish_reason":null}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-chat")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("Hi")]))

    #expect(result.text == "ok")
    #expect(result.finishReason == "other")
}

@Test func deepSeekLanguageMapsMissingUsageToEmptyUsageLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"id":"ds-generate","model":"deepseek-chat","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-chat")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("Hi")]))

    #expect(result.text == "ok")
    #expect(result.usage == TokenUsage())
}

@Test func deepSeekLanguageSerializesToolResultOutputsLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-chat")

    _ = try await model.generate(LanguageModelRequest(messages: [
        AIMessage(role: .tool, content: [
            .toolResult(AIToolResult(
                toolCallID: "call-denied",
                toolName: "lookup",
                result: .object(["type": .string("execution-denied")])
            )),
            .toolResult(AIToolResult(
                toolCallID: "call-json",
                toolName: "lookup",
                result: .object(["type": .string("json"), "value": .object(["ok": .bool(true)])])
            )),
            .toolResult(AIToolResult(
                toolCallID: "call-content",
                toolName: "lookup",
                result: .object(["type": .string("content"), "value": .array([.object(["type": .string("text"), "text": .string("found")])])])
            ))
        ])
    ]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let messages = try #require(body["messages"]?.arrayValue)
    #expect(messages[0]["content"]?.stringValue == "Tool call execution denied.")
    #expect(messages[1]["content"]?.stringValue == #"{"ok":true}"#)
    let contentValue = try decodeJSONBody(Data(try #require(messages[2]["content"]?.stringValue).utf8))
    #expect(contentValue[0]?["type"]?.stringValue == "text")
    #expect(contentValue[0]?["text"]?.stringValue == "found")
}

@Test func deepSeekProviderAddsVersionedUserAgentSuffix() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        headers: ["User-Agent": "custom-client/1.0"],
        transport: transport
    ))
    let model = try provider.languageModel("deepseek-chat")

    _ = try await model.generate(LanguageModelRequest(messages: [.user("Hi")]))

    let request = try #require(await transport.requests().first)
    #expect(request.headers["authorization"] == "Bearer deepseek-key")
    #expect(request.headers["user-agent"] == "custom-client/1.0 ai-sdk-deepseek/3.0.73")
}

@Test func deepSeekLanguageGeneratesMissingToolCallIDLikeUpstreamV4() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"id":"ds-generate","model":"deepseek-chat","choices":[{"message":{"content":"","tool_calls":[{"index":0,"type":"function","function":{"name":"weather","arguments":"{\"location\":\"Tokyo\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"total_tokens":3}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-chat")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("Weather?")]))

    let call = try #require(result.toolCalls.first)
    #expect(call.id.count == 16)
    #expect(call.id != "tool-call-0")
    #expect(call.name == "weather")
    #expect(try decodeJSONBody(Data(call.arguments.utf8))["location"]?.stringValue == "Tokyo")
}

@Test func deepSeekLanguageStreamsErrorChunksAndParseErrorsAsErrorPartsLikeUpstream() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"error":{"message":"quota exhausted","type":"rate_limit_error","code":"rate_limit"}}

    data: not-json

    data: [DONE]

    """))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-chat")

    var errors: [String] = []
    var finishReason: String?
    var finishUsage: TokenUsage?
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Hi")])) {
        switch part {
        case let .error(message, _):
            errors.append(message)
        case let .finishMetadata(reason, usage, _):
            finishReason = reason
            finishUsage = usage
        default:
            break
        }
    }

    #expect(errors.first == "quota exhausted")
    #expect(errors.count == 2)
    #expect(finishReason == "error")
    #expect(finishUsage == TokenUsage())
}

@Test func deepSeekChatModelUsesNativeReasoningMapping() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.chatModel("deepseek-chat")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        reasoning: "xhigh"
    ))

    #expect(model.providerID == "deepseek.chat")
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["thinking"]?["type"]?.stringValue == "enabled")
    #expect(body["reasoning_effort"]?.stringValue == "max")
}

@Test func deepSeekTopLevelReasoningCompatibilityWarnings() async throws {
    let transport = RecordingTransport(responses: [
        jsonResponse(#"{"choices":[{"message":{"content":"minimal"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#),
        jsonResponse(#"{"choices":[{"message":{"content":"xhigh"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#)
    ])
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-reasoner")

    let minimal = try await model.generate(LanguageModelRequest(messages: [.user("Hi")], reasoning: "minimal"))
    let xhigh = try await model.generate(LanguageModelRequest(messages: [.user("Hi")], reasoning: "xhigh"))

    #expect(minimal.warnings == [
        AIWarning(
            type: "compatibility",
            feature: "reasoning",
            message: "reasoning \"minimal\" is not directly supported by this model. mapped to effort \"low\"."
        )
    ])
    #expect(xhigh.warnings == [
        AIWarning(
            type: "compatibility",
            feature: "reasoning",
            message: "reasoning \"xhigh\" is not directly supported by this model. mapped to effort \"max\"."
        )
    ])

    let requests = await transport.requests()
    let minimalBody = try decodeJSONBody(try #require(requests[0].body))
    let xhighBody = try decodeJSONBody(try #require(requests[1].body))
    #expect(minimalBody["reasoning_effort"]?.stringValue == "low")
    #expect(xhighBody["reasoning_effort"]?.stringValue == "max")
}

@Test func deepSeekProviderReasoningEffortPassesThrough() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-reasoner")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        providerOptions: ["deepseek": ["reasoningEffort": "xhigh"]]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["thinking"] == nil)
    #expect(body["reasoning_effort"]?.stringValue == "max")
}

@Test func deepSeekProviderOptionsValidateAndStripLikeUpstreamSchema() async throws {
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: RecordingTransport(responses: [])))
    let model = try provider.languageModel("deepseek-reasoner")

    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.deepseek", message: "DeepSeek provider options must be an object.")) {
        _ = try await model.generate(LanguageModelRequest(
            messages: [.user("Hi")],
            providerOptions: ["deepseek": "not-an-object"]
        ))
    }

    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.deepseek.reasoningEffort", message: "DeepSeek reasoningEffort must be low, medium, high, xhigh, or max.")) {
        _ = try await model.generate(LanguageModelRequest(
            messages: [.user("Hi")],
            providerOptions: ["deepseek": ["reasoningEffort": "minimal"]]
        ))
    }

    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.deepseek.thinking", message: "DeepSeek thinking must be an object.")) {
        _ = try await model.generate(LanguageModelRequest(
            messages: [.user("Hi")],
            providerOptions: ["deepseek": ["thinking": "enabled"]]
        ))
    }

    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.deepseek.thinking.type", message: "DeepSeek thinking.type must be adaptive, enabled, or disabled.")) {
        _ = try await model.generate(LanguageModelRequest(
            messages: [.user("Hi")],
            providerOptions: ["deepseek": ["thinking": ["type": "on"]]]
        ))
    }

    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.deepseek.strictJsonSchema", message: "DeepSeek strictJsonSchema must be a boolean.")) {
        _ = try await model.generate(LanguageModelRequest(
            messages: [.user("Hi")],
            providerOptions: ["deepseek": ["strictJsonSchema": "true"]]
        ))
    }
}

@Test func deepSeekProviderOptionsStripUnknownKeysAndFilterNestedThinking() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-reasoner")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        providerOptions: [
            "deepseek": [
                "thinking": [
                    "type": "adaptive",
                    "unsupported": "drop-me"
                ],
                "reasoningEffort": "max",
                "unsupportedProperty": "drop-me"
            ]
        ],
        extraBody: [
            "deepseek": [
                "thinking": ["type": "enabled"],
                "reasoningEffort": "low"
            ]
        ]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["thinking"]?["type"]?.stringValue == "enabled")
    #expect(body["thinking"]?["unsupported"] == nil)
    #expect(body["reasoning_effort"]?.stringValue == "max")
    #expect(body["unsupportedProperty"] == nil)
    #expect(body["reasoningEffort"] == nil)
    #expect(body["deepseek"] == nil)
}

@Test func deepSeekProviderOptionsNullNamespaceKeepsExtraBodyEscapeHatch() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-reasoner")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        providerOptions: ["deepseek": .null],
        extraBody: [
            "deepseek": [
                "thinking": ["type": "enabled"],
                "reasoningEffort": "high"
            ]
        ]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["thinking"]?["type"]?.stringValue == "enabled")
    #expect(body["reasoning_effort"]?.stringValue == "high")
}

@Test func deepSeekReasoningNoneDisablesThinkingAndDropsEffort() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-chat")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        reasoning: "none",
        providerOptions: ["deepseek": ["reasoningEffort": "xhigh"]]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["thinking"]?["type"]?.stringValue == "disabled")
    #expect(body["reasoning_effort"] == nil)
    #expect(body["reasoningEffort"] == nil)
}

@Test func deepSeekLanguageConvertsUserImageFilePartsLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-chat")

    let result = try await model.generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .text("Hello"),
            .file(mimeType: "image/png", data: Data([0, 1, 2, 3]), filename: "image.png")
        ])
    ]))

    #expect(result.warnings.isEmpty)
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let content = try #require(body["messages"]?[0]?["content"]?.arrayValue)
    #expect(content[0]["type"]?.stringValue == "text")
    #expect(content[0]["text"]?.stringValue == "Hello")
    #expect(content[1]["type"]?.stringValue == "image_url")
    #expect(content[1]["image_url"]?["url"]?.stringValue == "data:image/png;base64,AAECAw==")
}

@Test func deepSeekLanguageDropsUnsupportedNonImageFilesWithWarning() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-chat")

    let result = try await model.generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .text("Hello"),
            .file(mimeType: "application/pdf", data: Data([0, 1, 2, 3]), filename: "file.pdf")
        ])
    ]))

    #expect(result.warnings == [
        AIWarning(type: "unsupported", feature: "user message part type: file")
    ])
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["messages"]?[0]?["content"]?.stringValue == "Hello")
}

@Test func deepSeekLanguageWarnsForProviderDefinedToolsAndUnsupportedToolChoice() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-chat")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        tools: [
            "lookup": [
                "type": "object",
                "properties": ["query": ["type": "string"]]
            ],
            "deepseek.search": [
                "type": "provider",
                "id": "deepseek.search"
            ]
        ],
        toolChoice: ["type": "provider"]
    ))

    #expect(result.warnings == [
        AIWarning(type: "unsupported", feature: "provider-defined tool deepseek.search"),
        AIWarning(type: "unsupported", feature: "tool choice type: provider")
    ])
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let tools = try #require(body["tools"]?.arrayValue)
    #expect(tools.count == 1)
    #expect(tools[0]["function"]?["name"]?.stringValue == "lookup")
    #expect(body["tool_choice"] == nil)
}

@Test func deepSeekLanguageMapsJsonResponseFormatNativeOptionsAndFunctionTools() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"{\"forecast\":\"sunny\"}"},"finish_reason":"stop"}],"usage":{"total_tokens":5}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-reasoner")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Weather?")],
        topK: 10,
        presencePenalty: 0.1,
        frequencyPenalty: 0.2,
        seed: 123,
        responseFormat: .json(),
        tools: [
            "weather": [
                "type": "object",
                "description": "Look up weather.",
                "properties": ["location": ["type": "string"]],
                "required": ["location"]
            ]
        ],
        providerOptions: [
            "deepseek": ["thinking": ["type": "enabled"]]
        ],
        extraBody: [
            "toolChoice": ["type": "tool", "toolName": "weather"]
        ]
    ))

    #expect(result.warnings == [
        AIWarning(type: "unsupported", feature: "topK"),
        AIWarning(type: "unsupported", feature: "seed"),
        AIWarning(type: "deprecated", feature: "frequencyPenalty", message: "frequencyPenalty is deprecated by DeepSeek and has been omitted."),
        AIWarning(type: "deprecated", feature: "presencePenalty", message: "presencePenalty is deprecated by DeepSeek and has been omitted.")
    ])
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["messages"]?[0]?["role"]?.stringValue == "system")
    #expect(body["messages"]?[0]?["content"]?.stringValue == "Return JSON.")
    #expect(body["messages"]?[1]?["role"]?.stringValue == "user")
    #expect(body["response_format"]?["type"]?.stringValue == "json_object")
    #expect(body["thinking"]?["type"]?.stringValue == "enabled")
    #expect(body["presence_penalty"] == nil)
    #expect(body["frequency_penalty"] == nil)
    #expect(body["topK"] == nil)
    #expect(body["seed"] == nil)
    #expect(body["deepseek"] == nil)
    let tools = try #require(body["tools"]?.arrayValue)
    #expect(tools[0]["type"]?.stringValue == "function")
    #expect(tools[0]["function"]?["name"]?.stringValue == "weather")
    #expect(tools[0]["function"]?["description"]?.stringValue == "Look up weather.")
    #expect(tools[0]["function"]?["parameters"]?["properties"]?["location"]?["type"]?.stringValue == "string")
    #expect(body["tool_choice"]?["type"]?.stringValue == "function")
    #expect(body["tool_choice"]?["function"]?["name"]?.stringValue == "weather")
    #expect(body["responseFormat"] == nil)
}

@Test func deepSeekLanguageInjectsSchemaInstructionAndWarning() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"{\"value\":\"ok\"}"},"finish_reason":"stop"}],"usage":{"total_tokens":5}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-reasoner")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Answer.")],
        responseFormat: .json(schema: [
            "type": "object",
            "properties": ["value": ["type": "string"]],
            "required": ["value"]
        ])
    ))

    #expect(result.warnings == [
        AIWarning(
            type: "compatibility",
            feature: "responseFormat JSON schema",
            message: "JSON response schema is injected into the system message."
        )
    ])
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let instruction = try #require(body["messages"]?[0]?["content"]?.stringValue)
    #expect(instruction.hasPrefix("Return JSON that conforms to the following schema: "))
    #expect(instruction.contains("\"type\":\"object\""))
    #expect(instruction.contains("\"value\""))
    #expect(body["response_format"]?["type"]?.stringValue == "json_object")
    #expect(body["responseFormat"] == nil)
}

@Test func deepSeekLanguageStreamsFragmentedToolCalls() async throws {
    let transport = RecordingTransport(response: sseResponse(#"""
    data: {"id":"ds-1","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}

    data: {"id":"ds-1","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{\"location\":"}}]},"finish_reason":null}]}

    data: {"id":"ds-1","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"San Francisco\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}

    data: [DONE]

    """#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-reasoner")

    var reasoning: [String] = []
    var deltas: [String] = []
    var inputLifecycle: [String] = []
    var finalCall: AIToolCall?
    var finishReason: String?
    var totalTokens: Int?
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Weather?")])) {
        switch part {
        case let .reasoningDeltaPart(_, delta, _):
            reasoning.append(delta)
        case let .toolInputStart(id, name, _, _, _, _):
            inputLifecycle.append("start:\(id):\(name)")
        case let .toolInputDelta(id, delta, _):
            inputLifecycle.append("delta:\(id):\(delta)")
        case let .toolInputEnd(id, _):
            inputLifecycle.append("end:\(id)")
        case let .toolCallDelta(_, _, argumentsDelta, _):
            deltas.append(argumentsDelta)
        case let .toolCall(call):
            finalCall = call
        case let .finishMetadata(reason, usage, _):
            finishReason = reason
            totalTokens = usage?.totalTokens
        default:
            break
        }
    }

    let call = try #require(finalCall)
    #expect(reasoning == ["think"])
    #expect(deltas == ["{\"location\":", "\"San Francisco\"}"])
    #expect(inputLifecycle == [
        "start:call_weather:weather",
        "delta:call_weather:{\"location\":",
        "delta:call_weather:\"San Francisco\"}",
        "end:call_weather"
    ])
    #expect(call.id == "call_weather")
    #expect(call.name == "weather")
    #expect(try decodeJSONBody(Data(call.arguments.utf8))["location"]?.stringValue == "San Francisco")
    #expect(finishReason == "tool-calls")
    #expect(totalTokens == 13)
}

@Test func deepSeekLanguageStreamRequiresFirstToolDeltaIDAndNameLikeUpstream() async throws {
    let transport = RecordingTransport(response: sseResponse(#"""
    data: {"id":"ds-1","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"weather","arguments":"{}"}}]},"finish_reason":null}]}

    data: [DONE]

    """#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-reasoner")

    await #expect(throws: AIError.invalidResponse(provider: "deepseek.chat", message: "Expected 'id' to be a string.")) {
        for try await _ in model.stream(LanguageModelRequest(messages: [.user("Weather?")])) {}
    }
}

@Test func deepSeekV4AssistantMessagesIncludeEmptyReasoningContent() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"ok","reasoning_content":"r"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-v4")

    let result = try await model.generate(LanguageModelRequest(messages: [
        .assistant("Previous answer", reasoning: "prior hidden chain"),
        .user("Continue")
    ]))

    #expect(result.text == "ok")
    #expect(result.reasoning == "r")
    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["messages"]?[0]?["role"]?.stringValue == "assistant")
    #expect(body["messages"]?[0]?["reasoning_content"]?.stringValue == "prior hidden chain")
}

@Test func deepSeekReasonerDropsPriorAssistantReasoningBeforeLastUserMessage() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-reasoner")

    _ = try await model.generate(LanguageModelRequest(messages: [
        .assistant("Earlier answer", reasoning: "drop this reasoning"),
        .user("Continue")
    ]))

    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["messages"]?[0]?["role"]?.stringValue == "assistant")
    #expect(body["messages"]?[0]?["reasoning_content"] == nil)
}

@Test func deepSeekV4AssistantMessagesBackfillEmptyReasoningContent() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-v4")

    _ = try await model.generate(LanguageModelRequest(messages: [.assistant("Previous answer"), .user("Continue")]))

    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["messages"]?[0]?["role"]?.stringValue == "assistant")
    #expect(body["messages"]?[0]?["reasoning_content"]?.stringValue == "")
}

@Test func deepSeekCurrentOptionsNamesPrefixAndStrictToolsMatchUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"object":"chat.completion","system_fingerprint":"fp_123","choices":[{"index":2,"message":{"role":"assistant","content":"done"},"finish_reason":"stop","logprobs":{"content":[]}}],"usage":{"prompt_tokens":4,"completion_tokens":1,"completion_tokens_details":{"reasoning_tokens":3},"future_usage":{"nested":true}}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        baseURL: "https://api.deepseek.com/beta",
        transport: transport
    ))
    let model = try provider.languageModel("deepseek-chat")
    let result = try await model.generate(LanguageModelRequest(
        messages: [
            AIMessage(role: .system, content: [.text("Guide")], providerMetadata: ["deepseek": ["name": "guide"]]),
            AIMessage(role: .user, content: [.text("Continue")], providerMetadata: ["deepseek": ["name": "alice"]]),
            AIMessage(role: .assistant, content: [.text("Once")], providerMetadata: ["deepseek": ["name": "writer", "prefix": true]])
        ],
        tools: ["lookup": ["type": "object", "strict": true, "properties": [:]]],
        providerOptions: ["deepseek": ["logprobs": true, "topLogprobs": 4, "userId": "user_123"]]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["messages"]?[0]?["name"]?.stringValue == "guide")
    #expect(body["messages"]?[1]?["name"]?.stringValue == "alice")
    #expect(body["messages"]?[2]?["name"]?.stringValue == "writer")
    #expect(body["messages"]?[2]?["prefix"]?.boolValue == true)
    #expect(body["logprobs"]?.boolValue == true)
    #expect(body["top_logprobs"]?.intValue == 4)
    #expect(body["user_id"]?.stringValue == "user_123")
    #expect(body["tools"]?[0]?["function"]?["strict"]?.boolValue == true)
    #expect(result.usage?.outputTextTokens == 0)
    #expect(result.usage?.rawValue?["future_usage"]?["nested"]?.boolValue == true)
    #expect(result.providerMetadata["deepseek"]?["responseObject"]?.stringValue == "chat.completion")
    #expect(result.providerMetadata["deepseek"]?["choiceIndex"]?.intValue == 2)
    #expect(result.providerMetadata["deepseek"]?["messageRole"]?.stringValue == "assistant")
    #expect(result.providerMetadata["deepseek"]?["logprobs"]?["content"]?.arrayValue?.isEmpty == true)
    #expect(result.providerMetadata["deepseek"]?["systemFingerprint"]?.stringValue == "fp_123")
}

@Test func deepSeekRejectsCurrentInvalidOptionsAndStrictToolsOutsideBeta() async throws {
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        transport: RecordingTransport(responses: [])
    ))
    let model = try provider.languageModel("deepseek-chat")

    await #expect(throws: AIError.self) {
        _ = try await model.generate(LanguageModelRequest(
            messages: [.user("Hi")],
            providerOptions: ["deepseek": ["userId": "not valid"]]
        ))
    }
    await #expect(throws: AIError.self) {
        _ = try await model.generate(LanguageModelRequest(
            messages: [.user("Hi")],
            providerOptions: ["deepseek": ["topLogprobs": 21]]
        ))
    }
    await #expect(throws: AIError.self) {
        _ = try await model.generate(LanguageModelRequest(
            messages: [.user("Hi")],
            tools: ["lookup": ["type": "object", "strict": true, "properties": [:]]]
        ))
    }
}

@Test func deepSeekStrictToolValidationIgnoresProviderDefinedTools() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        baseURL: "https://api.deepseek.com/beta",
        transport: transport
    ))
    let model = try provider.languageModel("deepseek-chat")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        tools: [
            "lookup": ["type": "object", "strict": true, "properties": [:]],
            "deepseek.search": ["type": "provider", "id": "deepseek.search"]
        ]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["tools"]?.arrayValue?.count == 1)
    #expect(body["tools"]?[0]?["function"]?["strict"]?.boolValue == true)
    #expect(result.warnings.contains(AIWarning(
        type: "unsupported",
        feature: "provider-defined tool deepseek.search"
    )))

    await #expect(throws: AIError.self) {
        _ = try await model.generate(LanguageModelRequest(
            messages: [.user("Hi")],
            tools: [
                "strict": ["type": "object", "strict": true, "properties": [:]],
                "ordinary": ["type": "object", "properties": [:]],
                "deepseek.search": ["type": "provider", "id": "deepseek.search"]
            ]
        ))
    }
    #expect((await transport.requests()).count == 1)
}

@Test func deepSeekStreamingAccumulatesContentAndReasoningLogprobsAcrossChunks() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think"},"finish_reason":null,"logprobs":{"content":[{"token":"A","logprob":-0.1}],"reasoning_content":[{"token":"R1","logprob":-0.2}]}}]}

    data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop","logprobs":{"content":[{"token":"B","logprob":-0.3}],"reasoning_content":[{"token":"R2","logprob":-0.4}]}}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}

    data: [DONE]

    """))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-v4-flash")

    var finalMetadata: [String: JSONValue] = [:]
    for try await part in model.stream(LanguageModelRequest(
        messages: [.user("Hi")],
        providerOptions: ["deepseek": ["logprobs": true, "topLogprobs": 1]]
    )) {
        if case let .finishMetadata(_, _, providerMetadata) = part {
            finalMetadata = providerMetadata
        }
    }

    let content = try #require(finalMetadata["deepseek"]?["logprobs"]?["content"]?.arrayValue)
    let reasoning = try #require(finalMetadata["deepseek"]?["logprobs"]?["reasoning_content"]?.arrayValue)
    #expect(content.compactMap { $0["token"]?.stringValue } == ["A", "B"])
    #expect(reasoning.compactMap { $0["token"]?.stringValue } == ["R1", "R2"])
    #expect(finalMetadata["deepseek"]?["responseObject"]?.stringValue == "chat.completion.chunk")
    #expect(finalMetadata["deepseek"]?["choiceIndex"]?.intValue == 0)
    #expect(finalMetadata["deepseek"]?["messageRole"]?.stringValue == "assistant")
}