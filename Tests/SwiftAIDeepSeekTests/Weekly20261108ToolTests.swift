import Foundation
import Testing
@testable import SwiftAIDeepSeek
import SwiftAIProvider
import SwiftAITransport

@Suite("Weekly20261108 DeepSeek hosted strict tools and replay")
struct Weekly20261108ToolTests {
    @Test(arguments: [Optional<Bool>.none, true, false])
    func hostedEndpointForwardsUnknownStrictPolicy(_ companionStrict: Bool?) async throws {
        let transport = RecordingTransport(responses: [
            jsonResponse(#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#),
            sseResponse("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
        ])
        let http = AIProviderHTTPClient(providerID: "azure.deepseek", transport: transport) { _, path in
            URL(string: "https://resource.openai.azure.com/openai/v1\(path)")!
        }
        let model = DeepSeekLanguageModel(modelID: "DeepSeek-V4-Pro", config: .init(
            http: http, baseURL: "https://resource.openai.azure.com/openai/v1",
            supportsThinking: false, supportsStructuredOutputs: true
        ))
        var companion: [String: JSONValue] = ["type": "object", "properties": [:]]
        if let companionStrict { companion["strict"] = .bool(companionStrict) }
        let request = LanguageModelRequest(messages: [.user("hello")], tools: [
            "strict": ["type": "object", "properties": [:], "strict": true],
            "companion": .object(companion)
        ])
        #expect(try await model.generate(request).text == "ok")
        for try await _ in model.stream(request) {}
        let requests = await transport.requests()
        #expect(requests.count == 2)
        for sent in requests {
            let body = try JSONDecoder().decode(JSONValue.self, from: #require(sent.body))
            let functions = try #require(body["tools"]?.arrayValue).compactMap { $0["function"] }
            #expect(functions.first { $0["name"] == "strict" }?["strict"] == true)
            #expect(functions.first { $0["name"] == "companion" }?["strict"] == companionStrict.map(JSONValue.bool))
            #expect(body["thinking"] == nil)
        }
    }

    @Test(arguments: [#"{"city":"Tokyo"}"#, "null", "42", "raw arguments\n\"quoted\"", ""])
    func toolArgumentsPreserveJSONAndQuoteMalformedText(_ input: String) async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#))
        let provider = try DeepSeekProvider(configuration: .init(apiKey: "key", transport: transport))
        _ = try await provider.languageModel("deepseek-chat").generate(.init(messages: [
            .init(role: .assistant, content: [.toolCall(.init(id: "call", name: "lookup", arguments: input))])
        ]))
        let sent = try #require(await transport.requests().first)
        let body = try JSONDecoder().decode(JSONValue.self, from: #require(sent.body))
        let arguments = try #require(body["messages"]?[0]?["tool_calls"]?[0]?["function"]?["arguments"]?.stringValue)
        if let parsed = try? JSONDecoder().decode(JSONValue.self, from: Data(input.utf8)) {
            #expect(arguments == input)
            #expect(try JSONDecoder().decode(JSONValue.self, from: Data(arguments.utf8)) == parsed)
        } else {
            #expect(try JSONDecoder().decode(JSONValue.self, from: Data(arguments.utf8)) == .string(input))
        }
    }
}
