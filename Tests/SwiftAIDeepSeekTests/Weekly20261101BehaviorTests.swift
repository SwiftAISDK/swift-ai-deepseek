import Foundation
import Testing
import SwiftAIProvider
import SwiftAITransport
@testable import SwiftAIDeepSeek

@Suite struct Weekly20261101DeepSeekBehaviorTests {
    @Test(arguments: [("medium", "low"), ("xhigh", "high"), ("minimal", "low"), ("max", "max")], [false, true])
    func mapsPortableReasoningWithStreamWarnings(_ reasoning: (String, String), _ streaming: Bool) async throws {
        let raw = #"{"choices":[{"message":{"content":"done"},"delta":{"content":"done"},"finish_reason":"stop"}]}"#
        let transport = RecordingTransport(response: streaming ? sseResponse("data: " + raw) : jsonResponse(raw))
        let model = try DeepSeekProvider(configuration: .init(apiKey: "key", transport: transport)).languageModel("deepseek-reasoner")
        let request = LanguageModelRequest(messages: [.user("Hi")], reasoning: reasoning.0)
        var warnings: [AIWarning] = []
        if streaming { for try await part in model.stream(request) { if case let .streamStart(value) = part { warnings = value } } }
        else { warnings = try await model.generate(request).warnings }
        let body = try decodeJSONBody(try #require(await transport.requests().first?.body))
        #expect(body["reasoning_effort"] == .string(reasoning.1))
        #expect(warnings.contains { $0.type == "compatibility" && $0.feature == "reasoning" } == (reasoning.0 != reasoning.1))
    }
    @Test func explicitThinkingDoesNotSuppressPortableEffortAndProviderDefaultStaysOmitted() throws {
        var body: [String: JSONValue] = ["thinking": ["type": "enabled"]]
        let warnings = deepSeekApplyReasoning("medium", to: &body)
        #expect(body["reasoning_effort"] == "low")
        #expect(warnings.contains { $0.type == "compatibility" && $0.feature == "reasoning" })
        var omitted: [String: JSONValue] = [:]
        #expect(deepSeekApplyReasoning("provider-default", to: &omitted).isEmpty)
        #expect(omitted.isEmpty)
    }

}
