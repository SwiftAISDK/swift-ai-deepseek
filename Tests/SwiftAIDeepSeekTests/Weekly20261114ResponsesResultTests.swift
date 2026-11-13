import Foundation
import Testing
@testable import SwiftAIDeepSeek
import SwiftAIProvider

@Suite("Weekly20261114 DeepSeek Responses result")
struct Weekly20261114ResponsesResultTests {
    @Test func unaryPreservesInterleavedContentAndUsage() async throws {
        let raw = #"{"id":"resp","created_at":10,"model":"returned","status":"completed","output":[{"type":"reasoning","id":"rs","content":[{"text":"Think"}]},{"type":"message","content":[{"text":"Hi","logprobs":[{"token":"Hi"}]}]},{"type":"web_search_call","id":"ws","status":"completed","action":{"type":"search","queries":["q"]}},{"type":"function_call","call_id":"call","name":"lookup","arguments":"{}"}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"input_tokens_details":{"cached_tokens":2},"output_tokens_details":{"reasoning_tokens":7},"server_tool_use":{"tool_search":1}}}"#
        let transport = RecordingTransport(response: jsonResponse(raw))
        let provider = try DeepSeekProvider(configuration: .init(apiKey: "key", transport: transport))
        let result = try await provider.responses("deepseek-flash").generate(.init(messages: [], tools: ["search": DeepSeekTools.webSearch()]))
        #expect(result.text == "Hi"); #expect(result.reasoning == "Think")
        #expect(result.finishReason == "tool-calls")
        #expect(result.usage?.inputTokensNoCache == 8); #expect(result.usage?.outputTextTokens == 0)
        #expect(result.toolCalls.map(\.name) == ["search", "lookup"])
        #expect(result.toolResults.first?.result["action"]?["queries"] == ["q"])
        #expect(result.providerMetadata["deepseek"]?["serverToolUse"] == ["toolSearch": 1, "toolFetch": 0])
        #expect(result.providerMetadata["deepseek"]?["logprobs"]?[0]?["token"] == "Hi")
        #expect(result.responseMetadata.id == "resp"); #expect(result.responseMetadata.modelID == "returned")
        guard case let .reasoning(_, metadata) = result.content.first else { Issue.record("Missing reasoning content"); return }
        #expect(metadata["deepseek"]?["itemId"] == "rs")
    }
    @Test(arguments: [("max_output_tokens", "length"), ("content_filter", "content-filter"), ("other", "other")])
    func incompleteReasonsMap(_ raw: String, _ expected: String) {
        #expect(deepSeekResponsesFinish(["status": "incomplete", "incomplete_details": ["reason": .string(raw)]], hasFunctionCall: false) == expected)
    }
    @Test func successfulHTTPFailedResponseHasRetryClassification() async throws {
        let provider = try DeepSeekProvider(configuration: .init(apiKey: "key", transport: RecordingTransport(response: jsonResponse(#"{"status":"failed","error":{"code":"insufficient_quota","message":"quota"}}"#))))
        do { _ = try await provider.responses("deepseek-flash").generate(.init(messages: [])); Issue.record("Expected failed response") }
        catch AIError.apiCall(let error) { #expect(error.statusCode == 429); #expect(!error.isRetryable); #expect(error.message == "quota") }
    }
    @Test func streamTracksReasoningToolsSearchAndTerminalUsage() async throws {
        let events: [JSONValue] = [
            ["type": "response.created", "response": ["id": "resp", "model": "deepseek-flash", "created_at": 10]],
            ["type": "response.reasoning_text.delta", "item_id": "rs", "delta": "Think"],
            ["type": "response.output_item.done", "item": ["type": "reasoning", "id": "rs"]],
            ["type": "response.output_text.delta", "item_id": "m", "delta": "Hi"],
            ["type": "response.output_item.added", "item": ["type": "function_call", "id": "fc", "call_id": "call", "name": "lookup", "arguments": ""]],
            ["type": "response.function_call_arguments.delta", "item_id": "fc", "delta": "{}"],
            ["type": "response.output_item.done", "item": ["type": "function_call", "id": "fc", "call_id": "call", "name": "lookup"]],
            ["type": "response.output_item.done", "item": ["type": "web_search_call", "id": "ws", "status": "completed"]],
            ["type": "response.completed", "response": ["id": "resp", "status": "completed", "usage": ["input_tokens": 1, "output_tokens": 2]]]
        ]
        let body = try events.map { "data: " + String(decoding: try JSONEncoder().encode($0), as: UTF8.self) + "\n\n" }.joined()
        let provider = try DeepSeekProvider(configuration: .init(apiKey: "key", transport: RecordingTransport(response: sseResponse(body))))
        var parts: [LanguageStreamPart] = []
        for try await part in provider.responses("deepseek-flash").stream(.init(messages: [])) { parts.append(part) }
        #expect(parts.contains { if case .reasoningDeltaPart(id: "rs", delta: "Think", providerMetadata: _) = $0 { true } else { false } })
        #expect(parts.contains { if case let .toolCall(call) = $0 { call.id == "call" && call.arguments == "{}" } else { false } })
        #expect(parts.contains { if case let .toolResult(result) = $0 { result.toolCallID == "ws" } else { false } })
        #expect(parts.contains { if case .textEnd(id: "m", providerMetadata: _) = $0 { true } else { false } })
        guard case let .finishMetadata(reason, usage, metadata) = parts.last else { Issue.record("Missing finish"); return }
        #expect(reason == "tool-calls"); #expect(usage?.outputTokens == 2); #expect(metadata["deepseek"]?["responseId"] == "resp")
    }
    @Test func preabortedRequestsNeverReachInjectedTransport() async throws {
        let transport = RecordingTransport(response: jsonResponse("{}"))
        let model = try DeepSeekProvider(configuration: .init(apiKey: "key", transport: transport)).responses("m")
        let controller = AIAbortController(); controller.abort()
        await #expect(throws: AIAbortError.self) { try await model.generate(.init(messages: [], abortSignal: controller.signal)) }
        await #expect(throws: AIAbortError.self) {
            for try await _ in model.stream(.init(messages: [], abortSignal: controller.signal)) {}
        }
        #expect(await transport.requests().isEmpty)
    }

    @Test func interruptedAndFailedStreamsStayObservable() throws {
        var interrupted = DeepSeekResponsesStreamState(providerID: "deepseek.responses", modelID: "m", webSearchName: "search", headers: [:])
        _ = try interrupted.apply(["type": "response.output_text.delta", "item_id": "m", "delta": "partial"])
        let parts = interrupted.finish()
        #expect(parts.contains { if case .finishMetadata(reason: "interrupted", usage: _, providerMetadata: _) = $0 { true } else { false } })
        var failed = DeepSeekResponsesStreamState(providerID: "deepseek.responses", modelID: "m", webSearchName: "search", headers: [:])
        let errors = try failed.apply(["type": "response.failed", "response": ["status": "failed", "error": ["code": "server_error", "message": "boom"]]])
        #expect(errors.contains { if case let .error(message, _) = $0 { message == "boom" } else { false } })
        #expect(failed.finishReason == "error")
    }
}
