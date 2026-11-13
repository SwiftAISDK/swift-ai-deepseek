import Foundation
import Testing
@testable import SwiftAIDeepSeek
import SwiftAIProvider

@Suite("Weekly20261114 DeepSeek Responses request")
struct Weekly20261114ResponsesRequestTests {
    @Test func requestMapsOptionsAndLeavesChatDefaultUnchanged() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"id":"r","status":"completed","output":[]}"#))
        let provider = try DeepSeekProvider(configuration: .init(apiKey: "key", transport: transport))
        let model = provider.responses("deepseek-v4-pro")
        #expect(model.providerID == "deepseek.responses")
        #expect(try provider.languageModel("deepseek-v4-pro").providerID == "deepseek.chat")
        let options = DeepSeekResponsesModelOptions(reasoningEffort: .max, topLogprobs: 4, userID: "user_1")
        var input = LanguageModelRequest(messages: [.user("Hi")], temperature: 0.5, topP: 0.8, topK: 2,
            presencePenalty: 1, frequencyPenalty: 1, seed: 3, maxOutputTokens: 100, stopSequences: ["stop"],
            responseFormat: .json(schema: ["type": "object"], name: "answer"),
            tools: ["lookup": ["type": "object", "properties": [:], "strict": true], "search": DeepSeekTools.webSearch()],
            toolChoice: ["type": "tool", "toolName": "lookup"])
        input.providerOptions = options.providerOptions
        let result = try await model.generate(input)
        let request = try #require(await transport.requests().first)
        let body = try JSONDecoder().decode(JSONValue.self, from: #require(request.body))
        #expect(request.url.path == "/responses")
        #expect(body["reasoning"] == ["effort": "max"])
        #expect(body["user"] == "user_1")
        #expect(body["top_logprobs"] == 4)
        #expect(body["max_output_tokens"] == 100)
        #expect(body["text"]?["format"] == ["type": "json_schema", "name": "answer", "schema": ["type": "object"]])
        #expect(body["tool_choice"] == ["type": "function", "name": "lookup"])
        #expect(body["tools"]?[0]?["strict"] == nil)
        #expect(Set(result.warnings.compactMap(\.feature)).isSuperset(of: ["topK", "seed", "frequencyPenalty", "presencePenalty", "stopSequences", "strict tool calls"]))
    }
    @Test(arguments: [("none", "none"), ("minimal", "low"), ("medium", "high"), ("xhigh", "max")])
    func commonReasoningMapsToResponsesEffort(_ reasoning: String, _ expected: String) throws {
        let prepared = try deepSeekResponsesPrepared(.init(messages: [], reasoning: reasoning), modelID: "deepseek-flash")
        #expect(prepared.body["reasoning"] == ["effort": .string(expected)])
    }
    @Test func disabledThinkingWinsAndProviderToolChoiceWarns() throws {
        let request = LanguageModelRequest(messages: [], reasoning: "max", tools: ["search": DeepSeekTools.webSearch()],
            toolChoice: ["type": "tool", "toolName": "search"], providerOptions: ["deepseek": ["thinking": ["type": "disabled"], "reasoningEffort": "high"]])
        let prepared = try deepSeekResponsesPrepared(request, modelID: "deepseek-flash")
        #expect(prepared.body["reasoning"] == ["effort": "none"])
        #expect(prepared.body["tool_choice"] == nil)
        #expect(prepared.warnings.contains { $0.feature == "toolChoice for provider-defined tool deepseek.web_search" })
    }
    @Test(arguments: [JSONValue.object(["userId": "space user"]), ["topLogprobs": 21], ["reasoningEffort": "medium"], ["thinking": ["type": "adaptive"]]])
    func invalidOptionsFailBeforeHTTP(_ options: JSONValue) async throws {
        let transport = RecordingTransport(response: jsonResponse("{}"))
        let provider = try DeepSeekProvider(configuration: .init(apiKey: "key", transport: transport))
        await #expect(throws: AIError.self) { try await provider.responses("deepseek-flash").generate(.init(messages: [], providerOptions: ["deepseek": options])) }
        #expect(await transport.requests().isEmpty)
    }
    @Test func orderedHistoryImagesAndSearchReplayArePreserved() throws {
        let search: JSONValue = ["type": "web_search_call", "id": "ws", "status": "completed", "action": ["type": "search", "queries": ["q"]]]
        let messages: [AIMessage] = [.system("First"), .system("Second"), .init(role: .user, content: [.text(""), .data(mimeType: "image/jpg", data: Data([1,2,3]), providerMetadata: ["deepseek": ["imageDetail": "original"]]), .providerReference(mimeType: "image/png", reference: ["deepseek": "f"])]),
            .init(role: .assistant, content: [.reasoning("think", providerMetadata: ["deepseek": ["itemId": "reason"]]), .text(""), .toolCall(.init(id: "c", name: "lookup", arguments: #"{"q":1}"#)), .toolCall(.init(id: "ws", name: "custom", arguments: "{}", providerExecuted: true, providerMetadata: ["deepseek": ["webSearchCall": search]]))]),
            .init(role: .tool, content: [.toolResult(.init(toolCallID: "c", toolName: "lookup", result: ["type": "json", "value": ["ok": true]]))])]
        var warnings: [AIWarning] = []
        let input = try deepSeekResponsesInput(messages, warnings: &warnings)
        #expect(input[0] == ["role": "system", "content": "First"])
        #expect(input[1] == ["role": "system", "content": "Second"])
        #expect(input[2]["content"]?[0] == ["type": "input_text", "text": ""])
        #expect(input[2]["content"]?[1] == ["type": "input_image", "image_url": "data:image/jpeg;base64,AQID", "detail": "original"])
        #expect(input[2]["content"]?[2] == ["type": "input_image", "file_id": "f"])
        #expect(input[3]["id"] == "reason")
        #expect(input[4]["arguments"] == #"{"q":1}"#)
        #expect(input[5] == search)
        #expect(input[6]["output"] == #"{"ok":true}"#)
        #expect(warnings.isEmpty)
    }
    @Test func imageURLBoundsAndUnsupportedFileData() throws {
        let accepted = "https://example.com/" + String(repeating: "a", count: 8192-20)
        #expect(try deepSeekResponsesImage(.imageURL(accepted))?["image_url"] == .string(accepted))
        #expect(throws: AIError.self) { try deepSeekResponsesImage(.imageURL(accepted + "a")) }
        #expect(throws: AIError.self) { try deepSeekResponsesImage(.data(mimeType: "image/png", data: Data([1]), providerMetadata: ["deepseek": ["fileData": true]])) }
    }
}
