import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

/// Opt-in Responses endpoint; the default DeepSeek model keeps Chat Completions behavior.
public final class DeepSeekResponsesLanguageModel: LanguageModel, Sendable {
    public let modelID: String
    public let providerID: String
    let config: DeepSeekModelConfiguration
    var providerOptionsName: String { providerID.split(separator: ".").first.map(String.init) ?? "deepseek" }
    public var supportedURLs: [String: [AISupportedURLPattern]] { ["image/*": [.init { $0.hasPrefix("https://") || $0.hasPrefix("http://") }]] }
    public init(modelID: String, config: DeepSeekModelConfiguration) { self.modelID = modelID; self.config = config; providerID = config.providerID }

    public func generate(_ request: LanguageModelRequest) async throws -> TextGenerationResult {
        try request.abortSignal?.throwIfAborted()
        let prepared = try deepSeekResponsesPrepared(request, modelID: modelID, providerOptionsName: providerOptionsName)
        let response = try await config.sendJSONResponse(path: "/responses", modelID: modelID, body: .object(prepared.body), headers: request.headers, abortSignal: request.abortSignal)
        try request.abortSignal?.throwIfAborted()
        let raw = response.json
        guard raw.objectValue != nil else { throw AIError.invalidResponse(provider: providerID, message: "Invalid DeepSeek Responses result.") }
        if raw["status"] == "failed" || raw["error"] != nil && raw["error"] != .null {
            let mapped = deepSeekResponsesFailure(raw["error"]?["code"]?.stringValue)
            var error = AIAPICallError(provider: providerID, url: response.response.url?.absoluteString,
                requestBody: .object(prepared.body), statusCode: mapped.0, responseHeaders: response.response.headers,
                responseBody: String(decoding: response.response.body, as: UTF8.self), isRetryable: mapped.1)
            error.message = raw["error"]?["message"]?.stringValue ?? "DeepSeek response failed"
            throw AIError.apiCall(error)
        }
        if let output = raw["output"], output != .null, output.arrayValue == nil {
            throw AIError.invalidResponse(provider: providerID, message: "DeepSeek response output must be an array.")
        }
        let items = raw["output"]?.arrayValue ?? []
        var content: [AIResultContentPart] = [], text = "", reasoning = "", logprobs: [JSONValue] = []
        var hasFunctionCall = false
        for item in items {
            try deepSeekResponsesValidateItem(item, providerID: providerID)
            switch item["type"]?.stringValue {
            case "reasoning":
                let value = (item["content"]?.arrayValue ?? []).compactMap { $0["text"]?.stringValue }.joined()
                reasoning += value
                content.append(.reasoning(value, providerMetadata: item["id"]?.stringValue.map { [providerOptionsName: ["itemId": .string($0)]] } ?? [:]))
            case "message":
                for part in item["content"]?.arrayValue ?? [] {
                    logprobs += part["logprobs"]?.arrayValue ?? []
                    if let value = part["text"]?.stringValue, !value.isEmpty { text += value; content.append(.text(value)) }
                }
            case "function_call":
                hasFunctionCall = true
                content.append(.toolCall(.init(id: item["call_id"]?.stringValue ?? "", name: item["name"]?.stringValue ?? "", arguments: item["arguments"]?.stringValue ?? "")))
            case "web_search_call": content += deepSeekResponsesWebSearch(item, name: prepared.webSearchName, providerOptionsName: providerOptionsName)
            default: break
            }
        }
        return TextGenerationResult(text: text, content: content, reasoning: reasoning,
            finishReason: deepSeekResponsesFinish(raw, hasFunctionCall: hasFunctionCall), usage: deepSeekResponsesUsage(raw["usage"]),
            providerMetadata: deepSeekResponsesMetadata(id: raw["id"], usage: raw["usage"], logprobs: logprobs, providerOptionsName: providerOptionsName), rawValue: raw, warnings: prepared.warnings,
            requestMetadata: .init(body: .object(prepared.body), headers: request.headers, url: config.baseURL + "/responses"),
            responseMetadata: .init(id: raw["id"]?.stringValue, timestamp: raw["created_at"]?.doubleValue.map { Date(timeIntervalSince1970: $0) },
                                    modelID: raw["model"]?.stringValue ?? modelID, headers: response.response.headers))
    }
    public func stream(_ request: LanguageModelRequest) -> AsyncThrowingStream<LanguageStreamPart, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try request.abortSignal?.throwIfAborted()
                    var prepared = try deepSeekResponsesPrepared(request, modelID: modelID, providerOptionsName: providerOptionsName)
                    prepared.body["stream"] = true
                    let http = try config.request(path: "/responses", modelID: modelID, body: .object(prepared.body), headers: request.headers, abortSignal: request.abortSignal)
                    let response = try await config.streamRequest(http)
                    try request.abortSignal?.throwIfAborted()
                    guard (200..<300).contains(response.statusCode) else { throw config.httpStatusError(try await bufferedHTTPResponse(from: response, request: http)) }
                    var state = DeepSeekResponsesStreamState(providerID: providerID, modelID: modelID, webSearchName: prepared.webSearchName, headers: response.headers)
                    continuation.yield(.streamStart(warnings: prepared.warnings))
                    for try await event in serverSentEvents(from: response.body) {
                        try request.abortSignal?.throwIfAborted()
                        if event.data == "[DONE]" { break }
                        do {
                            let raw = try decodeJSONBody(Data(event.data.utf8))
                            if request.includeRawChunks { continuation.yield(.raw(raw)) }
                            for part in try state.apply(raw) { continuation.yield(part) }
                        } catch {
                            state.finishReason = "error"
                            continuation.yield(.error(message: error.localizedDescription))
                        }
                    }
                    for part in state.finish() { continuation.yield(part) }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}

func deepSeekResponsesWebSearch(_ item: JSONValue, name: String, providerOptionsName: String = "deepseek") -> [AIResultContentPart] {
    let id = item["id"]?.stringValue ?? ""
    let call = AIToolCall(id: id, name: name, arguments: "{}", providerExecuted: true, providerMetadata: [providerOptionsName: ["webSearchCall": item]])
    let result = AIToolResult(toolCallID: id, toolName: name,
        result: .object(["status": item["status"].flatMap { value -> JSONValue? in value == .null ? nil : value }, "action": item["action"].flatMap { value -> JSONValue? in value == .null ? nil : value }]), providerExecuted: true)
    return [.toolCall(call), .toolResult(result)]
}
func deepSeekResponsesUsage(_ value: JSONValue?) -> TokenUsage? {
    guard let value, value != .null else { return TokenUsage() }
    let input = value["input_tokens"]?.intValue ?? 0, output = value["output_tokens"]?.intValue ?? 0
    let cache = value["input_tokens_details"]?["cached_tokens"]?.intValue ?? 0
    let reasoning = value["output_tokens_details"]?["reasoning_tokens"]?.intValue ?? 0
    return .init(inputTokens: input, outputTokens: output, totalTokens: value["total_tokens"]?.intValue,
        inputTokensNoCache: input - cache, inputTokensCacheRead: cache, outputTextTokens: max(0, output - reasoning), outputReasoningTokens: reasoning, rawValue: value)
}
func deepSeekResponsesMetadata(id: JSONValue?, usage: JSONValue?, logprobs: [JSONValue], providerOptionsName: String = "deepseek") -> [String: JSONValue] {
    var fields: [String: JSONValue] = [:]
    fields["responseId"] = id.flatMap { value -> JSONValue? in value == .null ? nil : value }
    if let server = usage?["server_tool_use"], server != .null { fields["serverToolUse"] = ["toolSearch": server["tool_search"] ?? 0, "toolFetch": server["tool_fetch"] ?? 0] }
    if !logprobs.isEmpty { fields["logprobs"] = .array(logprobs) }
    return [providerOptionsName: .object(fields)]
}
func deepSeekResponsesFinish(_ raw: JSONValue, hasFunctionCall: Bool) -> String {
    switch raw["status"]?.stringValue {
    case "completed": return hasFunctionCall ? "tool-calls" : "stop"
    case "incomplete": return raw["incomplete_details"]?["reason"] == "max_output_tokens" ? "length" : raw["incomplete_details"]?["reason"] == "content_filter" ? "content-filter" : "other"
    case "failed": return "error"
    default: return "other"
    }
}
func deepSeekResponsesFailure(_ code: String?) -> (Int, Bool) {
    switch code {
    case "rate_limit_exceeded", "rate_limit_error": (429, true)
    case "insufficient_quota": (429, false)
    case "authentication_error", "invalid_api_key": (401, false)
    case "permission_error": (403, false)
    case "bad_request", "context_length_exceeded", "invalid_request_error": (400, false)
    case "timeout", "timeout_error": (504, true)
    default: (500, true)
    }
}
func deepSeekResponsesValidateItem(_ item: JSONValue, providerID: String) throws {
    guard let type = item["type"]?.stringValue else { throw AIError.invalidResponse(provider: providerID, message: "Responses item has no type.") }
    let required: [String]
    switch type { case "function_call": required = ["call_id", "name"]; case "web_search_call": required = ["id"]; default: required = [] }
    guard required.allSatisfy({ item[$0]?.stringValue != nil }) else { throw AIError.invalidResponse(provider: providerID, message: "Invalid Responses output item.") }
}
