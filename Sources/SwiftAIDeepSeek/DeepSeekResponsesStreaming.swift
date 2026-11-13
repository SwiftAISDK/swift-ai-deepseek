import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

struct DeepSeekResponsesStreamState {
    let providerID: String
    let modelID: String
    let webSearchName: String
    let headers: [String: String]
    var providerOptionsName: String { providerID.split(separator: ".").first.map(String.init) ?? "deepseek" }
    var finishReason = "interrupted"
    private var responseID: JSONValue?
    private var usage: JSONValue?
    private var logprobs: [JSONValue] = []
    private var openText: [String] = [], openReasoning: [String] = []
    private struct FunctionCall { var id: String; var name: String; var arguments: String }
    private var calls: [String: FunctionCall] = [:]
    private var hasFunctionCall = false
    init(providerID: String, modelID: String, webSearchName: String, headers: [String: String]) {
        self.providerID = providerID; self.modelID = modelID; self.webSearchName = webSearchName; self.headers = headers
    }
    mutating func apply(_ raw: JSONValue) throws -> [LanguageStreamPart] {
        switch raw["type"]?.stringValue {
        case "response.created":
            let response = raw["response"] ?? [:]; responseID = response["id"]
            return [.responseMetadata(.init(id: response["id"]?.stringValue, timestamp: response["created_at"]?.doubleValue.map { Date(timeIntervalSince1970: $0) }, modelID: response["model"]?.stringValue ?? modelID, headers: headers))]
        case "response.output_item.added":
            guard let item = raw["item"] else { throw invalid() }; try deepSeekResponsesValidateItem(item, providerID: providerID)
            let id = item["id"]?.stringValue
            switch item["type"]?.stringValue {
            case "message": if let id { openText.append(id); return [.textStart(id: id)] }
            case "reasoning": if let id { openReasoning.append(id); return [.reasoningStart(id: id, providerMetadata: [providerOptionsName: ["itemId": .string(id)]])] }
            case "function_call":
                let callID = item["call_id"]?.stringValue ?? "", name = item["name"]?.stringValue ?? ""
                calls[id ?? callID] = .init(id: callID, name: name, arguments: item["arguments"]?.stringValue ?? "")
                return [.toolInputStart(id: callID, name: name)]
            default: break
            }
        case "response.output_text.delta", "response.reasoning_text.delta":
            guard let id = raw["item_id"]?.stringValue, let delta = raw["delta"]?.stringValue else { throw invalid() }
            if raw["type"] == "response.output_text.delta" {
                logprobs += raw["logprobs"]?.arrayValue ?? []
                let start = !openText.contains(id)
                if start { openText.append(id) }
                return (start ? [.textStart(id: id)] : []) + [.textDeltaPart(id: id, delta: delta)]
            }
            let start = !openReasoning.contains(id)
            if start { openReasoning.append(id) }
            return (start ? [.reasoningStart(id: id, providerMetadata: [providerOptionsName: ["itemId": .string(id)]])] : []) + [.reasoningDeltaPart(id: id, delta: delta)]
        case "response.function_call_arguments.delta":
            guard let id = raw["item_id"]?.stringValue, let delta = raw["delta"]?.stringValue else { throw invalid() }
            guard var call = calls[id] else { return [] }
            call.arguments += delta; calls[id] = call
            return [.toolInputDelta(id: call.id, delta: delta)]
        case "response.output_item.done":
            guard let item = raw["item"] else { throw invalid() }; try deepSeekResponsesValidateItem(item, providerID: providerID)
            let id = item["id"]?.stringValue
            switch item["type"]?.stringValue {
            case "message":
                if let id, openText.contains(id) { openText.removeAll { $0 == id }; return [.textEnd(id: id)] }
            case "reasoning":
                if let id, openReasoning.contains(id) { openReasoning.removeAll { $0 == id }; return [.reasoningEnd(id: id, providerMetadata: [providerOptionsName: ["itemId": .string(id)]])] }
            case "function_call":
                let callID = item["call_id"]?.stringValue ?? "", name = item["name"]?.stringValue ?? ""
                let tracked = calls.removeValue(forKey: id ?? callID); hasFunctionCall = true
                return (tracked == nil ? [.toolInputStart(id: callID, name: name)] : []) + [
                    .toolInputEnd(id: callID), .toolCall(.init(id: callID, name: name, arguments: item["arguments"]?.stringValue ?? tracked?.arguments ?? ""))]
            case "web_search_call":
                return deepSeekResponsesWebSearch(item, name: webSearchName, providerOptionsName: providerOptionsName).compactMap { part in
                    switch part { case let .toolCall(call): return .toolCall(call); case let .toolResult(result): return .toolResult(result); default: return nil }
                }
            default: break
            }
        case "response.completed", "response.incomplete", "response.failed":
            guard let response = raw["response"], response.objectValue != nil else { throw invalid() }
            usage = response["usage"]; responseID = response["id"] ?? responseID
            finishReason = deepSeekResponsesFinish(response, hasFunctionCall: hasFunctionCall)
            if raw["type"] == "response.failed" {
                let mapped = deepSeekResponsesFailure(response["error"]?["code"]?.stringValue)
                return [.providerError(.init(message: response["error"]?["message"]?.stringValue ?? "DeepSeek response failed",
                    type: "response.failed", code: response["error"]?["code"], statusCode: mapped.0, isRetryable: mapped.1, data: raw))]
            }
        case "error":
            finishReason = "error"
            return [.providerError(.init(message: raw["message"]?.stringValue ?? "DeepSeek stream error", type: "error", code: raw["code"], data: raw))]
        default: break
        }
        return []
    }
    mutating func finish() -> [LanguageStreamPart] {
        var parts = openReasoning.map { LanguageStreamPart.reasoningEnd(id: $0) }
        parts += openText.map { .textEnd(id: $0) }; openReasoning = []; openText = []
        parts.append(.finishMetadata(reason: finishReason, usage: deepSeekResponsesUsage(usage), providerMetadata: deepSeekResponsesMetadata(id: responseID, usage: usage, logprobs: logprobs, providerOptionsName: providerOptionsName)))
        return parts
    }
    private func invalid() -> AIError { .invalidResponse(provider: providerID, message: "Invalid DeepSeek Responses stream event.") }
}
