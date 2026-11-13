import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

public struct DeepSeekResponsesModelOptions: Equatable, Sendable {
    public enum Thinking: String, Sendable { case enabled, disabled }
    public enum ReasoningEffort: String, Sendable { case low, high, max }
    public var thinking: Thinking?
    public var reasoningEffort: ReasoningEffort?
    public var topLogprobs: Int?
    public var userID: String?
    public init(thinking: Thinking? = nil, reasoningEffort: ReasoningEffort? = nil, topLogprobs: Int? = nil, userID: String? = nil) {
        self.thinking = thinking; self.reasoningEffort = reasoningEffort; self.topLogprobs = topLogprobs; self.userID = userID
    }
    public var providerOptions: [String: JSONValue] {
        ["deepseek": .object(["thinking": thinking.map { .object(["type": .string($0.rawValue)]) },
            "reasoningEffort": reasoningEffort.map { .string($0.rawValue) }, "topLogprobs": topLogprobs.map(JSONValue.init), "userId": userID.map(JSONValue.string)])]
    }
}
public enum DeepSeekTools {
    public static func webSearch() -> JSONValue { ["type": "provider", "id": "deepseek.web_search", "name": "web_search", "args": [:]] }
}

struct DeepSeekResponsesPrepared {
    var body: [String: JSONValue]
    var warnings: [AIWarning]
    var webSearchName: String
}
func deepSeekResponsesPrepared(_ request: LanguageModelRequest, modelID: String, providerOptionsName: String = "deepseek") throws -> DeepSeekResponsesPrepared {
    let provided = request.providerOptions[providerOptionsName]
    if let provided, provided != .null, provided.objectValue == nil { throw deepSeekResponsesArgument("providerOptions.deepseek", "Responses options must be an object.") }
    let options = provided?.objectValue ?? [:]
    let allowed = options.filter { ["thinking", "reasoningEffort", "topLogprobs", "userId"].contains($0.key) }
    _ = try deepSeekValidateProviderOptions(allowed)
    if let thinking = options["thinking"]?["type"], !["enabled", "disabled"].contains(thinking.stringValue) { throw deepSeekResponsesArgument("thinking", "Responses thinking must be enabled or disabled.") }
    if let effort = options["reasoningEffort"], !["low", "high", "max"].contains(effort.stringValue) { throw deepSeekResponsesArgument("reasoningEffort", "Responses effort must be low, high, or max.") }
    var warnings: [AIWarning] = []
    for (feature, present) in [("topK", request.topK != nil), ("seed", request.seed != nil), ("frequencyPenalty", request.frequencyPenalty != nil),
                               ("presencePenalty", request.presencePenalty != nil), ("stopSequences", !request.stopSequences.isEmpty)] where present {
        warnings.append(.init(type: "unsupported", feature: feature))
    }
    var body: [String: JSONValue] = ["model": .string(modelID), "input": .array(try deepSeekResponsesInput(request.messages, providerOptionsName: providerOptionsName, warnings: &warnings))]
    body["max_output_tokens"] = request.maxOutputTokens.map(JSONValue.init)
    body["temperature"] = request.temperature.map(JSONValue.number); body["top_p"] = request.topP.map(JSONValue.number)
    body["top_logprobs"] = options["topLogprobs"]; body["user"] = options["userId"]
    let effort: String?
    if options["thinking"]?["type"] == "disabled" { effort = "none" }
    else if let explicit = options["reasoningEffort"]?.stringValue { effort = explicit }
    else if request.reasoning == "none" { effort = "none" }
    else { effort = mapReasoningToProviderEffort(reasoning: request.reasoning, effortMap: ["minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"], warnings: &warnings) }
    if let effort { body["reasoning"] = ["effort": .string(effort)] }
    if case let .json(schema, name, _) = request.responseFormat {
        let value: JSONValue = schema.map { ["type": "json_schema", "name": .string(name ?? "response"), "schema": $0] } ?? ["type": "json_object"]
        body["text"] = ["format": value]
    }
    var tools: [JSONValue] = [], webSearchName = "web_search"
    for (toolName, tool) in request.tools.sorted(by: { $0.key < $1.key }) {
        if tool["type"] == "provider" {
            if tool["id"] == "deepseek.web_search" { tools.append(["type": "web_search"]); webSearchName = toolName }
            else { warnings.append(.init(type: "unsupported", feature: "provider-defined tool \(tool["id"]?.stringValue ?? "unknown")")) }
        } else {
            let function = tool["function"] ?? tool
            if function["strict"] != nil || tool["strict"] != nil { warnings.append(.init(type: "unsupported", feature: "strict tool calls", message: "The DeepSeek Responses API does not support strict mode.")) }
            var parameters = function.objectValue ?? [:]
            parameters.removeValue(forKey: "strict")
            tools.append(.object(["type": "function", "name": .string(toolName), "description": function["description"], "parameters": .object(parameters)]))
        }
    }
    if !request.tools.isEmpty {
        body["tools"] = .array(tools)
        if let choice = request.toolChoice {
            let name = choice["toolName"]?.stringValue ?? choice["function"]?["name"]?.stringValue ?? (choice["type"] == "tool" ? choice["name"]?.stringValue : nil)
            if let name {
                if let selected = request.tools[name], selected["type"] == "provider" {
                    warnings.append(.init(type: "unsupported", feature: "toolChoice for provider-defined tool \(selected["id"]?.stringValue ?? "unknown")"))
                } else { body["tool_choice"] = ["type": "function", "name": .string(name)] }
            } else { body["tool_choice"] = choice.stringValue.map(JSONValue.string) ?? choice["type"] }
        }
    }
    body.merge(request.extraBody) { _, explicit in explicit }
    return .init(body: body, warnings: warnings, webSearchName: webSearchName)
}

func deepSeekResponsesInput(_ messages: [AIMessage], providerOptionsName: String = "deepseek", warnings: inout [AIWarning]) throws -> [JSONValue] {
    var input: [JSONValue] = []
    for message in messages {
        switch message.role {
        case .system: input.append(["role": "system", "content": .string(message.combinedText)])
        case .user:
            var content: [JSONValue] = []
            for part in message.content {
                if case let .text(value, _) = part { content.append(["type": "input_text", "text": .string(value)]) }
                else if let image = try deepSeekResponsesImage(part, providerOptionsName: providerOptionsName) { content.append(image) }
                else { warnings.append(.init(type: "unsupported", feature: "user message part type: \(deepSeekResponsesPartType(part))")) }
            }
            input.append(["role": "user", "content": .array(content)])
        case .assistant:
            for part in message.content {
                switch part {
                case let .text(value, _) where !value.isEmpty: input.append(["role": "assistant", "content": .string(value)])
                case .text: break
                case let .reasoning(value, metadata): input.append(.object(["type": "reasoning", "id": metadata[providerOptionsName]?["itemId"]?.stringValue.map(JSONValue.string), "content": .array([["type": "reasoning_text", "text": .string(value)]])]))
                case let .toolCall(call):
                    if call.providerExecuted {
                        if let item = call.providerMetadata[providerOptionsName]?["webSearchCall"], item["type"] == "web_search_call" { input.append(item) }
                    } else { input.append(["type": "function_call", "call_id": .string(call.id), "name": .string(call.name), "arguments": .string(call.arguments)]) }
                case .toolResult: break
                default: warnings.append(.init(type: "unsupported", feature: "assistant message part type: \(deepSeekResponsesPartType(part))"))
                }
            }
        case .tool:
            for part in message.content {
                guard case let .toolResult(result) = part else { continue }
                let raw = result.modelOutput ?? result.result
                let output: JSONValue
                if raw["type"] == "content", let parts = raw["value"]?.arrayValue {
                    var content: [JSONValue] = []
                    for item in parts {
                        if item["type"] == "text" { content.append(["type": "input_text", "text": item["text"] ?? ""]) }
                        else if let image = try deepSeekResponsesToolImage(item, providerOptionsName: providerOptionsName) { content.append(image) }
                        else { warnings.append(.init(type: "unsupported", feature: "tool result content part type: \(item["type"]?.stringValue ?? "unknown")")) }
                    }
                    output = .array(content)
                } else { output = .string(deepSeekToolResultContent(result)) }
                input.append(["type": "function_call_output", "call_id": .string(result.toolCallID), "output": output])
            }
        }
    }
    return input
}
func deepSeekResponsesImage(_ part: AIContentPart, providerOptionsName: String = "deepseek") throws -> JSONValue? {
    if case let .providerReference(type, reference, _, _) = part, topLevelMediaType(type) == "image" {
        return ["type": "input_image", "file_id": .string(try resolveProviderReference(reference, provider: "deepseek"))]
    }
    let url: String?, bytes: Data?, mediaType: String
    switch part {
    case let .imageURL(value, _): url = value; bytes = nil; mediaType = "image/*"
    case let .data(type, data, _) where topLevelMediaType(type) == "image",
         let .file(type, data, _, _) where topLevelMediaType(type) == "image": url = nil; bytes = data; mediaType = type
    default: return nil
    }
    let value = part.providerMetadata[providerOptionsName]
    if let value, value != .null, value.objectValue == nil { throw deepSeekResponsesArgument("image", "Image options must be an object.") }
    if value?["fileData"] == true { throw deepSeekResponsesArgument("fileData", "DeepSeek fileData image parts with the Responses API are unsupported.") }
    if let fileData = value?["fileData"], fileData.boolValue == nil { throw deepSeekResponsesArgument("fileData", "Expected a boolean.") }
    let detail = value?["imageDetail"]
    if let detail, !["low", "high", "original", "auto"].contains(detail.stringValue) { throw deepSeekResponsesArgument("imageDetail", "Unsupported image detail.") }
    let imageURL: String
    if let url {
        guard url.utf16.count <= 8_192 else { throw deepSeekResponsesArgument("messages", "DeepSeek image URLs must not exceed 8192 characters.") }; imageURL = url
    } else {
        let bytes = bytes ?? Data()
        var resolved = try resolveFullMediaType(mediaType: mediaType, data: bytes)
        guard ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"].contains(resolved) else { throw deepSeekResponsesArgument("mediaType", "DeepSeek supports JPEG, PNG, GIF, and WebP image inputs.") }
        if resolved == "image/jpg" { resolved = "image/jpeg" }
        imageURL = "data:\(resolved);base64,\(bytes.base64EncodedString())"
    }
    return .object(["type": "input_image", "image_url": .string(imageURL), "detail": detail])
}
private func deepSeekResponsesToolImage(_ item: JSONValue, providerOptionsName: String) throws -> JSONValue? {
    guard item["type"] == "file", let type = item["mediaType"]?.stringValue, topLevelMediaType(type) == "image" else { return nil }
    let metadata = item["providerOptions"]?.objectValue ?? [:]
    let part: AIContentPart
    switch item["data"]?["type"]?.stringValue {
    case "reference":
        let refs = item["data"]?["reference"]?.objectValue?.compactMapValues(\.stringValue) ?? [:]
        part = .providerReference(mimeType: type, reference: refs, providerMetadata: metadata)
    case "url": part = .imageURL(item["data"]?["url"]?.stringValue ?? "", providerMetadata: metadata)
    case "data":
        guard let base64 = item["data"]?["data"]?.stringValue, let bytes = Data(base64Encoded: base64) else { throw deepSeekResponsesArgument("image", "Image data must be base64.") }
        part = .data(mimeType: type, data: bytes, providerMetadata: metadata)
    default: return nil
    }
    return try deepSeekResponsesImage(part, providerOptionsName: providerOptionsName)
}
func deepSeekResponsesArgument(_ argument: String, _ message: String) -> AIError { .invalidArgument(argument: argument, message: message) }

private func deepSeekResponsesPartType(_ part: AIContentPart) -> String {
    switch part {
    case .text: "text"
    case .reasoning: "reasoning"
    case .toolCall: "tool-call"
    case .toolResult: "tool-result"
    case .toolApprovalRequest: "tool-approval-request"
    case .toolApprovalResponse: "tool-approval-response"
    case .custom: "custom"
    default: "file"
    }
}
