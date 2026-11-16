import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


// A hosted endpoint with unknown strict-tool support forwards the caller's
// choice; only explicit native support applies DeepSeek's beta-mode policy.
func deepSeekTools(from tools: [String: JSONValue], supportsStrictToolCalls: Bool? = true) throws -> DeepSeekPreparedTools {
    var warnings: [AIWarning] = []
    let functionTools = tools.filter { _, schema in
        let object = schema.objectValue
        return object?["type"]?.stringValue != "provider" && object?["id"]?.stringValue == nil
    }
    let strictValues = functionTools.compactMap { _, schema -> Bool? in
        schema.objectValue?["strict"]?.boolValue
    }
    if strictValues.contains(true), supportsStrictToolCalls == false {
        throw AIError.invalidArgument(argument: "tools", message: "DeepSeek strict tool calls require a beta base URL ending in /beta.")
    }
    if supportsStrictToolCalls == true,
       strictValues.contains(true),
       strictValues.count != functionTools.count || strictValues.contains(false) {
        throw AIError.invalidArgument(argument: "tools", message: "DeepSeek strict mode requires every function tool in the request to set strict true.")
    }
    let values: [JSONValue] = tools.compactMap { name, schema in
        let object = schema.objectValue
        if object?["type"]?.stringValue == "provider" || object?["id"]?.stringValue != nil {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "provider-defined tool \(object?["id"]?.stringValue ?? name)"
            ))
            return nil
        }
        // Upstream `deepseek-prepare-tools` sends `tool.inputSchema` as
        // `parameters`; description and strict stay at the function level.
        var function: [String: JSONValue] = [
            "name": .string(name),
            "parameters": aiFunctionToolInputSchema(schema)
        ]
        if let description = aiFunctionToolDescription(schema) {
            function["description"] = .string(description)
        }
        if let strict = object?["strict"], strict != .null {
            function["strict"] = strict
        }
        return .object([
            "type": .string("function"),
            "function": .object(function)
        ])
    }
    return DeepSeekPreparedTools(tools: values, warnings: warnings)
}

func deepSeekToolChoice(from value: JSONValue?) -> JSONValue? {
    if let string = value?.stringValue {
        switch string {
        case "auto", "none", "required":
            return .string(string)
        default:
            return nil
        }
    }
    guard let object = value?.objectValue else { return nil }
    switch object["type"]?.stringValue {
    case "auto", "none", "required":
        return object["type"]
    case "tool":
        guard let toolName = object["toolName"]?.stringValue ?? object["tool_name"]?.stringValue else {
            return nil
        }
        return .object([
            "type": .string("function"),
            "function": .object(["name": .string(toolName)])
        ])
    default:
        return nil
    }
}

func deepSeekToolChoiceWarnings(from value: JSONValue?) -> [AIWarning] {
    guard let value else { return [] }
    if let string = value.stringValue {
        switch string {
        case "auto", "none", "required":
            return []
        default:
            return [AIWarning(type: "unsupported", feature: "tool choice type: \(string)")]
        }
    }
    guard let object = value.objectValue else { return [] }
    let type = object["type"]?.stringValue
    switch type {
    case "auto", "none", "required":
        return []
    case "tool":
        return (object["toolName"]?.stringValue ?? object["tool_name"]?.stringValue) == nil
            ? [AIWarning(type: "unsupported", feature: "tool choice type: tool")]
            : []
    case let type?:
        return [AIWarning(type: "unsupported", feature: "tool choice type: \(type)")]
    case nil:
        return [AIWarning(type: "unsupported", feature: "tool choice type: undefined")]
    }
}

func deepSeekJSONString(_ value: JSONValue) -> String? {
    guard let data = try? encodeJSONBody(value) else { return nil }
    return String(data: data, encoding: .utf8)
}

func deepSeekUsage(from raw: JSONValue) -> TokenUsage? {
    guard let usage = raw["usage"], usage != .null else { return TokenUsage() }
    let inputTokens = usage["prompt_tokens"]?.intValue ?? usage["input_tokens"]?.intValue ?? 0
    let outputTokens = usage["completion_tokens"]?.intValue ?? usage["output_tokens"]?.intValue ?? 0
    let cacheReadTokens = usage["prompt_cache_hit_tokens"]?.intValue
        ?? usage["prompt_tokens_details"]?["cached_tokens"]?.intValue ?? 0
    let reasoningTokens = usage["completion_tokens_details"]?["reasoning_tokens"]?.intValue ?? 0
    let totalTokens = usage["total_tokens"]?.intValue ?? inputTokens + outputTokens
    return TokenUsage(
        inputTokens: inputTokens,
        outputTokens: outputTokens,
        totalTokens: totalTokens,
        inputTokensNoCache: inputTokens - cacheReadTokens,
        inputTokensCacheRead: cacheReadTokens,
        inputTokensCacheWrite: nil,
        outputTextTokens: max(0, outputTokens - reasoningTokens),
        outputReasoningTokens: reasoningTokens,
        rawValue: usage
    )
}

/// Mirrors upstream's loose usage schema: known token fields must be numbers
/// (or null) and the detail containers objects; unknown fields pass through.
func deepSeekUsageIsValid(_ raw: JSONValue) -> Bool {
    guard let usage = raw["usage"], usage != .null else { return true }
    guard usage.objectValue != nil else { return false }
    func isNumberOrNull(_ value: JSONValue?) -> Bool {
        guard let value, value != .null else { return true }
        if case .number = value { return true }
        return false
    }
    for key in ["prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "total_tokens"]
        where !isNumberOrNull(usage[key]) {
        return false
    }
    for (container, key) in [("prompt_tokens_details", "cached_tokens"), ("completion_tokens_details", "reasoning_tokens")] {
        guard let details = usage[container], details != .null else { continue }
        guard details.objectValue != nil, isNumberOrNull(details[key]) else { return false }
    }
    return true
}

func deepSeekToolResultContent(_ result: AIToolResult) -> String {
    let output = result.modelOutput ?? result.result
    if let text = output.stringValue {
        return text
    }
    guard let object = output.objectValue, let type = object["type"]?.stringValue else {
        return deepSeekJSONString(output) ?? ""
    }
    switch type {
    case "text", "error-text":
        return object["value"]?.stringValue ?? ""
    case "execution-denied":
        return object["reason"]?.stringValue ?? "Tool call execution denied."
    case "json", "error-json":
        return deepSeekJSONString(object["value"] ?? .object([:])) ?? ""
    case "content":
        return deepSeekJSONString(object["value"] ?? JSONValue.array([JSONValue]())) ?? ""
    default:
        return deepSeekJSONString(output) ?? ""
    }
}

func deepSeekToolCalls(from value: JSONValue?) -> [AIToolCall] {
    value?.arrayValue?.enumerated().compactMap { index, item in
        guard let name = item["function"]?["name"]?.stringValue else { return nil }
        return AIToolCall(
            id: resolvedToolCallID(item["id"]?.stringValue, whenMissing: generateId()),
            name: name,
            arguments: item["function"]?["arguments"]?.stringValue ?? "",
            rawValue: item
        )
    } ?? []
}

func deepSeekFinishReason(_ reason: String?) -> String? {
    switch reason {
    case "stop":
        return "stop"
    case "length":
        return "length"
    case "content_filter":
        return "content-filter"
    case "tool_calls":
        return "tool-calls"
    case "insufficient_system_resource":
        return "error"
    default:
        return "other"
    }
}

func deepSeekProviderMetadata(from raw: JSONValue, providerOptionsName: String = "deepseek") -> [String: JSONValue] {
    var metadata: [String: JSONValue] = [:]
    if let hit = raw["usage"]?["prompt_cache_hit_tokens"] {
        metadata["promptCacheHitTokens"] = hit
    }
    if let miss = raw["usage"]?["prompt_cache_miss_tokens"] {
        metadata["promptCacheMissTokens"] = miss
    }
    if let object = raw["object"], object != .null { metadata["responseObject"] = object }
    if let choice = raw["choices"]?[0] {
        if let index = choice["index"], index != .null { metadata["choiceIndex"] = index }
        if let role = choice["message"]?["role"], role != .null { metadata["messageRole"] = role }
        if let calls = choice["message"]?["tool_calls"]?.arrayValue {
            metadata["toolCallTypes"] = .array(calls.compactMap { call in
                guard let type = call["type"], type != .null else { return nil }
                return type
            })
        }
        if let logprobs = choice["logprobs"], logprobs != .null { metadata["logprobs"] = logprobs }
    }
    if let fingerprint = raw["system_fingerprint"], fingerprint != .null { metadata["systemFingerprint"] = fingerprint }
    // Upstream always returns the provider namespace, even when it is empty.
    return [providerOptionsName: .object(metadata)]
}

/// Accumulates streamed chunk metadata the way upstream's stream transform
/// does: latest non-null scalar values, tool-call types ordered by index and
/// concatenated content/reasoning logprobs.
struct DeepSeekStreamMetadata {
    private var promptCacheHitTokens: JSONValue?
    private var promptCacheMissTokens: JSONValue?
    private var responseObject: JSONValue?
    private var choiceIndex: JSONValue?
    private var messageRole: JSONValue?
    private var systemFingerprint: JSONValue?
    private var toolCallTypes: [Int: JSONValue] = [:]
    private var contentLogprobs: [JSONValue] = []
    private var reasoningLogprobs: [JSONValue] = []

    mutating func apply(_ raw: JSONValue) {
        if let usage = raw["usage"], usage != .null {
            promptCacheHitTokens = usage["prompt_cache_hit_tokens"].flatMap { $0 == .null ? nil : $0 }
            promptCacheMissTokens = usage["prompt_cache_miss_tokens"].flatMap { $0 == .null ? nil : $0 }
        }
        if let object = raw["object"], object != .null { responseObject = object }
        if let fingerprint = raw["system_fingerprint"], fingerprint != .null { systemFingerprint = fingerprint }
        guard let choice = raw["choices"]?[0] else { return }
        if let index = choice["index"], index != .null { choiceIndex = index }
        if let content = choice["logprobs"]?["content"]?.arrayValue { contentLogprobs += content }
        if let reasoning = choice["logprobs"]?["reasoning_content"]?.arrayValue { reasoningLogprobs += reasoning }
        guard let delta = choice["delta"], delta != .null else { return }
        if let role = delta["role"], role != .null { messageRole = role }
        for call in delta["tool_calls"]?.arrayValue ?? [] {
            if let index = call["index"]?.intValue, let type = call["type"], type != .null {
                toolCallTypes[index] = type
            }
        }
    }

    func providerMetadata(providerOptionsName: String) -> [String: JSONValue] {
        var metadata: [String: JSONValue] = [:]
        metadata["promptCacheHitTokens"] = promptCacheHitTokens
        metadata["promptCacheMissTokens"] = promptCacheMissTokens
        metadata["responseObject"] = responseObject
        metadata["choiceIndex"] = choiceIndex
        metadata["messageRole"] = messageRole
        if !toolCallTypes.isEmpty {
            metadata["toolCallTypes"] = .array(toolCallTypes.sorted { $0.key < $1.key }.map(\.value))
        }
        if !contentLogprobs.isEmpty || !reasoningLogprobs.isEmpty {
            var logprobs: [String: JSONValue] = [:]
            if !contentLogprobs.isEmpty { logprobs["content"] = .array(contentLogprobs) }
            if !reasoningLogprobs.isEmpty { logprobs["reasoning_content"] = .array(reasoningLogprobs) }
            metadata["logprobs"] = .object(logprobs)
        }
        metadata["systemFingerprint"] = systemFingerprint
        return [providerOptionsName: .object(metadata)]
    }
}
