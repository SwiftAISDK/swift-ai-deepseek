import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func deepSeekTools(from tools: [String: JSONValue], supportsStrictToolCalls: Bool = true) throws -> DeepSeekPreparedTools {
    var warnings: [AIWarning] = []
    let functionTools = tools.filter { _, schema in
        let object = schema.objectValue
        return object?["type"]?.stringValue != "provider" && object?["id"]?.stringValue == nil
    }
    let strictValues = functionTools.compactMap { _, schema -> Bool? in
        schema.objectValue?["strict"]?.boolValue
    }
    if strictValues.contains(true), !supportsStrictToolCalls {
        throw AIError.invalidArgument(argument: "tools", message: "DeepSeek strict tool calls require a beta base URL ending in /beta.")
    }
    if supportsStrictToolCalls,
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
        var parameters = schema
        var function: [String: JSONValue] = [
            "name": .string(name),
            "parameters": parameters
        ]
        if var parameterObject = parameters.objectValue {
            if let description = parameterObject["description"]?.stringValue {
                function["description"] = .string(description)
            }
            if let strict = parameterObject.removeValue(forKey: "strict") {
                function["strict"] = strict
                parameters = .object(parameterObject)
                function["parameters"] = parameters
            }
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
    guard let usage = raw["usage"] else { return TokenUsage() }
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

func deepSeekProviderMetadata(from raw: JSONValue) -> [String: JSONValue] {
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
        if let role = choice["message"]?["role"] ?? choice["delta"]?["role"], role != .null { metadata["messageRole"] = role }
        if let logprobs = choice["logprobs"], logprobs != .null { metadata["logprobs"] = logprobs }
        if let calls = choice["message"]?["tool_calls"]?.arrayValue {
            metadata["toolCallTypes"] = .array(calls.compactMap { $0["type"] })
        }
    }
    if let fingerprint = raw["system_fingerprint"], fingerprint != .null { metadata["systemFingerprint"] = fingerprint }
    guard !metadata.isEmpty else { return [:] }
    return ["deepseek": .object(metadata)]
}

func deepSeekMergeProviderMetadata(_ source: [String: JSONValue], into target: inout [String: JSONValue]) {
    for (key, value) in source {
        if case let .object(existing) = target[key],
           case var .object(incoming) = value {
            if case let .object(existingLogprobs) = existing["logprobs"],
               case let .object(incomingLogprobs) = incoming["logprobs"] {
                var mergedLogprobs = existingLogprobs.merging(incomingLogprobs) { _, new in new }
                for logprobKey in ["content", "reasoning_content"] {
                    if case let .array(existingValues) = existingLogprobs[logprobKey],
                       case let .array(incomingValues) = incomingLogprobs[logprobKey] {
                        mergedLogprobs[logprobKey] = .array(existingValues + incomingValues)
                    }
                }
                incoming["logprobs"] = .object(mergedLogprobs)
            }
            target[key] = .object(existing.merging(incoming) { _, new in new })
        } else {
            target[key] = value
        }
    }
}