import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func deepSeekOptions(from request: LanguageModelRequest) throws -> [String: JSONValue] {
    var output = request.extraBody
    if let nested = output.removeValue(forKey: "deepseek")?.objectValue {
        output.merge(nested) { _, nested in nested }
    }
    if let value = request.providerOptions["deepseek"] {
        guard value != .null else {
            deepSeekMoveKey("reasoningEffort", to: "reasoning_effort", in: &output)
            return output
        }
        guard let nested = value.objectValue else {
            throw AIError.invalidArgument(argument: "providerOptions.deepseek", message: "DeepSeek provider options must be an object.")
        }
        for key in deepSeekProviderOptionKeys {
            output.removeValue(forKey: key)
        }
        output.merge(try deepSeekValidateProviderOptions(nested)) { _, nested in nested }
    }
    deepSeekMoveKey("reasoningEffort", to: "reasoning_effort", in: &output)
    return output
}

let deepSeekProviderOptionKeys: Set<String> = [
    "logprobs", "topLogprobs", "userId", "thinking", "reasoningEffort", "strictJsonSchema"
]

func deepSeekValidateProviderOptions(_ options: [String: JSONValue]) throws -> [String: JSONValue] {
    var output: [String: JSONValue] = [:]
    for (key, value) in options where deepSeekProviderOptionKeys.contains(key) {
        guard value != .null else {
            throw AIError.invalidArgument(argument: "providerOptions.deepseek.\(key)", message: "DeepSeek \(key) cannot be null.")
        }
        switch key {
        case "logprobs":
            guard value.boolValue != nil else {
                throw AIError.invalidArgument(argument: "providerOptions.deepseek.logprobs", message: "DeepSeek logprobs must be a boolean.")
            }
            output[key] = value
        case "topLogprobs":
            guard let number = value.doubleValue,
                  number.isFinite,
                  number.rounded(.towardZero) == number,
                  (0...20).contains(number) else {
                throw AIError.invalidArgument(argument: "providerOptions.deepseek.topLogprobs", message: "DeepSeek topLogprobs must be an integer between 0 and 20.")
            }
            output[key] = value
        case "userId":
            guard let userID = value.stringValue,
                  !userID.isEmpty,
                  userID.utf8.count <= 512,
                  userID.unicodeScalars.allSatisfy({ scalar in
                      (scalar.value >= 48 && scalar.value <= 57)
                          || (scalar.value >= 65 && scalar.value <= 90)
                          || (scalar.value >= 97 && scalar.value <= 122)
                          || scalar == "_" || scalar == "-"
                  }) else {
                throw AIError.invalidArgument(argument: "providerOptions.deepseek.userId", message: "DeepSeek userId must match ^[A-Za-z0-9_-]+$ and be at most 512 characters long.")
            }
            output[key] = value
        case "thinking":
            guard let object = value.objectValue else {
                throw AIError.invalidArgument(argument: "providerOptions.deepseek.thinking", message: "DeepSeek thinking must be an object.")
            }
            var thinking: [String: JSONValue] = [:]
            if let type = object["type"] {
                guard type != .null else {
                    throw AIError.invalidArgument(argument: "providerOptions.deepseek.thinking.type", message: "DeepSeek thinking.type cannot be null.")
                }
                guard let typeValue = type.stringValue, ["adaptive", "enabled", "disabled"].contains(typeValue) else {
                    throw AIError.invalidArgument(argument: "providerOptions.deepseek.thinking.type", message: "DeepSeek thinking.type must be adaptive, enabled, or disabled.")
                }
                thinking["type"] = type
            }
            output[key] = .object(thinking)
        case "reasoningEffort":
            guard let effort = value.stringValue, ["low", "medium", "high", "xhigh", "max"].contains(effort) else {
                throw AIError.invalidArgument(argument: "providerOptions.deepseek.reasoningEffort", message: "DeepSeek reasoningEffort must be low, medium, high, xhigh, or max.")
            }
            output[key] = value
        case "strictJsonSchema":
            guard value.boolValue != nil else {
                throw AIError.invalidArgument(argument: "providerOptions.deepseek.strictJsonSchema", message: "DeepSeek strictJsonSchema must be a boolean.")
            }
            output[key] = value
        default:
            break
        }
    }
    return output
}

func deepSeekResolvedResponseFormat(request: LanguageModelRequest, options: inout [String: JSONValue]) -> JSONValue? {
    if let responseFormat = request.responseFormat {
        options.removeValue(forKey: "responseFormat")
        return deepSeekResponseFormatJSON(responseFormat)
    }
    return options.removeValue(forKey: "responseFormat")
}

func deepSeekResponseFormatJSON(_ responseFormat: AIResponseFormat) -> JSONValue? {
    switch responseFormat {
    case .text:
        return nil
    case let .json(schema, name, description):
        return .object([
            "type": .string("json"),
            "schema": schema,
            "name": name.map(JSONValue.string),
            "description": description.map(JSONValue.string)
        ])
    }
}

func deepSeekWarnings(request: LanguageModelRequest, responseFormat: JSONValue?, supportsStructuredOutputs: Bool = false) -> [AIWarning] {
    var warnings: [AIWarning] = []
    if request.topK != nil {
        warnings.append(AIWarning(type: "unsupported", feature: "topK"))
    }
    if request.seed != nil {
        warnings.append(AIWarning(type: "unsupported", feature: "seed"))
    }
    if !supportsStructuredOutputs,
       responseFormat?["type"]?.stringValue == "json",
       responseFormat?["schema"] != nil {
        warnings.append(AIWarning(
            type: "compatibility",
            feature: "responseFormat JSON schema",
            message: "JSON response schema is injected into the system message."
        ))
    }
    return warnings
}

func deepSeekMoveKey(_ source: String, to destination: String, in values: inout [String: JSONValue]) {
    if let value = values.removeValue(forKey: source) {
        values[destination] = value
    }
}

func deepSeekReasoningEffort(_ value: String) -> String {
    switch value {
    case "minimal":
        return "low"
    case "xhigh", "max":
        return "max"
    default:
        return value
    }
}

func deepSeekApplyReasoning(_ reasoning: String?, to body: inout [String: JSONValue]) -> [AIWarning] {
    guard let reasoning, body["thinking"] == nil else { return [] }
    if reasoning == "none" {
        body["thinking"] = .object(["type": .string("disabled")])
        body.removeValue(forKey: "reasoning_effort")
        return []
    }
    body["thinking"] = .object(["type": .string("enabled")])
    var warnings: [AIWarning] = []
    if body["reasoning_effort"] == nil,
       let effort = deepSeekReasoningEffort(from: reasoning) {
        body["reasoning_effort"] = .string(effort)
        if effort != reasoning {
            warnings.append(AIWarning(
                type: "compatibility",
                feature: "reasoning",
                message: "reasoning \"\(reasoning)\" is not directly supported by this model. mapped to effort \"\(effort)\"."
            ))
        }
    }
    return warnings
}

func deepSeekReasoningEffort(from reasoning: String) -> String? {
    switch reasoning {
    case "minimal", "low":
        return "low"
    case "medium":
        return "medium"
    case "high":
        return "high"
    case "xhigh", "max":
        return "max"
    default:
        return nil
    }
}