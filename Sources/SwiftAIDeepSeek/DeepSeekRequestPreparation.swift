import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func deepSeekPreparedCall(
    for request: LanguageModelRequest,
    modelID: String,
    stream: Bool,
    supportsThinking: Bool = true,
    supportsStructuredOutputs: Bool = false,
    supportsAssistantPrefixCompletion: Bool = false,
    supportsStrictToolCalls: Bool = false,
    supportsPenaltySampling: Bool = false,
    providerOptionsName: String = "deepseek"
) throws -> DeepSeekPreparedCall {
    var options = try deepSeekOptions(from: request)
    let responseFormat = deepSeekResolvedResponseFormat(request: request, options: &options)
    let optionToolChoice = options.removeValue(forKey: "toolChoice")
    let strictJsonSchema = options.removeValue(forKey: "strictJsonSchema")
    let toolChoice = request.toolChoice ?? optionToolChoice
    let preparedMessages = try deepSeekMessages(
        request.messages,
        responseFormat: responseFormat,
        modelID: modelID,
        providerOptionsName: providerOptionsName,
        supportsAssistantPrefixCompletion: supportsAssistantPrefixCompletion,
        supportsStructuredOutputs: supportsStructuredOutputs
    )
    var body: [String: JSONValue] = [
        "model": .string(modelID),
        "messages": .array(preparedMessages.messages)
    ]
    if stream {
        body["stream"] = true
        body["stream_options"] = .object(["include_usage": true])
    }
    if let temperature = request.temperature { body["temperature"] = .number(temperature) }
    if let topP = request.topP { body["top_p"] = .number(topP) }
    if supportsPenaltySampling, let frequencyPenalty = request.frequencyPenalty { body["frequency_penalty"] = .number(frequencyPenalty) }
    if supportsPenaltySampling, let presencePenalty = request.presencePenalty { body["presence_penalty"] = .number(presencePenalty) }
    if let maxOutputTokens = request.maxOutputTokens { body["max_tokens"] = .number(Double(maxOutputTokens)) }
    if !request.stopSequences.isEmpty { body["stop"] = .array(request.stopSequences) }
    let preparedTools = try deepSeekTools(from: request.tools, supportsStrictToolCalls: supportsStrictToolCalls)
    if !preparedTools.tools.isEmpty {
        body["tools"] = .array(preparedTools.tools)
        if let toolChoice = deepSeekToolChoice(from: toolChoice) {
            body["tool_choice"] = toolChoice
        }
    }
    body.merge(options) { _, new in new }
    var optionCompatibilityWarnings: [AIWarning] = []
    if body["thinking"]?["type"]?.stringValue == "adaptive" {
        body["thinking"] = .object(["type": .string("enabled")])
        optionCompatibilityWarnings.append(AIWarning(
            type: "compatibility",
            feature: "thinking.type",
            message: "thinking.type adaptive is not a canonical DeepSeek value and was mapped to enabled."
        ))
    }
    if let effort = body["reasoning_effort"]?.stringValue,
       effort == "medium" || effort == "xhigh" {
        let mapped = effort == "medium" ? "high" : "max"
        body["reasoning_effort"] = .string(mapped)
        optionCompatibilityWarnings.append(AIWarning(
            type: "compatibility",
            feature: "reasoningEffort",
            message: "reasoningEffort \(effort) is not a canonical DeepSeek value and was mapped to \(mapped)."
        ))
    }
    if let topLogprobs = body.removeValue(forKey: "topLogprobs") {
        body["top_logprobs"] = topLogprobs
        body["logprobs"] = true
    }
    if let userID = body.removeValue(forKey: "userId") {
        body["user_id"] = userID
    }
    let reasoningWarnings = supportsThinking ? deepSeekApplyReasoning(request.reasoning, to: &body) : []
    if !supportsThinking {
        body.removeValue(forKey: "thinking")
    }
    if let responseFormat, responseFormat["type"]?.stringValue == "json" {
        if supportsStructuredOutputs, let schema = responseFormat["schema"] {
            var jsonSchema: [String: JSONValue] = [
                "schema": schema,
                "strict": strictJsonSchema ?? .bool(true),
                "name": responseFormat["name"] ?? .string("response")
            ]
            if let description = responseFormat["description"] {
                jsonSchema["description"] = description
            }
            body["response_format"] = .object([
                "type": .string("json_schema"),
                "json_schema": .object(jsonSchema)
            ])
        } else {
            body["response_format"] = .object(["type": .string("json_object")])
        }
    }

    if body["thinking"]?["type"]?.stringValue == "disabled" {
        body.removeValue(forKey: "reasoning_effort")
    }
    var samplingWarnings: [AIWarning] = []
    if !supportsPenaltySampling, request.frequencyPenalty != nil {
        samplingWarnings.append(AIWarning(type: "deprecated", feature: "frequencyPenalty", message: "frequencyPenalty is deprecated by DeepSeek and has been omitted."))
    }
    if !supportsPenaltySampling, request.presencePenalty != nil {
        samplingWarnings.append(AIWarning(type: "deprecated", feature: "presencePenalty", message: "presencePenalty is deprecated by DeepSeek and has been omitted."))
    }
    let thinkingEnabled = supportsThinking
        && body["thinking"]?["type"]?.stringValue != "disabled"
        && (body["thinking"] != nil || modelID == "deepseek-reasoner" || deepSeekIsV4Model(modelID))
    if thinkingEnabled {
        if body.removeValue(forKey: "temperature") != nil {
            samplingWarnings.append(AIWarning(type: "unsupported", feature: "temperature", message: "temperature has no effect when DeepSeek thinking is enabled."))
        }
        if body.removeValue(forKey: "top_p") != nil {
            samplingWarnings.append(AIWarning(type: "unsupported", feature: "topP", message: "topP has no effect when DeepSeek thinking is enabled."))
        }
    }
    return DeepSeekPreparedCall(
        body: body,
        warnings: preparedMessages.warnings
            + deepSeekWarnings(request: request, responseFormat: responseFormat, supportsStructuredOutputs: supportsStructuredOutputs)
            + reasoningWarnings
            + optionCompatibilityWarnings
            + preparedTools.warnings
            + samplingWarnings
            + (request.tools.isEmpty ? [] : deepSeekToolChoiceWarnings(from: toolChoice))
    )
}

func deepSeekMessages(
    _ messages: [AIMessage],
    responseFormat: JSONValue?,
    modelID: String,
    providerOptionsName: String = "deepseek",
    supportsAssistantPrefixCompletion: Bool = false,
    supportsStructuredOutputs: Bool = false
) throws -> DeepSeekPreparedMessages {
    var output: [JSONValue] = []
    var warnings: [AIWarning] = []
    let isDeepSeekV4 = deepSeekIsV4Model(modelID)
    let lastUserMessageIndex = messages.lastIndex { $0.role == .user } ?? -1

    if responseFormat?["type"]?.stringValue == "json" {
        if let schema = responseFormat?["schema"], !supportsStructuredOutputs {
            let schemaText = deepSeekJSONString(schema) ?? schema.stringValue ?? ""
            output.append(.object([
                "role": .string("system"),
                "content": .string("Return JSON that conforms to the following schema: \(schemaText)")
            ]))
        } else if responseFormat?["schema"] == nil {
            output.append(.object([
                "role": .string("system"),
                "content": .string("Return JSON.")
            ]))
        }
    }

    for (index, message) in messages.enumerated() {
        let messageOptions = try deepSeekMessageOptions(message.providerMetadata, providerOptionsName: providerOptionsName)
        if messageOptions.prefix && message.role != .assistant {
            throw AIError.invalidArgument(argument: "messages", message: "DeepSeek assistant prefix completion requires prefix true on an assistant message.")
        }
        switch message.role {
        case .system:
            var object: [String: JSONValue] = [
                "role": .string("system"),
                "content": .string(message.combinedText)
            ]
            if let name = messageOptions.name { object["name"] = .string(name) }
            output.append(.object(object))
        case .user:
            let hasImagePart = message.content.contains(where: deepSeekIsImagePart)
            if !hasImagePart {
                var text = ""
                for part in message.content {
                    if case let .text(value, _) = part {
                        text += value
                    } else {
                        warnings.append(AIWarning(type: "unsupported", feature: deepSeekUserPartFeature(part)))
                    }
                }
                var object: [String: JSONValue] = [
                    "role": .string("user"),
                    "content": .string(text)
                ]
                if let name = messageOptions.name { object["name"] = .string(name) }
                output.append(.object(object))
                continue
            }

            var content: [JSONValue] = []
            for part in message.content {
                switch part {
                case let .text(text, _):
                    content.append(.object([
                        "type": .string("text"),
                        "text": .string(text)
                    ]))
                case let .imageURL(url, providerMetadata):
                    let options = try deepSeekFilePartOptions(
                        providerMetadata,
                        providerOptionsName: providerOptionsName
                    )
                    guard url.utf16.count <= 8_192 else {
                        throw AIError.invalidArgument(
                            argument: "messages",
                            message: "DeepSeek image URLs must not exceed 8192 characters."
                        )
                    }
                    guard !options.fileData else {
                        throw AIError.invalidArgument(
                            argument: "messages",
                            message: "DeepSeek `fileData` image parts require inline data, not a URL."
                        )
                    }
                    content.append(deepSeekImageURLPart(url, detail: options.imageDetail))
                case let .data(mimeType, data, providerMetadata)
                    where topLevelMediaType(mimeType) == "image":
                    content.append(try deepSeekInlineImagePart(
                        mimeType: mimeType,
                        data: data,
                        filename: nil,
                        providerMetadata: providerMetadata,
                        providerOptionsName: providerOptionsName
                    ))
                case let .file(mimeType, data, filename, providerMetadata)
                    where topLevelMediaType(mimeType) == "image":
                    content.append(try deepSeekInlineImagePart(
                        mimeType: mimeType,
                        data: data,
                        filename: filename,
                        providerMetadata: providerMetadata,
                        providerOptionsName: providerOptionsName
                    ))
                case let .providerReference(mimeType, reference, _, providerMetadata)
                    where topLevelMediaType(mimeType) == "image":
                    _ = try deepSeekFilePartOptions(
                        providerMetadata,
                        providerOptionsName: providerOptionsName
                    )
                    content.append(.object([
                        "type": .string("file"),
                        "file_id": .string(try resolveProviderReference(reference, provider: "deepseek"))
                    ]))
                default:
                    warnings.append(AIWarning(type: "unsupported", feature: deepSeekUserPartFeature(part)))
                }
            }
            var object: [String: JSONValue] = [
                "role": .string("user"),
                "content": .array(content)
            ]
            if let name = messageOptions.name { object["name"] = .string(name) }
            output.append(.object(object))
        case .assistant:
            if messageOptions.prefix {
                guard index == messages.count - 1 else {
                    throw AIError.invalidArgument(argument: "messages", message: "DeepSeek assistant prefix completion requires the prefixed assistant message to be final.")
                }
                guard supportsAssistantPrefixCompletion else {
                    throw AIError.invalidArgument(argument: "messages", message: "DeepSeek assistant prefix completion requires a beta base URL ending in /beta.")
                }
            }
            let toolCalls = message.content.compactMap { part -> AIToolCall? in
                if case let .toolCall(call) = part { call } else { nil }
            }
            let reasoning = (index <= lastUserMessageIndex && !isDeepSeekV4) ? nil : message.reasoning
            var object: [String: JSONValue] = [
                "role": .string("assistant"),
                "content": .string(deepSeekText(from: message))
            ]
            if let name = messageOptions.name { object["name"] = .string(name) }
            if messageOptions.prefix { object["prefix"] = true }
            if isDeepSeekV4 {
                object["reasoning_content"] = .string(reasoning ?? "")
            } else if let reasoning {
                object["reasoning_content"] = .string(reasoning)
            }
            if !toolCalls.isEmpty {
                object["tool_calls"] = .array(toolCalls.map { call in
                    .object([
                        "id": .string(call.id),
                        "type": .string("function"),
                        "function": .object([
                            "name": .string(call.name),
                            "arguments": .string(call.arguments)
                        ])
                    ])
                })
            }
            output.append(.object(object))
        case .tool:
            if messageOptions.name != nil {
                warnings.append(AIWarning(type: "unsupported", feature: "message name on tool messages"))
            }
            for part in message.content {
                guard case let .toolResult(result) = part else { continue }
                output.append(.object([
                    "role": .string("tool"),
                    "tool_call_id": .string(result.toolCallID),
                    "content": try deepSeekToolResultContentValue(result, providerOptionsName: providerOptionsName, warnings: &warnings)
                ]))
            }
        }
    }

    return DeepSeekPreparedMessages(messages: output, warnings: warnings)
}

private func deepSeekToolResultContentValue(
    _ result: AIToolResult,
    providerOptionsName: String,
    warnings: inout [AIWarning]
) throws -> JSONValue {
    let output = result.modelOutput ?? result.result
    guard let object = output.objectValue,
          object["type"]?.stringValue == "content",
          let values = object["value"]?.arrayValue else {
        return .string(deepSeekToolResultContent(result))
    }

    let hasImagePart = values.contains { item in
        guard item["type"]?.stringValue == "file",
              topLevelMediaType(item["mediaType"]?.stringValue ?? "") == "image" else {
            return false
        }
        switch item["data"]?["type"]?.stringValue {
        case "reference", "url", "data": return true
        default: return false
        }
    }
    guard hasImagePart else {
        return .string(deepSeekToolResultContent(result))
    }

    var content: [JSONValue] = []
    for item in values {
        if item["type"]?.stringValue == "text" {
            content.append(.object([
                "type": .string("text"),
                "text": .string(item["text"]?.stringValue ?? "")
            ]))
            continue
        }
        guard item["type"]?.stringValue == "file",
              let mediaType = item["mediaType"]?.stringValue,
              topLevelMediaType(mediaType) == "image",
              let data = item["data"] else {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "tool result content part type: \(item["type"]?.stringValue ?? "unknown")"
            ))
            continue
        }
        switch data["type"]?.stringValue {
        case "reference":
            let reference = data["reference"]?.objectValue?.reduce(into: AIProviderReference()) { output, entry in
                if let value = entry.value.stringValue { output[entry.key] = value }
            } ?? [:]
            content.append(.object([
                "type": .string("file"),
                "file_id": .string(try resolveProviderReference(reference, provider: "deepseek"))
            ]))
        case "url":
            let url = data["url"]?.stringValue ?? ""
            let resolvedMediaType = mediaType
                .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: false)
                .first
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
                ?? mediaType.lowercased()
            guard isFullMediaType(resolvedMediaType) else {
                throw AIError.invalidArgument(
                    argument: "mediaType",
                    message: "File of media type \"\(mediaType)\" must specify subtype since it could not be auto-detected."
                )
            }
            guard deepSeekSupportedImageMediaTypes.contains(resolvedMediaType) else {
                throw AIError.invalidArgument(argument: "mediaType", message: "DeepSeek supports JPEG, PNG, GIF, and WebP image inputs.")
            }
            guard url.utf16.count <= 8_192 else {
                throw AIError.invalidArgument(
                    argument: "messages",
                    message: "DeepSeek image URLs must not exceed 8192 characters."
                )
            }
            let options = try deepSeekFilePartOptions(
                item["providerOptions"]?.objectValue ?? [:],
                providerOptionsName: providerOptionsName
            )
            content.append(deepSeekImageURLPart(url, detail: options.imageDetail))
        case "data":
            let base64 = data["data"]?.stringValue ?? ""
            let resolvedMediaType: String
            if isFullMediaType(mediaType) {
                resolvedMediaType = mediaType
            } else if let bytes = Data(base64Encoded: base64) {
                resolvedMediaType = try resolveFullMediaType(mediaType: mediaType, data: bytes)
            } else {
                throw AIError.invalidArgument(
                    argument: "mediaType",
                    message: "File of media type \"\(mediaType)\" must specify subtype since it could not be auto-detected."
                )
            }
            guard deepSeekSupportedImageMediaTypes.contains(resolvedMediaType) else {
                throw AIError.invalidArgument(
                    argument: "mediaType",
                    message: "DeepSeek supports JPEG, PNG, GIF, and WebP image inputs."
                )
            }
            let dataURLMediaType = resolvedMediaType == "image/jpg" ? "image/jpeg" : resolvedMediaType
            let options = try deepSeekFilePartOptions(
                item["providerOptions"]?.objectValue ?? [:],
                providerOptionsName: providerOptionsName
            )
            content.append(deepSeekImageURLPart(
                "data:\(dataURLMediaType);base64,\(base64)",
                detail: options.imageDetail
            ))
        default:
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "tool result content part type: file with data type: \(data["type"]?.stringValue ?? "unknown")"
            ))
        }
    }
    return .array(content)
}

func deepSeekIsV4Model(_ modelID: String) -> Bool {
    // Only known legacy aliases and V1-V3 protocols opt out. Future IDs use
    // the current protocol, including required assistant reasoning_content.
    modelID.range(
        of: #"(^|[/.])deepseek-(?:chat|reasoner|v[1-3](?:\.\d+)?)(?=[-/:.]|$)"#,
        options: .regularExpression
    ) == nil
}

private func deepSeekMessageOptions(
    _ providerMetadata: [String: JSONValue],
    providerOptionsName: String
) throws -> (name: String?, prefix: Bool) {
    let value = providerMetadata[providerOptionsName] ?? providerMetadata["deepseek"]
    guard let value, value != .null else { return (nil, false) }
    guard let object = value.objectValue else {
        throw AIError.invalidArgument(argument: "messages.providerMetadata.\(providerOptionsName)", message: "DeepSeek message provider options must be an object.")
    }
    let name: String?
    if let rawName = object["name"], rawName != .null {
        guard let parsed = rawName.stringValue else {
            throw AIError.invalidArgument(argument: "messages.providerMetadata.\(providerOptionsName).name", message: "DeepSeek message name must be a string.")
        }
        name = parsed
    } else {
        name = nil
    }
    let prefix: Bool
    if let rawPrefix = object["prefix"], rawPrefix != .null {
        guard rawPrefix.boolValue == true else {
            throw AIError.invalidArgument(argument: "messages.providerMetadata.\(providerOptionsName).prefix", message: "DeepSeek assistant prefix must be true when provided.")
        }
        prefix = true
    } else {
        prefix = false
    }
    return (name, prefix)
}

private func deepSeekIsImagePart(_ part: AIContentPart) -> Bool {
    switch part {
    case .imageURL:
        return true
    case let .data(mimeType, _, _),
         let .file(mimeType, _, _, _),
         let .providerReference(mimeType, _, _, _):
        return topLevelMediaType(mimeType) == "image"
    default:
        return false
    }
}

private func deepSeekImageURLPart(_ url: String, detail: String? = nil) -> JSONValue {
    var imageURL: [String: JSONValue] = ["url": .string(url)]
    if let detail {
        imageURL["detail"] = .string(detail)
    }
    return .object([
        "type": .string("image_url"),
        "image_url": .object(imageURL)
    ])
}

private func deepSeekInlineImagePart(
    mimeType: String,
    data: Data,
    filename: String?,
    providerMetadata: [String: JSONValue],
    providerOptionsName: String
) throws -> JSONValue {
    let options = try deepSeekFilePartOptions(
        providerMetadata,
        providerOptionsName: providerOptionsName
    )
    let resolvedMediaType = try resolveFullMediaType(mediaType: mimeType, data: data)
    guard deepSeekSupportedImageMediaTypes.contains(resolvedMediaType) else {
        throw AIError.invalidArgument(
            argument: "mediaType",
            message: "DeepSeek supports JPEG, PNG, GIF, and WebP image inputs."
        )
    }
    let dataURLMediaType = resolvedMediaType == "image/jpg" ? "image/jpeg" : resolvedMediaType
    let dataURL = "data:\(dataURLMediaType);base64,\(data.base64EncodedString())"
    if options.fileData {
        guard options.imageDetail == nil else {
            throw AIError.invalidArgument(
                argument: "messages",
                message: "DeepSeek `imageDetail` cannot be combined with `fileData`."
            )
        }
        var file: [String: JSONValue] = [
            "type": .string("file"),
            "file_data": .string(dataURL)
        ]
        if let filename {
            file["filename"] = .string(filename)
        }
        return .object(file)
    }
    return deepSeekImageURLPart(dataURL, detail: options.imageDetail)
}

private func deepSeekFilePartOptions(
    _ providerMetadata: [String: JSONValue],
    providerOptionsName: String
) throws -> (imageDetail: String?, fileData: Bool) {
    let value = providerMetadata[providerOptionsName] ?? providerMetadata["deepseek"]
    guard let value, value != .null else { return (nil, false) }
    guard let object = value.objectValue else {
        throw AIError.invalidArgument(
            argument: "messages.providerMetadata.\(providerOptionsName)",
            message: "DeepSeek file-part provider options must be an object."
        )
    }

    let imageDetail: String?
    if let rawDetail = object["imageDetail"] {
        guard let detail = rawDetail.stringValue,
              ["low", "high", "original", "auto"].contains(detail) else {
            throw AIError.invalidArgument(
                argument: "messages.providerMetadata.\(providerOptionsName).imageDetail",
                message: "DeepSeek imageDetail must be low, high, original, or auto."
            )
        }
        imageDetail = detail
    } else {
        imageDetail = nil
    }

    let fileData: Bool
    if let rawFileData = object["fileData"] {
        guard rawFileData.boolValue == true else {
            throw AIError.invalidArgument(
                argument: "messages.providerMetadata.\(providerOptionsName).fileData",
                message: "DeepSeek fileData must be true when provided."
            )
        }
        fileData = true
    } else {
        fileData = false
    }
    return (imageDetail, fileData)
}

private let deepSeekSupportedImageMediaTypes: Set<String> = [
    "image/gif", "image/jpeg", "image/jpg", "image/png", "image/webp"
]

func deepSeekUserPartFeature(_ part: AIContentPart) -> String {
    switch part {
    case .data, .file, .providerReference, .imageURL:
        return "user message part type: file"
    case .toolCall:
        return "user message part type: tool-call"
    case .toolResult:
        return "user message part type: tool-result"
    case .toolApprovalRequest:
        return "user message part type: tool-approval-request"
    case .toolApprovalResponse:
        return "user message part type: tool-approval-response"
    case .reasoning:
        return "user message part type: reasoning"
    case .reasoningFile:
        return "user message part type: reasoning-file"
    case .custom:
        return "user message part type: custom"
    case .text:
        return "user message part type: text"
    }
}

func deepSeekText(from message: AIMessage) -> String {
    message.content.compactMap(\.text).joined()
}