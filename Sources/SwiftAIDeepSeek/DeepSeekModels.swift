import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func deepSeekStreamProviderError(from raw: JSONValue) -> AIStreamProviderError? {
    guard let error = raw["error"] else { return nil }
    let type = error["type"]?.stringValue
    let code = deepSeekStreamErrorCode(error["code"])
    let metadata = deepSeekStreamErrorMetadata(type: type, code: code)
    return AIStreamProviderError(
        message: error["message"]?.stringValue ?? deepSeekJSONString(error) ?? "DeepSeek stream error.",
        type: type,
        code: code,
        statusCode: metadata.statusCode,
        isRetryable: metadata.isRetryable,
        data: raw
    )
}

private func deepSeekStreamErrorCode(_ value: JSONValue?) -> JSONValue? {
    guard let value, value.stringValue != nil || value.doubleValue != nil else {
        return nil
    }
    return value
}

private func deepSeekStreamErrorMetadata(
    type: String?,
    code: JSONValue?
) -> (statusCode: Int?, isRetryable: Bool?) {
    let codeString = code?.stringValue
    if codeString == "insufficient_quota" || type == "insufficient_quota" {
        return (429, false)
    }
    if let statusCode = deepSeekStreamHTTPStatusCode(code) {
        return (statusCode, deepSeekStreamStatusIsRetryable(statusCode))
    }
    for discriminator in [codeString, type] {
        switch discriminator {
        case "rate_limit_exceeded", "rate_limit_error":
            return (429, true)
        case "server_error", "api_error", "internal_server_error":
            return (500, true)
        case "overloaded_error", "service_unavailable":
            return (503, true)
        case "timeout", "timeout_error":
            return (504, true)
        case "authentication_error", "invalid_api_key":
            return (401, false)
        case "permission_error":
            return (403, false)
        case "not_found_error", "model_not_found":
            return (404, false)
        case "bad_request", "context_length_exceeded", "invalid_request_error":
            return (400, false)
        default:
            continue
        }
    }
    return (nil, nil)
}

private func deepSeekStreamHTTPStatusCode(_ value: JSONValue?) -> Int? {
    let statusCode: Int?
    if let intValue = value?.intValue {
        statusCode = intValue
    } else if let stringValue = value?.stringValue,
              stringValue.count == 3,
              stringValue.allSatisfy(\.isNumber) {
        statusCode = Int(stringValue)
    } else {
        statusCode = nil
    }
    guard let statusCode, 400...599 ~= statusCode else { return nil }
    return statusCode
}

private func deepSeekStreamStatusIsRetryable(_ statusCode: Int) -> Bool {
    statusCode == 408 || statusCode == 409 || statusCode == 429 || statusCode >= 500
}

public final class DeepSeekLanguageModel: LanguageModel, @unchecked Sendable {
    public let providerID: String
    public let modelID: String
    private let config: DeepSeekModelConfiguration

    public init(modelID: String, config: DeepSeekModelConfiguration) {
        self.providerID = config.providerID
        self.modelID = modelID
        self.config = config
    }

    public func generate(_ request: LanguageModelRequest) async throws -> TextGenerationResult {
        let prepared = try deepSeekPreparedCall(
            for: request,
            modelID: modelID,
            stream: false,
            supportsThinking: config.supportsThinking,
            supportsStructuredOutputs: config.supportsStructuredOutputs,
            supportsAssistantPrefixCompletion: config.baseURL.hasSuffix("/beta"),
            supportsStrictToolCalls: config.baseURL.hasSuffix("/beta"),
            supportsPenaltySampling: providerID.hasPrefix("azure."),
            providerOptionsName: providerID.split(separator: ".").first.map(String.init) ?? "deepseek"
        )
        let response = try await config.sendJSONResponse(
            path: "/chat/completions",
            modelID: modelID,
            body: .object(prepared.body),
            headers: request.headers,
            abortSignal: request.abortSignal
        )
        let raw = response.json
        guard let choices = raw["choices"]?.arrayValue, !choices.isEmpty else {
            throw AIError.invalidResponse(provider: providerID, message: "Response did not contain any choices.")
        }
        let choice: JSONValue? = choices[0]
        let reasoning = choice?["message"]?["reasoning_content"]?.stringValue ?? ""
        let toolCalls = deepSeekToolCalls(from: choice?["message"]?["tool_calls"])
        guard let text = choice?["message"]?["content"]?.stringValue ?? (toolCalls.isEmpty ? nil : "") else {
            throw AIError.invalidResponse(provider: providerID, message: "No text content found in DeepSeek response.")
        }
        return TextGenerationResult(
            text: text,
            reasoning: reasoning,
            finishReason: deepSeekFinishReason(choice?["finish_reason"]?.stringValue),
            usage: deepSeekUsage(from: raw),
            toolCalls: toolCalls,
            providerMetadata: deepSeekProviderMetadata(from: raw),
            rawValue: raw,
            warnings: prepared.warnings,
            responseMetadata: aiResponseMetadata(from: raw, response: response.response, modelID: modelID)
        )
    }

    public func stream(_ request: LanguageModelRequest) -> AsyncThrowingStream<LanguageStreamPart, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let prepared = try deepSeekPreparedCall(
                        for: request,
                        modelID: modelID,
                        stream: true,
                        supportsThinking: config.supportsThinking,
                        supportsStructuredOutputs: config.supportsStructuredOutputs,
                        supportsAssistantPrefixCompletion: config.baseURL.hasSuffix("/beta"),
                        supportsStrictToolCalls: config.baseURL.hasSuffix("/beta"),
                        supportsPenaltySampling: providerID.hasPrefix("azure."),
                        providerOptionsName: providerID.split(separator: ".").first.map(String.init) ?? "deepseek"
                    )
                    let httpRequest = try config.request(
                        path: "/chat/completions",
                        modelID: modelID,
                        body: .object(prepared.body),
                        headers: request.headers,
                        abortSignal: request.abortSignal
                    )
                    let response = try await config.streamRequest(httpRequest)
                    guard (200..<300).contains(response.statusCode) else {
                        throw apiCallError(
                            provider: providerID,
                            response: try await bufferedHTTPResponse(from: response, request: httpRequest)
                        )
                    }
                    let responseHead = httpResponseHead(from: response, request: httpRequest)

                    continuation.yield(.streamStart(warnings: prepared.warnings))
                    var latestUsage: TokenUsage? = TokenUsage()
                    var finishReason: String? = "interrupted"
                    var providerMetadata: [String: JSONValue] = [:]
                    var toolCalls = DeepSeekStreamingToolCalls()
                    var didEmitResponseMetadata = false
                    var activeReasoningID: String?
                    var activeTextID: String?
                    for try await event in serverSentEvents(from: response.body) {
                        if event.data == "[DONE]" { break }
                        let raw: JSONValue
                        do {
                            raw = try decodeJSONBody(Data(event.data.utf8))
                        } catch {
                            finishReason = "error"
                            continuation.yield(.error(message: error.localizedDescription))
                            continue
                        }
                        if request.includeRawChunks {
                            continuation.yield(.raw(raw))
                        }
                        if let providerError = deepSeekStreamProviderError(from: raw) {
                            finishReason = "error"
                            continuation.yield(.providerError(providerError))
                            continue
                        }
                        if !didEmitResponseMetadata {
                            didEmitResponseMetadata = true
                            continuation.yield(.responseMetadata(aiResponseMetadata(from: raw, response: responseHead, modelID: modelID)))
                        }
                        latestUsage = deepSeekUsage(from: raw) ?? latestUsage
                        deepSeekMergeProviderMetadata(deepSeekProviderMetadata(from: raw), into: &providerMetadata)
                        if let reasoning = raw["choices"]?[0]?["delta"]?["reasoning_content"]?.stringValue, !reasoning.isEmpty {
                            let id = activeReasoningID ?? "reasoning-0"
                            if activeReasoningID == nil {
                                activeReasoningID = id
                                continuation.yield(.reasoningStart(id: id))
                            }
                            continuation.yield(.reasoningDeltaPart(id: id, delta: reasoning))
                        }
                        if let delta = raw["choices"]?[0]?["delta"]?["content"]?.stringValue, !delta.isEmpty {
                            let id = activeTextID ?? "txt-0"
                            if activeTextID == nil {
                                activeTextID = id
                                continuation.yield(.textStart(id: id))
                            }
                            if let reasoningID = activeReasoningID {
                                continuation.yield(.reasoningEnd(id: reasoningID))
                                activeReasoningID = nil
                            }
                            continuation.yield(.textDeltaPart(id: id, delta: delta))
                        }
                        if let toolCallDeltas = raw["choices"]?[0]?["delta"]?["tool_calls"]?.arrayValue,
                           !toolCallDeltas.isEmpty {
                            if let reasoningID = activeReasoningID {
                                continuation.yield(.reasoningEnd(id: reasoningID))
                                activeReasoningID = nil
                            }
                            for toolCallDelta in toolCallDeltas {
                                if !toolCalls.hasMatchingBuffer(for: toolCallDelta) {
                                    guard toolCallDelta["id"]?.stringValue != nil else {
                                        throw AIError.invalidResponse(provider: providerID, message: "Expected 'id' to be a string.")
                                    }
                                    guard toolCallDelta["function"]?["name"]?.stringValue != nil else {
                                        throw AIError.invalidResponse(provider: providerID, message: "Expected 'function.name' to be a string.")
                                    }
                                }
                                for part in toolCalls.apply(delta: toolCallDelta) {
                                    continuation.yield(part)
                                }
                            }
                        }
                        if let finishReasonValue = raw["choices"]?[0]?["finish_reason"], finishReasonValue != .null {
                            finishReason = deepSeekFinishReason(finishReasonValue.stringValue)
                        }
                    }
                    if let reasoningID = activeReasoningID {
                        continuation.yield(.reasoningEnd(id: reasoningID))
                    }
                    if let textID = activeTextID {
                        continuation.yield(.textEnd(id: textID))
                    }
                    for part in toolCalls.finishedParts() {
                        continuation.yield(part)
                    }
                    continuation.yield(.finishMetadata(
                        reason: finishReason,
                        usage: latestUsage,
                        providerMetadata: providerMetadata
                    ))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
}

typealias DeepSeekStreamingToolCalls = OpenAIStyleStreamingToolCalls

struct DeepSeekPreparedCall {
    var body: [String: JSONValue]
    var warnings: [AIWarning]
}

struct DeepSeekPreparedMessages {
    var messages: [JSONValue]
    var warnings: [AIWarning]
}

struct DeepSeekPreparedTools {
    var tools: [JSONValue]
    var warnings: [AIWarning]
}