import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

/// DeepSeek's Files V4 upload surface. Unlike the generic OpenAI-compatible
/// multipart route, DeepSeek fixes the purpose to `user_data` and exposes a
/// bounded `expiresAfter` provider option.
public final class DeepSeekFileClient: AIFileClient, @unchecked Sendable {
    public let providerID = "deepseek.files"
    private let config: DeepSeekModelConfiguration

    public init(config: DeepSeekModelConfiguration) {
        self.config = config
    }

    public func uploadFile(_ request: FileUploadRequest) async throws -> FileUploadResult {
        if request.fileData.isStream {
            await request.fileData.cancelStream()
            throw unsupportedStreamingFileUpload(providerID: providerID)
        }
        try request.abortSignal?.throwIfAborted()
        let expiresAfter = try deepSeekFileExpiresAfter(request.providerOptions)
        try deepSeekValidateFileUpload(request)
        var form = MultipartFormData()
        form.appendFile(
            name: "file",
            fileName: request.filename ?? "blob",
            mimeType: request.mediaType,
            data: request.data
        )
        form.appendField(name: "purpose", value: "user_data")
        if let expiresAfter {
            form.appendField(name: "expires_after[anchor]", value: "created_at")
            form.appendField(name: "expires_after[seconds]", value: String(expiresAfter))
        }

        let uploadRequest = try config.rawRequest(
            path: "/files",
            modelID: "",
            body: form.finalize(),
            contentType: "multipart/form-data; boundary=\(form.boundary)",
            headers: request.headers,
            abortSignal: request.abortSignal
        )
        let response = try await config.transport.send(uploadRequest)
        guard (200..<300).contains(response.statusCode) else {
            throw deepSeekFileHTTPStatusError(response: response)
        }

        let raw = try response.jsonValue()
        guard let id = raw["id"]?.stringValue else {
            throw AIError.invalidResponse(provider: providerID, message: "DeepSeek Files response is missing id.")
        }
        try deepSeekValidateOptionalFileResponseFields(raw, providerID: providerID)

        var metadata: [String: JSONValue] = [:]
        if let object = raw["object"], object != .null { metadata["object"] = object }
        if let filename = raw["filename"], filename != .null { metadata["filename"] = filename }
        if let purpose = raw["purpose"], purpose != .null { metadata["purpose"] = purpose }
        if let bytes = raw["bytes"], bytes != .null { metadata["bytes"] = bytes }
        if let createdAt = raw["created_at"], createdAt != .null { metadata["createdAt"] = createdAt }
        if let expiresAt = raw["expires_at"], expiresAt != .null { metadata["expiresAt"] = expiresAt }

        var warnings: [AIWarning] = []
        if request.displayName != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "displayName"))
        }
        if request.purpose != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "purpose"))
        }
        if !request.extraBody.isEmpty {
            warnings.append(AIWarning(type: "unsupported", feature: "extraBody"))
        }

        var requestBody: [String: JSONValue] = [
            "file": .object(fileUploadMetadata(request, defaultFilename: "blob")),
            "purpose": .string("user_data")
        ]
        if let expiresAfter {
            requestBody["expiresAfter"] = .number(Double(expiresAfter))
        }

        return FileUploadResult(
            providerReference: ["deepseek": id],
            filename: raw["filename"]?.stringValue ?? request.filename,
            mediaType: request.mediaType,
            byteSize: fileInteger(raw["bytes"]),
            createdAt: fileDate(raw["created_at"]),
            expiresAt: fileDate(raw["expires_at"]),
            rawValue: raw,
            warnings: warnings,
            providerMetadata: ["deepseek": .object(metadata)],
            requestMetadata: AIRequestMetadata(body: .object(requestBody), headers: request.headers),
            responseMetadata: aiResponseMetadata(from: raw, response: response)
        )
    }
}

private let deepSeekMaximumFileSizeBytes = 64 * 1_024 * 1_024
private let deepSeekMaximumFilenameScalars = 512
private let deepSeekFileSupportedMediaTypes: Set<String> = [
    "image/gif", "image/jpeg", "image/jpg", "image/png", "image/webp"
]
private let deepSeekFileGenericMediaTypes: Set<String> = [
    "", "application/binary", "application/octet-stream", "binary/octet-stream", "image", "image/*"
]
private let deepSeekFileSupportedExtensions: Set<String> = ["gif", "jpeg", "jpg", "png", "webp"]

private func deepSeekValidateFileUpload(_ request: FileUploadRequest) throws {
    guard request.data.count <= deepSeekMaximumFileSizeBytes else {
        throw AIError.invalidArgument(
            argument: "data",
            message: "DeepSeek file uploads must not exceed 64 MiB (\(deepSeekMaximumFileSizeBytes) bytes). Received \(request.data.count) bytes."
        )
    }

    if let filename = request.filename {
        let scalarCount = filename.unicodeScalars.count
        guard scalarCount <= deepSeekMaximumFilenameScalars else {
            throw AIError.invalidArgument(
                argument: "filename",
                message: "DeepSeek filenames must not exceed 512 characters. Received \(scalarCount) characters."
            )
        }
    }

    let normalizedMediaType = request.mediaType
        .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: false)
        .first
        .map(String.init)?
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased() ?? ""
    let detectedMediaType = detectMediaType(data: request.data)

    if let detectedMediaType,
       !deepSeekFileSupportedMediaTypes.contains(detectedMediaType) {
        throw AIError.invalidArgument(
            argument: "data",
            message: "DeepSeek file uploads support JPEG, PNG, GIF, and WebP images. Detected unsupported file content type \"\(detectedMediaType)\"."
        )
    }

    if deepSeekFileSupportedMediaTypes.contains(normalizedMediaType) {
        return
    }

    guard deepSeekFileGenericMediaTypes.contains(normalizedMediaType) else {
        throw AIError.invalidArgument(
            argument: "mediaType",
            message: "DeepSeek file uploads support JPEG, PNG, GIF, and WebP images. Received unsupported media type \"\(request.mediaType)\"."
        )
    }

    if detectedMediaType != nil || deepSeekHasSupportedFileExtension(request.filename) {
        return
    }

    throw AIError.invalidArgument(
        argument: "mediaType",
        message: "DeepSeek file uploads support JPEG, PNG, GIF, and WebP images. Provide a supported media type or a filename ending in .jpg, .jpeg, .png, .gif, or .webp. Received \"\(request.mediaType)\"."
    )
}

private func deepSeekHasSupportedFileExtension(_ filename: String?) -> Bool {
    guard let filename,
          let separator = filename.lastIndex(of: "."),
          separator < filename.index(before: filename.endIndex) else {
        return false
    }
    let extensionStart = filename.index(after: separator)
    return deepSeekFileSupportedExtensions.contains(filename[extensionStart...].lowercased())
}

private func deepSeekFileHTTPStatusError(response: AIHTTPResponse) -> AIError {
    let body = (try? response.jsonValue())?["error"]?["message"]?.stringValue
        ?? response.bodyText
    return .apiCall(
        provider: "deepseek.files",
        statusCode: response.statusCode,
        body: body,
        headers: response.headers
    )
}

private func deepSeekFileExpiresAfter(_ providerOptions: [String: JSONValue]) throws -> Int? {
    guard let namespace = providerOptions["deepseek"], namespace != .null else { return nil }
    guard let options = namespace.objectValue else {
        throw AIError.invalidArgument(
            argument: "providerOptions.deepseek",
            message: "DeepSeek file provider options must be an object."
        )
    }
    guard let value = options["expiresAfter"] else { return nil }
    guard let number = value.doubleValue,
          number.isFinite,
          let seconds = Int(exactly: number),
          (3_600...2_592_000).contains(seconds) else {
        throw AIError.invalidArgument(
            argument: "providerOptions.deepseek.expiresAfter",
            message: "DeepSeek expiresAfter must be an integer between 3600 and 2592000."
        )
    }
    return seconds
}

private func deepSeekValidateOptionalFileResponseFields(_ raw: JSONValue, providerID: String) throws {
    if let object = raw["object"], object != .null, object.stringValue != "file" {
        throw AIError.invalidResponse(provider: providerID, message: "DeepSeek Files response field object must be \"file\".")
    }
    if let filename = raw["filename"], filename != .null, filename.stringValue == nil {
        throw AIError.invalidResponse(provider: providerID, message: "DeepSeek Files response field filename must be a string.")
    }
    if let purpose = raw["purpose"], purpose != .null, purpose.stringValue != "user_data" {
        throw AIError.invalidResponse(provider: providerID, message: "DeepSeek Files response field purpose must be \"user_data\".")
    }
    for key in ["bytes", "created_at", "expires_at"] {
        if let value = raw[key], value != .null, !deepSeekIsNonnegativeSafeInteger(value) {
            throw AIError.invalidResponse(provider: providerID, message: "DeepSeek Files response field \(key) must be a nonnegative safe integer.")
        }
    }
}

private func deepSeekIsNonnegativeSafeInteger(_ value: JSONValue) -> Bool {
    guard let number = value.doubleValue,
          number.isFinite,
          number >= 0,
          number.rounded(.towardZero) == number else {
        return false
    }
    return number <= 9_007_199_254_740_991
}
