@testable import SwiftAIDeepSeek
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

@Test func deepSeekV4SerializesInlineURLAndReferencedImagesLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-v4-flash-vision-exp")

    _ = try await model.generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .text("Compare these images"),
            .data(mimeType: "image/png", data: Data([0, 1, 2, 3])),
            .imageURL("https://example.com/cat.png"),
            .providerReference(
                mimeType: "image/png",
                reference: ["deepseek": "file-api-deepseek", "openai": "file-openai"]
            )
        ])
    ]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let content = try #require(body["messages"]?[0]?["content"]?.arrayValue)
    #expect(content[0]["type"]?.stringValue == "text")
    #expect(content[0]["text"]?.stringValue == "Compare these images")
    #expect(content[1]["type"]?.stringValue == "image_url")
    #expect(content[1]["image_url"]?["url"]?.stringValue == "data:image/png;base64,AAECAw==")
    #expect(content[2]["image_url"]?["url"]?.stringValue == "https://example.com/cat.png")
    #expect(content[3]["type"]?.stringValue == "file")
    #expect(content[3]["file_id"]?.stringValue == "file-api-deepseek")
}

@Test func deepSeekV4AppliesImageDetailAndFileDataOptionsLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-v4-flash-vision-exp")

    _ = try await model.generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .imageURL(
                "https://example.com/image.webp",
                providerMetadata: ["deepseek": ["imageDetail": "low"]]
            ),
            .file(
                mimeType: "image/jpg",
                data: Data([0, 1, 2, 3]),
                filename: "sample.jpg",
                providerMetadata: ["deepseek": ["fileData": true]]
            )
        ])
    ]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let content = try #require(body["messages"]?[0]?["content"]?.arrayValue)
    #expect(content[0]["type"]?.stringValue == "image_url")
    #expect(content[0]["image_url"]?["url"]?.stringValue == "https://example.com/image.webp")
    #expect(content[0]["image_url"]?["detail"]?.stringValue == "low")
    #expect(content[1]["type"]?.stringValue == "file")
    #expect(content[1]["file_data"]?.stringValue == "data:image/jpeg;base64,AAECAw==")
    #expect(content[1]["filename"]?.stringValue == "sample.jpg")
}

@Test func deepSeekV4RejectsInvalidImageOptionsFormatsAndLongURLsBeforeFetching() async throws {
    let transport = RecordingTransport(response: jsonResponse("{}"))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-v4-flash-vision-exp")

    await #expect(throws: AIError.self) {
        _ = try await model.generate(LanguageModelRequest(messages: [
            AIMessage(role: .user, content: [
                .file(
                    mimeType: "image/png",
                    data: Data([0, 1, 2, 3]),
                    providerMetadata: ["deepseek": ["fileData": true, "imageDetail": "high"]]
                )
            ])
        ]))
    }
    await #expect(throws: AIError.self) {
        _ = try await model.generate(LanguageModelRequest(messages: [
            AIMessage(role: .user, content: [
                .imageURL(
                    "https://example.com/image.png",
                    providerMetadata: ["deepseek": ["fileData": true]]
                )
            ])
        ]))
    }
    await #expect(throws: AIError.self) {
        _ = try await model.generate(LanguageModelRequest(messages: [
            AIMessage(role: .user, content: [
                .imageURL("https://example.com/\(String(repeating: "a", count: 8_192))")
            ])
        ]))
    }
    await #expect(throws: AIError.self) {
        _ = try await model.generate(LanguageModelRequest(messages: [
            AIMessage(role: .user, content: [
                .data(mimeType: "image/svg+xml", data: Data("<svg/>".utf8))
            ])
        ]))
    }
    #expect(await transport.requests().isEmpty)
}

@Test func deepSeekV4RejectsImageReferenceWithoutDeepSeekIdentifier() async throws {
    let transport = RecordingTransport(response: jsonResponse("{}"))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(apiKey: "deepseek-key", transport: transport))
    let model = try provider.languageModel("deepseek-v4-flash-vision-exp")

    await #expect(throws: AINoSuchProviderReferenceError.self) {
        _ = try await model.generate(LanguageModelRequest(messages: [
            AIMessage(role: .user, content: [
                .providerReference(mimeType: "image/png", reference: ["openai": "file-openai"])
            ])
        ]))
    }
    #expect(await transport.requests().isEmpty)
}

@Test func deepSeekFilesUploadsUserDataWithExpiryAndReturnsMetadata() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {
      "id":"file-api-xyz789",
      "object":"file",
      "bytes":1024,
      "created_at":1700000000,
      "filename":"comic-cat.png",
      "purpose":"user_data",
      "expires_at":1700003600
    }
    """))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        headers: ["Custom-Header": "custom-value"],
        transport: transport
    ))

    let result = try await provider.files().uploadFile(FileUploadRequest(
        data: Data([1, 2, 3]),
        mediaType: "image/png",
        filename: "comic-cat.png",
        providerOptions: ["deepseek": ["expiresAfter": 3_600]]
    ))

    #expect(result.providerReference == ["deepseek": "file-api-xyz789"])
    #expect(result.filename == "comic-cat.png")
    #expect(result.mediaType == "image/png")
    #expect(result.warnings == [])
    #expect(result.providerMetadata["deepseek"]?["purpose"]?.stringValue == "user_data")
    #expect(result.providerMetadata["deepseek"]?["object"]?.stringValue == "file")
    #expect(result.providerMetadata["deepseek"]?["bytes"]?.intValue == 1024)
    #expect(result.providerMetadata["deepseek"]?["createdAt"]?.intValue == 1_700_000_000)
    #expect(result.providerMetadata["deepseek"]?["expiresAt"]?.intValue == 1_700_003_600)

    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://api.deepseek.com/files")
    #expect(request.headers["authorization"] == "Bearer deepseek-key")
    #expect(request.headers["custom-header"] == "custom-value")
    let requestBody = try #require(request.body)
    let multipart = try #require(String(data: requestBody, encoding: .utf8))
    #expect(multipart.contains("name=\"purpose\"\r\n\r\nuser_data"))
    #expect(multipart.contains("name=\"expires_after[anchor]\"\r\n\r\ncreated_at"))
    #expect(multipart.contains("name=\"expires_after[seconds]\"\r\n\r\n3600"))
}

@Test func deepSeekFilesPreflightsImageFormatSizeAndUnicodeScalarFilename() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"id":"file-api-valid"}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        transport: transport
    ))
    let files = provider.files()

    _ = try await files.uploadFile(FileUploadRequest(
        data: Data([0x47, 0x49, 0x46, 0x38, 0x39, 0x61]),
        mediaType: "application/octet-stream"
    ))
    #expect((await transport.requests()).count == 1)

    await #expect(throws: AIError.invalidArgument(
        argument: "data",
        message: "DeepSeek file uploads support JPEG, PNG, GIF, and WebP images. Detected unsupported file content type \"application/pdf\"."
    )) {
        _ = try await files.uploadFile(FileUploadRequest(
            data: Data([0x25, 0x50, 0x44, 0x46]),
            mediaType: "image/png",
            filename: "image.png"
        ))
    }
    await #expect(throws: AIError.invalidArgument(
        argument: "data",
        message: "DeepSeek file uploads must not exceed 64 MiB (67108864 bytes). Received 67108865 bytes."
    )) {
        _ = try await files.uploadFile(FileUploadRequest(
            data: Data(count: 64 * 1_024 * 1_024 + 1),
            mediaType: "image/png",
            filename: "image.png"
        ))
    }
    let tooManyScalars = String(repeating: "e\u{301}", count: 255) + "abc"
    #expect(tooManyScalars.count < 512)
    #expect(tooManyScalars.unicodeScalars.count == 513)
    await #expect(throws: AIError.invalidArgument(
        argument: "filename",
        message: "DeepSeek filenames must not exceed 512 characters. Received 513 characters."
    )) {
        _ = try await files.uploadFile(FileUploadRequest(
            data: Data([0x89, 0x50, 0x4E, 0x47]),
            mediaType: "image/png",
            filename: tooManyScalars
        ))
    }
    #expect((await transport.requests()).count == 1)
}

@Test func deepSeekFilesValidatesExpiryRangeAndIntegerShape() async throws {
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        transport: RecordingTransport(response: jsonResponse("{}"))
    ))
    let files = provider.files()
    let expected = AIError.invalidArgument(
        argument: "providerOptions.deepseek.expiresAfter",
        message: "DeepSeek expiresAfter must be an integer between 3600 and 2592000."
    )

    await #expect(throws: expected) {
        _ = try await files.uploadFile(FileUploadRequest(
            data: Data([1]),
            mediaType: "image/png",
            providerOptions: ["deepseek": ["expiresAfter": 3_599]]
        ))
    }
    await #expect(throws: expected) {
        _ = try await files.uploadFile(FileUploadRequest(
            data: Data([1]),
            mediaType: "image/png",
            providerOptions: ["deepseek": ["expiresAfter": 3_600.5]]
        ))
    }
}

@Test func deepSeekFilesUsesBlobFilenameWhenCallerOmitsFilenameLikeFormData() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"id":"file-api-blob"}"#))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        transport: transport
    ))

    let result = try await provider.files().uploadFile(FileUploadRequest(
        data: Data([1, 2, 3]),
        mediaType: "image/png"
    ))

    #expect(result.filename == nil)
    #expect(result.requestMetadata.body?["file"]?["filename"]?.stringValue == "blob")
    let request = try #require(await transport.requests().first)
    let requestBody = try #require(request.body)
    let multipart = try #require(String(data: requestBody, encoding: .utf8))
    #expect(multipart.contains("name=\"file\"; filename=\"blob\""))
}

@Test func deepSeekFilesExtractsNestedAPIErrorMessageLikeUpstream() async throws {
    let transport = RecordingTransport(response: AIHTTPResponse(
        statusCode: 400,
        headers: ["content-type": "application/json", "x-request-id": "req-files-1"],
        body: Data(#"{"error":{"message":"File type is not supported","type":"invalid_request_error","code":"invalid_file"}}"#.utf8)
    ))
    let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        transport: transport
    ))

    do {
        _ = try await provider.files().uploadFile(FileUploadRequest(
            data: Data([1]),
            mediaType: "image/png"
        ))
        Issue.record("Expected DeepSeek Files API error.")
    } catch let error as AIError {
        let apiError = try #require(error.apiCallError)
        #expect(apiError.provider == "deepseek.files")
        #expect(apiError.statusCode == 400)
        #expect(apiError.responseBody == "File type is not supported")
        #expect(apiError.responseHeaders["x-request-id"] == "req-files-1")
        #expect(!apiError.isRetryable)
    }
}

@Test func deepSeekFilesRejectsMisleadingOptionalResponseMetadataLikeUpstream() async throws {
    let invalidFields: [(String, JSONValue)] = [
        ("object", "not-a-file"),
        ("object", 42),
        ("filename", 42),
        ("purpose", "assistants"),
        ("purpose", 42),
        ("bytes", -1),
        ("bytes", 1.5),
        ("created_at", -1),
        ("created_at", 1.5),
        ("expires_at", -1),
        ("expires_at", 1.5)
    ]

    for (field, value) in invalidFields {
        var response: [String: JSONValue] = ["id": "file-invalid"]
        response[field] = value
        let responseJSON = String(decoding: try encodeJSONBody(.object(response)), as: UTF8.self)
        let transport = RecordingTransport(response: jsonResponse(responseJSON))
        let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
            apiKey: "deepseek-key",
            transport: transport
        ))

        do {
            _ = try await provider.files().uploadFile(FileUploadRequest(
                data: Data([1, 2, 3]),
                mediaType: "image/png"
            ))
            Issue.record("Expected invalid DeepSeek Files response for \(field)=\(value).")
        } catch AIError.invalidResponse(let providerID, _) {
            #expect(providerID == "deepseek.files")
        } catch {
            Issue.record("Expected AIError.invalidResponse for \(field), got \(error).")
        }
    }
}

@Test func deepSeekFilesPreserveUploadOnlyContractAndCancelUnsupportedStreams() async throws {
    let transport = RecordingTransport(responses: [])
    let files = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        transport: transport
    )).files()
    let probe = DeepSeekFileStreamCancellationProbe()

    #expect(files.supportedFileOperations == [.upload])
    await #expect(throws: AIError.invalidArgument(
        argument: "data",
        message: "deepseek.files does not support streaming file upload."
    )) {
        _ = try await files.uploadFile(FileUploadRequest(
            stream: AsyncThrowingStream { continuation in
                continuation.onTermination = { _ in probe.record() }
            },
            mediaType: "image/png"
        ))
    }

    #expect(await waitForDeepSeekFileStreamCancellation(probe))
    #expect(await transport.requests().isEmpty)
}

@Test func deepSeekFilesForwardCallHeadersAndAbortSignal() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"id":"file-forwarded"}"#))
    let files = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        transport: transport
    )).files()
    let controller = AIAbortController()

    _ = try await files.uploadFile(FileUploadRequest(
        data: Data([0x89, 0x50, 0x4e, 0x47]),
        mediaType: "image/png",
        headers: ["X-Request-Test": "forwarded"],
        abortSignal: controller.signal
    ))

    let request = try #require(await transport.requests().first)
    #expect(request.headers["X-Request-Test"] == "forwarded")
    #expect(request.abortSignal === controller.signal)

    let abortedTransport = RecordingTransport(responses: [])
    let abortedFiles = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings(
        apiKey: "deepseek-key",
        transport: abortedTransport
    )).files()
    let abortedController = AIAbortController()
    abortedController.abort(reason: "cancel upload", reasonName: "AbortError")
    await #expect(throws: AIAbortError.self) {
        _ = try await abortedFiles.uploadFile(FileUploadRequest(
            data: Data([0x89, 0x50, 0x4e, 0x47]),
            mediaType: "image/png",
            abortSignal: abortedController.signal
        ))
    }
    #expect(await abortedTransport.requests().isEmpty)
}

private final class DeepSeekFileStreamCancellationProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    var wasCancelled: Bool { lock.withLock { cancelled } }
    func record() { lock.withLock { cancelled = true } }
}

private func waitForDeepSeekFileStreamCancellation(
    _ probe: DeepSeekFileStreamCancellationProbe
) async -> Bool {
    for _ in 0..<100 {
        if probe.wasCancelled { return true }
        try? await Task.sleep(nanoseconds: 1_000_000)
    }
    return probe.wasCancelled
}
