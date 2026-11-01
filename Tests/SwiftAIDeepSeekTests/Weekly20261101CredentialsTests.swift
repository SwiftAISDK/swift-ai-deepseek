import Foundation
import Testing
import SwiftAIProvider
import SwiftAITransport
@testable import SwiftAIDeepSeek

private actor WeeklyCredentialSequence {
    private(set) var count = 0
    func next() -> String { count += 1; return "rotating-\(count)" }
}
private actor WeeklyCredentialProbe: AITransport {
    private(set) var requests: [AIHTTPRequest] = []
    struct Sent: Error {}
    func send(_ request: AIHTTPRequest) async throws -> AIHTTPResponse {
        requests.append(request)
        throw Sent()
    }
}

@Suite("Weekly20261101 deepseek credentials")
struct Weekly20261101CredentialsTests {
    @Test func resolvesFreshCredentialOnlyWhenRequestIsSent() async throws {
        let sequence = WeeklyCredentialSequence()
        let transport = WeeklyCredentialProbe()
        let provider = try DeepSeekProvider(configuration: DeepSeekProviderSettings(apiKeyProvider: { await sequence.next() }, transport: transport))
        let model = try provider.languageModel("deepseek-v4-flash-vision-exp")
        #expect(await sequence.count == 0)
        for _ in 0..<2 { _ = try? await model.generate(LanguageModelRequest(messages: [.user("Hello")])) }
        let requests = await transport.requests
        #expect(requests.count == 2)
        #expect(await sequence.count == 2)
        for (index, request) in requests.enumerated() {
            #expect(request.headers.values.contains(where: { $0.contains("rotating-\(index + 1)") }))
            #expect(!request.headers.values.contains(where: { $0.contains("__swift_ai_credential_") }))
        }
    }
}
