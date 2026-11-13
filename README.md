<!-- Generated distribution; edit the canonical repository. -->
# swift-ai-deepseek 1.2.0

Independent Swift-native port of Vercel AI SDK. Source and upstream dispositions are in PROVENANCE.json and Porting/.

## Distribution manifest

Add this package to your SwiftPM dependencies:

```swift
.package(url: "https://github.com/SwiftAISDK/swift-ai-deepseek.git", from: "1.2.0")
```

Products: `SwiftAIDeepSeek`.
Contribute and report issues in the canonical SwiftAISDK repository.

## Authored usage guide

# SwiftAIDeepSeek

Swift-native port of `@ai-sdk/deepseek`. Maintained in the canonical SwiftAISDK repository and exported as an independent Swift package.

```swift
import SwiftAIDeepSeek

let provider = try DeepSeekProvider(configuration: .init(apiKey: "..."))
```

Settings accept a custom base URL, headers, query parameters and an injected
`AITransport`. Models use the common SwiftAIProvider protocols.

Run the standalone suite from the canonical checkout:

```sh
swift test --package-path Packages/swift-ai-deepseek
```

The canonical repository owns all edits. Distribution manifests and repositories
are generated from the catalog; see `Porting/map.json` and the canonical provider guide.

See [provider usage and capabilities](https://github.com/SwiftAISDK/SwiftAISDK/blob/main/docs-site/scripts/provider-guides/deepseek.md) for authentication, options and supported surfaces.
## Behavior notes

Tool replay preserves valid JSON argument text and quotes malformed text as a JSON string. Native DeepSeek Chat Completions strict tools still require the beta endpoint and consistent strict mode. Azure-hosted adapter configurations have unknown strict-tool capability and forward each tool's explicit true/false/omitted choice without imposing the native beta policy.

## Current upstream review

The reviewed baseline is `@ai-sdk/deepseek` 3.0.86. The existing `languageModel`/`chatModel` factories retain Chat Completions. Opt into Responses and provider web search explicitly:

```swift
let model = provider.responses("deepseek-v4-pro")
let result = try await model.generate(.init(
    messages: [.user("Find the latest release notes.")],
    tools: ["search": DeepSeekTools.webSearch()],
    providerOptions: DeepSeekResponsesModelOptions(reasoningEffort: .high).providerOptions
))
```

Responses supports structured JSON, image URLs/inline images/provider references, ordered reasoning/tool replay, and unary or incremental SSE results. Reasoning `itemId`, full web-search call/action replay, response IDs, logprobs and exact server-tool usage survive in provider metadata. `thinking: .disabled` sends reasoning effort `none`; portable reasoning maps `medium` to `high` and `xhigh` to `max`, with compatibility warnings. `topLogprobs` and ASCII `userID` are validated before HTTP.

Strict tools, seed, topK, penalties and stop sequences produce unsupported warnings. Image URLs are limited to 8192 UTF-16 units, supported inline formats are JPEG/PNG/GIF/WebP, and `fileData` image mode is rejected for Responses. Incomplete/failed/abruptly-ended streams retain their terminal distinction. Model-level behavior is mock-tested; no live-service guarantee is implied.
