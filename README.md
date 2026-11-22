<!-- Generated distribution; edit the canonical repository. -->
# swift-ai-deepseek 1.3.1

Independent Swift-native port of Vercel AI SDK. Source and upstream dispositions are in PROVENANCE.json and Porting/.

## Distribution manifest

Add this package to your SwiftPM dependencies:

```swift
.package(url: "https://github.com/SwiftAISDK/swift-ai-deepseek.git", from: "1.3.1")
```

Products: `SwiftAIDeepSeek`.
Contribute and report issues in the canonical SwiftAISDK repository.

## Authored usage guide

# DeepSeek for Swift

`SwiftAIDeepSeek` adds DeepSeek to [SwiftAISDK](https://github.com/SwiftAISDK/SwiftAISDK), an independent Swift port of the Vercel AI SDK. It ports `@ai-sdk/deepseek` and supports text generation and provider files.

Provider guide with models, options, streaming and errors:

- [DeepSeek](https://swiftaisdk.github.io/SwiftAISDK/providers/deepseek/)

## Usage

Add the `SwiftAI` product from `swift-ai-core` and the `SwiftAIDeepSeek` product from this package to your target, then create the provider:

```swift
import SwiftAI
import SwiftAIDeepSeek

let provider = try AIProviders.deepSeek(configuration: DeepSeekProviderSettings())
```

By default the provider reads `DEEPSEEK_API_KEY` from the environment; values set in `DeepSeekProviderSettings` take precedence.

To use every provider through one dependency, add the complete [SwiftAISDK](https://github.com/SwiftAISDK/SwiftAISDK) package instead; see [Install](https://swiftaisdk.github.io/SwiftAISDK/getting-started/install/).

## Contributing

This package is exported from `Packages/swift-ai-deepseek` in the canonical [SwiftAISDK](https://github.com/SwiftAISDK/SwiftAISDK) repository. Open issues and pull requests there.
