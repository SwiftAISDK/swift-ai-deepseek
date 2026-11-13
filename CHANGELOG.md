# Changelog

## 1.2.0

- Add explicit Responses factories, typed options and web search without changing existing Chat Completions defaults. Preserve ordered reasoning/tool replay, structured output, image restrictions, usage, web-search action metadata and distinct failed/incomplete/interrupted stream outcomes. Reviewed npm delta: @ai-sdk/deepseek 3.0.81 to 3.0.86. Validation: `node Scripts/test-packages.mjs` passed all 52 owning suites and 1731 facade tests; tooling 27/27, generation/isolation checks, standalone exports and 106 minimum/latest consumers, documentation check and 101-page build passed. Publication requires green exact-source Swift 6.3.3 CI. Paid/live verification and full-package parity are not claimed.
  Compatibility: Additive native APIs or capabilities under the project minor-release convention. npm baselines and Swift SemVer are independent; this reviewed delta does not establish full-package or live-service parity. Required direct minimum: swift-ai-core 1.2.0 to 1.3.0 for request URL metadata.
