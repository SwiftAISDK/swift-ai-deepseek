# Changelog

## 1.1.0

- Portable medium maps to low and xhigh maps to high against low/high/max, with matching generation/stream warnings; explicit provider effort retains existing canonical normalization/precedence. Explicit thinking settings no longer suppress independent portable effort mapping; provider-default leaves the field omitted, matching source request policy. Reviewed upstream: @ai-sdk/deepseek 3.0.73 to 3.0.77. Validation passed: node Scripts/test-packages.mjs (all 52 owning package suites and the compatibility facade); node Scripts/sdk.mjs generate --check; node --test Scripts/tests/*.test.cjs (27 passed); docs-site check and build (100 pages). Live provider calls were not run.
  Compatibility: Minor release for additive native API; existing static settings and call sites remain compatible. Updated direct dependency minimums are recorded in the manifest. Existing documented JS/framework and native product gaps remain; version-delta review does not imply full parity or live-service testing. Explicit custom authentication skips an unused credential supplier, a deliberate native callback-order adaptation.
