# Changelog

## 1.2.1

- Align the versioned request identity with the reviewed published package while retaining native open String model IDs and existing request, response, option, error, stream and telemetry behavior. Derived TypeScript model-ID/catalog reorganizations preserve the existing native runtime policies. Reviewed npm delta: @ai-sdk/deepseek 3.0.86 to 3.0.87. Validation passed: node Scripts/test-packages.mjs (55 owning packages, 2451 owning tests and 1736 compatibility tests), generation checks, 27 tooling tests, isolation/version checks, deterministic exports and minimum-dependency consumers. Documentation check/build passed (104 pages). Publication is gated by exact-source Swift 6.3.3 CI; subsequent public refs and clients are recorded in Docs/ReleaseVerification20261115.json. Paid/live-service checks were not run; this delta does not claim full-package parity.
  Compatibility: Compatible versioned request identity correction; existing public selectors and dependency minimums remain. npm baselines and Swift SemVer are independent.
