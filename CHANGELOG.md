# Changelog

All notable changes to the Griptape Nodes Minimax Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and each version is
the `library_version` in `minimax/griptape_nodes_library.json`.
**Breaking** marks a change that can stop a saved workflow from working without edits.

## [Unreleased]

<!-- Entries go in changelog.d/, one file each. See changelog.d/README.md. -->

## [0.2.0] - 2026-11-06

### Changed

- **Breaking:** Minimax nodes stop the run with an error where they used to end it quietly with
  empty outputs. This covers a rejected key on `Minimax Text-to-Image` and the three video nodes,
  a video task submitted with no task ID, a finished video with no download link, an image response
  with no images, and music that is not finished or has no audio URL. Workflows that read the empty
  outputs need to handle the error instead.
  [#13](https://github.com/griptape-ai/griptape-nodes-library-minimax/issues/13)
- Minimax nodes report failures as structured node errors. The editor shows Minimax's own message,
  with the error code, task ID and response under More info. Messages no longer carry the node name.
  A missing or refused API key links to Settings and to Minimax's key page, and a missing key is
  reported first and on its own. The library now requires Griptape Nodes 0.105.0 or later.
  [#13](https://github.com/griptape-ai/griptape-nodes-library-minimax/issues/13)

[Unreleased]: https://github.com/griptape-ai/griptape-nodes-library-minimax/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/griptape-ai/griptape-nodes-library-minimax/compare/fc789ae3078be8ac973dbc8d20a49efae9f9f715...v0.2.0
