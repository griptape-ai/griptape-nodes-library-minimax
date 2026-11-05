- **Breaking:** Minimax nodes stop the run with an error where they used to end it quietly with
  empty outputs. This covers a rejected key on `Minimax Text-to-Image` and the three video nodes,
  a video task submitted with no task ID, a finished video with no download link, an image response
  with no images, and music that is not finished or has no audio URL. Workflows that read the empty
  outputs need to handle the error instead.
  [#13](https://github.com/griptape-ai/griptape-nodes-library-minimax/issues/13)
