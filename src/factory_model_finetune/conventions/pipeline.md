# Data pipeline conventions
- Each stage is a pure function from one typed dataset to the next; I/O lives only at the edges.
- Pipelines are idempotent: re-running with the same input gives the same output and never duplicates rows.
- Validate schemas at every stage boundary and fail loudly with the column and value that broke it.
- Process large inputs in chunks or lazily; never assume a file fits in memory.
- Test stages on tiny in-memory fixtures, including empty input, missing columns and duplicate keys.
