# LLM app and agent conventions
- Keep prompts in their own files or constants, not scattered through logic; version them with the code.
- Every model call goes through one client wrapper that sets timeouts, retries and logs token usage.
- Validate model output against a schema before using it, and handle malformed output explicitly.
- Tools an agent can call are small, typed, and have clear failure messages the model can act on.
- Tests use a fake model client with canned replies; never call a real model in unit tests.
