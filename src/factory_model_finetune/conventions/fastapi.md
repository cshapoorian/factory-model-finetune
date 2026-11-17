# FastAPI service conventions
- One router module per resource; handlers stay thin and call plain functions in a service module.
- Request and response bodies are Pydantic models; never return raw dicts from handlers.
- Raise `HTTPException` (or a registered exception handler) with a specific status code and a message naming the bad value.
- Dependencies (database session, settings, auth) come in through `Depends`, so tests can override them.
- Test with `TestClient` against the real app object; cover each status code a handler can return.
