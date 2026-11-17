# Next.js conventions
- App Router: server components by default; add `"use client"` only where state or browser APIs need it.
- Route handlers and server actions validate input with a schema and return typed results.
- Keep database and secret access in server-only modules; never import them from client components.
- Configuration comes from environment variables, validated once at startup and documented in the README.
- Add one Playwright smoke test per critical user flow; unit-test the logic behind it.
