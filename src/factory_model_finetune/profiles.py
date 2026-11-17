"""The seven factory profiles as the prompts see them: description, conventions and review rubric.

The base, Python and data-science conventions are the factory's own text. The other profiles'
conventions and every description and rubric are written to match their style; real examples
harvested from the logs carry the factory's actual wording.
"""

from dataclasses import dataclass
from pathlib import Path

CONVENTIONS_DIR = Path(__file__).parent / "conventions"


@dataclass(frozen=True)
class Profile:
    """Immutable configuration for a specific coding profile.

    Attributes:
        name: Unique identifier for the profile (e.g., 'python', 'ts-react-webapp').
        description: Human-readable summary of the profile's scope and stack.
        convention_files: Ordered list of markdown filenames in CONVENTIONS_DIR
            to be concatenated for the full convention text.
        rubric: Markdown-formatted review criteria specific to this profile.
    """

    name: str
    description: str
    convention_files: tuple[str, ...]
    rubric: str

    @property
    def conventions(self) -> str:
        """Load and concatenate convention files in the order specified.

        Reads each file from CONVENTIONS_DIR using the names in convention_files,
        joins them with newlines, and strips trailing whitespace. This ensures
        shared conventions (like 'base') are applied before profile-specific ones.

        Returns:
            The combined convention text as a single string.
        """
        parts = (CONVENTIONS_DIR / f"{name}.md" for name in self.convention_files)
        return "\n".join(part.read_text() for part in parts).strip()


PROFILE_INFO: dict[str, Profile] = {
    p.name: p
    for p in (
        Profile(
            "python",
            "A plain Python package or CLI tool: src layout, uv, pytest, ruff and pyright.",
            ("base", "python"),
            "- Every public function has type hints and a docstring.\n"
            "- Errors name the bad value; nothing is swallowed.\n"
            "- Tests cover each acceptance criterion and edge case.",
        ),
        Profile(
            "python-fastapi-service",
            "A FastAPI web service with Pydantic models, a service layer and TestClient tests.",
            ("base", "python", "fastapi"),
            "- Handlers are thin and return Pydantic models with correct status codes.\n"
            "- Input is validated at the boundary; errors are specific.\n"
            "- Each endpoint has tests for success and each failure status.",
        ),
        Profile(
            "python-data-pipeline",
            "A batch data pipeline in Python: typed stages, idempotent runs, schema validation.",
            ("base", "python", "pipeline"),
            "- Stages are pure and idempotent; I/O stays at the edges.\n"
            "- Schemas are validated at stage boundaries.\n"
            "- Tests cover empty input, missing columns and duplicate keys.",
        ),
        Profile(
            "data-science-analysis",
            "A reproducible Python data analysis: tested modules, seeded models, honest reports.",
            ("base", "python", "data_science"),
            "- No leakage: splits happen before any fitting.\n"
            "- Seeds are explicit and every reported number comes from tested code.\n"
            "- Reports state caveats and limitations plainly.",
        ),
        Profile(
            "llm-app-agent",
            "An LLM application or agent in Python: prompts, tool calls, output validation.",
            ("base", "python", "llm_app"),
            "- Model output is validated before use and malformed output is handled.\n"
            "- Prompts live apart from logic; model calls go through one wrapper.\n"
            "- Tests use a fake model client and never touch the network.",
        ),
        Profile(
            "ts-react-webapp",
            "A TypeScript React single-page web app built with Vite and pnpm.",
            ("base", "typescript"),
            "- No untyped `any`; external data is validated at the boundary.\n"
            "- Components are small; data fetching is separate from presentation.\n"
            "- Tests check behaviour, not layout or styling.",
        ),
        Profile(
            "nextjs-fullstack",
            "A full-stack Next.js app (App Router) with server actions and a database.",
            ("base", "typescript", "nextjs"),
            "- Server-only code never leaks into client components.\n"
            "- Route handlers and actions validate input and return typed results.\n"
            "- Critical flows have a smoke test; logic has unit tests.",
        ),
    )
}
