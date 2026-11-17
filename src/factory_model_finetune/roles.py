"""The factory's model roles and profiles, and how much of the dataset each role gets."""

from factory_model_finetune.errors import ConfigError

REASONING_ROLES = ("spec", "plan", "review")
OTHER_ROLES = (
    "evidence",
    "trivial_edit",
    "docs_pass",
    "summarize",
    "commit_message",
    "pr_body",
    "detect_profile",
    "memory",
)
ALL_ROLES = REASONING_ROLES + OTHER_ROLES

PROFILES = (
    "python",
    "python-fastapi-service",
    "python-data-pipeline",
    "data-science-analysis",
    "llm-app-agent",
    "ts-react-webapp",
    "nextjs-fullstack",
)

# Share of the dataset's tokens per role. Weighted toward the hard roles the factory depends on
# most (spec, plan, review); every role keeps a share so none is forgotten.
DEFAULT_ROLE_MIX: dict[str, float] = {
    "spec": 22,
    "plan": 22,
    "review": 22,
    "evidence": 8,
    "trivial_edit": 5,
    "docs_pass": 5,
    "summarize": 5,
    "commit_message": 3,
    "pr_body": 4,
    "detect_profile": 2,
    "memory": 2,
}


def parse_role_mix(text: str) -> dict[str, float]:
    """Parse a comma-separated string of role=weight pairs and merge them into the default mix.

    The input string format is ``role1=weight1,role2=weight2``. Only the specified roles
    have their weights updated; unspecified roles retain their ``DEFAULT_ROLE_MIX`` values.

    Args:
        text: Comma-separated ``role=weight`` pairs. Roles must be in ``ALL_ROLES``.

    Returns:
        A new dictionary mapping role names to their assigned float weights.

    Raises:
        ConfigError: If a role is unknown, a weight is not a valid number, or a weight
            is not positive.
    """
    mix = dict(DEFAULT_ROLE_MIX)
    for part in filter(None, (p.strip() for p in text.split(","))):
        role, _, raw_weight = part.partition("=")
        if role not in mix:
            raise ConfigError(f"unknown role {role!r} in --role-mix; expected one of {ALL_ROLES}")
        try:
            weight = float(raw_weight)
        except ValueError:
            raise ConfigError(f"weight for {role!r} must be a number, got {raw_weight!r}") from None
        if weight <= 0:
            raise ConfigError(f"weight for {role!r} must be positive so the role stays represented")
        mix[role] = weight
    return mix
