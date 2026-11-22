"""Explicit research switches; default behavior retains every evidence safeguard."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DiscoveryPolicy:
    """One-factor ablations, recorded in checkpoint bindings and experiment logs.

    Disabling evidence authorization is an experimental policy, not a safe
    production default. It removes agreement-based edit permission, never SQL
    read-only enforcement, action validation, execution budgets or final replay.
    """

    challenge_search: bool = True
    blind_audit: bool = True
    evidence_authorization: bool = True

    def __post_init__(self):
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("DISCOVERY_POLICY_REQUIRES_BOOLEANS")

    def to_dict(self) -> dict:
        return asdict(self)


FULL_POLICY = DiscoveryPolicy()
