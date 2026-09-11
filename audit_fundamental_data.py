"""Generate Screener fundamental-data coverage and feature audits."""

from __future__ import annotations

from app.fundamentals.audit import run_fundamental_audit


def main() -> None:
    """Run all fundamental-data audits."""

    result = run_fundamental_audit()
    print(f"coverage rows: {len(result['coverage'])}")
    print(f"feature audit rows: {len(result['feature_audit'])}")
    print(f"status: {result['status']}")


if __name__ == "__main__":
    main()

