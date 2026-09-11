"""Build point-in-time fundamental features from processed Screener exports."""

from __future__ import annotations

from app.fundamentals.point_in_time import (
    build_candidate_v4_matrix,
    build_point_in_time_fundamentals,
)


def main() -> None:
    """Build the point-in-time fundamentals dataset and candidate V4 matrix."""

    pit = build_point_in_time_fundamentals()
    candidate = build_candidate_v4_matrix()
    print(f"point-in-time rows: {len(pit)}")
    print(f"candidate V4 matrix rows: {len(candidate)}")


if __name__ == "__main__":
    main()

