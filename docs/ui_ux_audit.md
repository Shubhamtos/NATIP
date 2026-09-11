# NATIP UI/UX Audit

## Current Issues

- Navigation was a horizontal radio list of analytical modules, so users had to already know where to start.
- There was no true home screen answering market posture, best opportunities, data health, and no-trade reasons.
- Advanced research modules were mixed with day-to-day decision workflows.
- Important states such as stale data, missing model files, empty scans, and research-only outputs were present but visually inconsistent.
- Existing pages used many raw metrics and tables before conclusions, increasing cognitive load for beginner users.
- Trading language was inconsistent across modules: BUY, Watch, Wait, setup status, and research-only diagnostics appeared at the same hierarchy.
- Large tables did not always explain how to interpret the rows before showing the data.
- Mobile use was limited by wide tab bars and dense controls.
- Risk and data-quality messages were often lower on the page than opportunity or chart content.
- Some research modules, especially astro and experimental screeners, needed clearer separation from validated decision logic.

## First Redesign Pass

- Added a simple application shell with sidebar navigation and a global header.
- Added a Home / Today's Decision Centre screen focused on market posture, opportunity snapshot, sector rotation, attention centre, and no-trade reasons.
- Preserved all existing analytical workflows and backend contracts.
- Grouped working legacy modules behind clearer product destinations.
- Added reusable visual components for badges, cards, empty states, freshness, and action language.
- Improved CSS tokens for a calmer Groww-inspired NATIP visual system without copying Groww assets.
- Kept research-only modules visibly separated from official recommendation logic.

## Known Gaps

- Full mobile bottom navigation is not implemented because Streamlit does not support a native persistent mobile tab bar.
- Watchlist, portfolio, paper trades, strategies, backtesting, decision history, health, and settings are implemented as product-ready shells that can be connected to persistence in later passes.
- Some existing analytical pages still contain dense legacy tables; they are preserved to avoid breaking working functionality.
