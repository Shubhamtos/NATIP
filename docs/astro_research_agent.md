# AstroResearchAgent

`AstroResearchAgent` is an experimental shadow-mode research module for NATIP.

It calculates deterministic astronomical features using Skyfield and a locally
cached JPL ephemeris file, then stores raw values in the local Feature Store.
It is not a trading agent.

## Safety Contract

- It must not generate BUY or SELL signals.
- It must not modify position size.
- It must not override the Risk Agent.
- It must not send broker orders.
- `astro.shadow_only` must remain `true`.
- `astro.max_score_adjustment` must remain `0`.

## Configuration

Environment-backed settings:

- `NATIP_ASTRO_ENABLED=false`
- `NATIP_ASTRO_SHADOW_ONLY=true`
- `NATIP_ASTRO_MAX_SCORE_ADJUSTMENT=0`
- `NATIP_ASTRO_EPHEMERIS_PATH=data/astro/de421.bsp`

NATIP does not download the JPL ephemeris automatically. Place the cached
Skyfield-supported `.bsp` file at the configured path.

## Features

The module stores:

- body ecliptic longitudes
- lunar phase angle
- lunar illumination
- days from new moon
- days from full moon
- Mercury retrograde status
- planetary aspect distances
- ingress-event flags
- calculation version
- availability timestamp in Asia/Kolkata

All timestamps are converted to `Asia/Kolkata`.

## Research Pipeline

Use `app.research.astro_walk_forward.run_astro_walk_forward_research` for
validation-only comparison of:

- unchanged base NATIP strategy
- base-plus-astro experiment columns

The research utility reports transaction costs, slippage, sample size,
out-of-sample fold performance, false-discovery correction, regime stability
and contribution by year.

## Important Limitation

This module is unsupported as a causal forecasting method. Astronomical
correlations can be spurious and unstable. Any result must be treated as
research evidence only and must not be promoted into live trading rules without
separate statistical validation, risk review and production approval.
