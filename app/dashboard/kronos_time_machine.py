"""On-demand daily Time Machine panel, isolated from Kronos model outputs."""

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.intelligence.daily_time_machine import (
    VERSION,
    DailyTradePlan,
    simulate_daily_plan,
    validate_daily_plan,
)
from app.intelligence.kronos_forecast import KronosForecastResult


def render_daily_time_machine(
    result: KronosForecastResult, history: pd.DataFrame, root: Path
) -> None:
    """Render an explicitly requested, daily long-plan stress test beside a forecast."""
    st.markdown("### Time Machine — Daily Trade-Plan Stress Test")
    st.caption(
        f"Experimental · {result.symbol} · {result.horizon} future daily sessions · historical resampling, not another Kronos forecast."
    )
    st.info(
        "Conditional on your entry being filled. Frequencies are not calibrated probabilities or forecast accuracy. No orders or recommendation changes occur."
    )
    identity = f"{VERSION}-{result.market}-{result.symbol}-{result.signal_date}-{result.horizon}"
    with st.form(f"daily-tm-form-{identity}"):
        levels = st.columns(3)
        currency = "$" if result.market == "US" else "INR"
        entry = levels[0].number_input(
            f"Hypothetical entry ({currency})",
            min_value=0.01,
            value=round(result.latest_price, 2),
            step=0.01,
        )
        stop = levels[1].number_input(
            f"Stop-loss ({currency})",
            min_value=0.01,
            value=max(0.01, round(result.latest_price * 0.95, 2)),
            step=0.01,
        )
        target = levels[2].number_input(
            f"Target ({currency})",
            min_value=0.01,
            value=max(0.02, round(result.latest_price * 1.10, 2)),
            step=0.01,
        )
        controls = st.columns(4)
        paths = controls[0].selectbox("Simulation paths", [1000, 2000, 5000], index=1)
        block = controls[1].selectbox("Consecutive days per block", [3, 5, 10], index=1)
        costs = controls[2].number_input(
            "Charges per side (bps)", min_value=0.0, max_value=500.0, value=10.0
        )
        slip = controls[3].number_input(
            "Slippage per side (bps)", min_value=0.0, max_value=500.0, value=5.0
        )
        validation = st.checkbox("Also run historical walk-forward plan replay", value=False)
        run = st.form_submit_button("Run Daily Time Machine")
    key = f"daily-tm-result-{identity}"
    if run:
        try:
            plan = DailyTradePlan(
                entry,
                stop,
                target,
                horizon=result.horizon,
                paths=paths,
                block_size=block,
                cost_bps=costs,
                slippage_bps=slip,
            )
            with st.spinner("Resampling historical daily blocks and checking stop/target paths..."):
                simulation = simulate_daily_plan(history, plan, as_of=result.signal_date)
                replay = (
                    validate_daily_plan(history, plan, as_of=result.signal_date)
                    if validation
                    else pd.DataFrame()
                )
            st.session_state[key] = (simulation, plan, replay)
        except Exception as exc:
            st.error(f"Daily Time Machine could not run: {exc}")
            return
    saved = st.session_state.get(key)
    if not saved:
        st.caption(
            "The 5% stop and 10% target are editable examples, not recommended levels. Nothing runs until you press the button."
        )
        return
    simulation, plan, replay = saved
    excluded = simulation.get("excluded_bars", [])
    if excluded:
        st.warning(
            f"Excluded {len(excluded)} invalid/zero-volume historical candles. No prices were filled; bootstrap blocks and replay windows cannot cross those gaps."
        )
        with st.expander("Excluded historical candles"):
            st.dataframe(pd.DataFrame(excluded), hide_index=True)
    st.caption(
        f"Saved run: entry {plan.entry:,.2f}, stop {plan.stop:,.2f}, target {plan.target:,.2f} · as of {simulation['as_of']} · {simulation['history_rows']} historical candles. Rerun to apply edited inputs."
    )
    cols = st.columns(4)
    cols[0].metric("Target first — simulated", f"{simulation['frequencies']['TARGET']:.1%}")
    cols[1].metric("Stop first — simulated", f"{simulation['frequencies']['STOP']:.1%}")
    cols[2].metric("Neither by horizon", f"{simulation['frequencies']['NEITHER']:.1%}")
    cols[3].metric("Reward / risk before costs", f"{simulation['reward_risk']:.2f}")
    st.write(
        f"Simulated net return: mean **{simulation['net_mean_pct']:+.2f}%**, median **{simulation['net_median_pct']:+.2f}%** after selected costs and slippage."
    )
    st.line_chart(simulation["bands"])
    st.caption(
        "10th/median/90th percentile simulated closing-price paths; not confidence intervals. Price paths continue after an exit, but trade returns stop at exit."
    )
    with st.expander("Sample paths and assumptions"):
        st.line_chart(simulation["sample_paths"])
        st.write(
            "Long-only, fixed stop and target. Blocks preserve consecutive daily behaviour within each block, but not across block boundaries. Stop gaps fill at the worse opening price; favourable target gaps receive only the target price. If both levels are touched within a day, stop is counted first. No entry-fill likelihood, exchange circuit/halt handling, FX conversion or broker-specific fees are modelled. Stops cannot guarantee a maximum loss."
        )
        st.write(
            "History uses adjusted prices already supplied to Kronos and is cut at its signal date. More paths reduce simulation noise, not market uncertainty. No known news or earnings shock is explicitly modelled."
        )
    if not replay.empty:
        st.markdown("#### Historical walk-forward plan replay")
        st.caption(
            "Non-overlapping windows; each simulation uses only earlier candles. Relative stop/target distances are reapplied at the next real open. Not a backtest of Kronos forecast accuracy or a selected BUY strategy."
        )
        actual = float(replay["Target Hit"].mean())
        predicted = float(replay["Predicted Target Frequency"].mean())
        brier = float(((replay["Predicted Target Frequency"] - replay["Target Hit"]) ** 2).mean())
        st.write(
            f"{len(replay)} historical trades · target-first actual **{actual:.1%}** vs simulated mean **{predicted:.1%}** · target-event Brier **{brier:.3f}**."
        )
        if len(replay) < 30:
            st.warning(
                "Small historical validation sample: fewer than 30 non-overlapping trades. No calibrated probability claim is supported."
            )
        st.dataframe(replay, hide_index=True)
        for group in ("Year", "Regime"):
            st.dataframe(
                replay.groupby(group).agg(
                    Trades=("Target Hit", "size"),
                    Target_Hit_Rate=("Target Hit", "mean"),
                    Mean_Net_Return=("Net Return %", "mean"),
                ),
                use_container_width=True,
            )
        st.download_button(
            "Download historical replay",
            replay.to_csv(index=False),
            f"{result.symbol}_daily_time_machine_replay.csv",
            key=f"daily-tm-replay-{identity}",
        )
    elif validation:
        st.warning(
            "No completed non-overlapping historical windows were available after the 253-bar warm-up."
        )
    export = json.dumps(
        {
            "symbol": result.symbol,
            "version": simulation["version"],
            "as_of": simulation["as_of"],
            "plan": asdict(plan),
            "frequencies": simulation["frequencies"],
            "historical_rows": simulation["history_rows"],
            "excluded_bars": simulation.get("excluded_bars", []),
            "eligible_blocks": simulation.get("eligible_blocks"),
        },
        indent=2,
    )
    st.download_button(
        "Download Time Machine run configuration",
        export,
        f"{result.symbol}_daily_time_machine.json",
        key=f"daily-tm-config-{identity}",
    )
