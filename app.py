"""
Synthetic marketing spend vs. revenue with diminishing returns, filterable by date.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit

RNG = np.random.default_rng(42)

# Chart colors (blue theme)
SCATTER_PALETTE = px.colors.sequential.Blues[2:]  # skip very light swatches for contrast
FIT_LINE = "rgba(37, 99, 235, 0.95)"  # blue-600
LINE_SPEND = "#1d4ed8"  # blue-700
LINE_REVENUE = "#60a5fa"  # blue-400

# --- Synthetic data: diminishing returns R ≈ R_max * (1 - exp(-k * S)) + noise ---
R_MAX = 180_000.0
# Larger k => revenue approaches R_max faster (stronger saturation at a given spend).
K = 5.2e-5

START = dt.date(2024, 1, 1)
END = dt.date(2024, 12, 31)
N_DAYS = (END - START).days + 1


def saturation_revenue(spend: np.ndarray | float) -> np.ndarray:
    """Structural diminishing-returns curve used to generate revenue before noise."""
    return R_MAX * (1.0 - np.exp(-K * np.asarray(spend, dtype=float)))


def fit_saturation_curve(
    spend: np.ndarray,
    revenue: np.ndarray,
    *,
    min_points: int = 5,
) -> tuple[float, float] | None:
    """
    Fit R = a * (1 - exp(-b * spend)) to observed points via nonlinear least squares.
    Returns (a, b) or None if the fit cannot be computed.
    """
    s = np.asarray(spend, dtype=float)
    r = np.asarray(revenue, dtype=float)
    if len(s) < min_points or np.unique(s).size < 3:
        return None

    def model(sp: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * (1.0 - np.exp(-b * sp))

    a0 = float(np.nanmax(r) * 1.2) if np.nanmax(r) > 0 else 100_000.0
    b0 = 2e-5
    try:
        (a, b), _ = curve_fit(
            model,
            s,
            r,
            p0=[a0, b0],
            bounds=([1.0, 1e-9], [1e12, 0.05]),
            maxfev=50_000,
        )
        return float(a), float(b)
    except (RuntimeError, ValueError):
        return None


def fitted_revenue(spend: np.ndarray | float, a: float, b: float) -> np.ndarray:
    return a * (1.0 - np.exp(-b * np.asarray(spend, dtype=float)))


@st.cache_data
def make_marketing_data(
    start: dt.date,
    end: dt.date,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    # Daily marketing spend (lognormal-ish, bounded)
    base_spend = rng.lognormal(mean=np.log(8_000), sigma=0.35, size=n)
    seasonal = 1.0 + 0.15 * np.sin(np.linspace(0, 4 * np.pi, n))
    marketing_spend = np.clip(base_spend * seasonal, 2_000, 50_000)

    saturation = saturation_revenue(marketing_spend)

    # Small weekday uplift and noise
    weekday = pd.Series(dates).dt.weekday.values
    dow_factor = 1.0 + 0.03 * np.sin(weekday * 2 * np.pi / 7)
    noise = rng.normal(0, 4_500, size=n)

    revenue = np.clip(saturation * dow_factor + noise, 0, None)

    return pd.DataFrame(
        {
            "Date": dates.normalize(),  # type: ignore[union-attr]
            "marketing_spend": np.round(marketing_spend, 2),
            "revenue": np.round(revenue, 2),
        }
    )


def main() -> None:
    st.set_page_config(page_title="Marketing spend vs. revenue", layout="wide")
    st.title("Marketing spend vs. revenue (synthetic)")
    st.caption(
        "Revenue is generated with **diminishing returns**: extra spend adds less and less "
        "marginal revenue (saturation curve)."
    )

    df = make_marketing_data(START, END)

    min_d = df["Date"].min().date()
    max_d = df["Date"].max().date()

    with st.sidebar:
        st.header("Filter by date")
        date_range = st.slider(
            "Date range",
            min_value=min_d,
            max_value=max_d,
            value=(min_d, max_d),
            format="YYYY-MM-DD",
        )

    start_sel, end_sel = date_range
    mask = (df["Date"].dt.date >= start_sel) & (df["Date"].dt.date <= end_sel)
    filtered = df.loc[mask].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in range", len(filtered))
    c2.metric("Total spend", f"${filtered['marketing_spend'].sum():,.0f}")
    c3.metric("Total revenue", f"${filtered['revenue'].sum():,.0f}")

    tab_scatter, tab_ts = st.tabs(["Spend vs. revenue", "Time series"])

    with tab_scatter:
        fig = px.scatter(
            filtered,
            x="marketing_spend",
            y="revenue",
            color=filtered["Date"].dt.strftime("%Y-%m"),
            color_discrete_sequence=SCATTER_PALETTE,
            hover_data={"Date": "|%Y-%m-%d", "marketing_spend": ":$,.0f", "revenue": ":$,.0f"},
            labels={
                "marketing_spend": "Marketing spend ($)",
                "revenue": "Revenue ($)",
                "color": "Month",
            },
        )
        fit_params = None
        if len(filtered):
            s_lo = float(filtered["marketing_spend"].min())
            s_hi = float(filtered["marketing_spend"].max())
            if s_hi <= s_lo:
                s_hi = s_lo + 1.0
            s_grid = np.linspace(s_lo, s_hi, 400)
            fit_params = fit_saturation_curve(
                filtered["marketing_spend"].values,
                filtered["revenue"].values,
            )
            if fit_params is not None:
                a_fit, b_fit = fit_params
                fig.add_trace(
                    go.Scatter(
                        x=s_grid,
                        y=fitted_revenue(s_grid, a_fit, b_fit),
                        mode="lines",
                        name="Fitted curve",
                        line=dict(color=FIT_LINE, width=2.5),
                        hovertemplate=("Spend=$%{x:,.0f}<br>Fitted R=$%{y:,.0f}<extra></extra>"),
                    )
                )
        fig.update_layout(
            height=520,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        if fit_params is not None:
            a_fit, b_fit = fit_params
            st.caption(
                f"Blue curve: **nonlinear least-squares fit** "
                f"`R = {a_fit:,.0f} * (1 - exp(-{b_fit:.4e} * spend))`"
                " to the points in this date range."
            )
        elif len(filtered) < 5:
            st.caption(
                "Select a date range with at least **5** days with varied spend"
                " to show a fitted curve."
            )
        else:
            st.caption(
                "Could not fit a saturation curve to the current selection "
                "(try widening the date range or check for degenerate spend values)."
            )
        st.plotly_chart(fig, width="stretch")

    with tab_ts:
        daily = filtered.sort_values("Date")
        fig2 = px.line(
            daily,
            x="Date",
            y=["marketing_spend", "revenue"],
            labels={"value": "Amount ($)", "Date": "Date", "variable": "Series"},
            color_discrete_map={
                "marketing_spend": LINE_SPEND,
                "revenue": LINE_REVENUE,
            },
        )
        fig2.update_layout(height=420, legend_title_text="")
        st.plotly_chart(fig2, width="stretch")

    with st.expander("Preview data"):
        st.dataframe(filtered, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
