# streamlit_app/page_methodology.py
# =====================================================================
# Methodology Page — explains the research question and implementation
# =====================================================================

import streamlit as st


def render() -> None:
    """Render the Methodology page."""

    st.header("Methodology")

    # ── Research Question ────────────────────────────────────────────
    st.subheader("Research Question")
    st.markdown(
        """
        **Can we improve the performance of a standard momentum trading strategy
        by detecting when the trade is crowded and reducing exposure accordingly?**

        Momentum strategies — buying recent winners and selling recent losers —
        are among the most well-documented anomalies in finance
        (Jegadeesh & Titman, 1993). However, when too many investors pile into the
        same momentum trade, the resulting crowding can lead to sharp reversals
        and large drawdowns.

        Lou & Polk (2021) propose **comomentum** — the average pairwise abnormal
        correlation among extreme momentum stocks — as a real-time measure of this
        crowding. High comomentum signals that arbitrageurs are trading in lockstep,
        predicting weaker future momentum returns.

        This project replicates and extends their framework: we construct the
        comomentum signal, then use it to **time** the momentum strategy by scaling
        down exposure when crowding is high and scaling up when it is low.
        """
    )

    st.divider()

    # ── Implementation Pipeline ──────────────────────────────────────
    st.subheader("Implementation Pipeline")

    # Step 1 — Momentum Signal
    st.markdown("#### Step 1 · Momentum Signal Construction")
    st.markdown(
        """
        At each week *t*, we compute a rolling momentum score for every stock
        using a **48-week lookback** window and a **4-week skip** of the most
        recent weeks (to avoid short-term reversal contamination). A stock must
        have all 52 weeks of returns present to receive a score.
        """
    )
    st.latex(
        r"\text{MOM}_{i,t} = \prod_{k=t-51}^{t-4} (1 + r_{i,k}) - 1"
    )
    st.markdown(
        """
        The raw scores are then **cross-sectionally z-scored** at each week to
        produce standardised momentum exposures used in the next step.
        """
    )

    # Step 2 — Fama-MacBeth
    st.markdown("#### Step 2 · Fama–MacBeth Factor Returns")
    st.markdown(
        """
        Following Fama & MacBeth (1973), at each week *t* we run a cross-sectional
        regression of stock returns on their lagged standardised momentum:
        """
    )
    st.latex(
        r"r_{i,t} = \alpha_t + \gamma_t \, \text{MOM}^{*}_{i,t-1} + \varepsilon_{i,t}"
    )
    st.markdown(
        """
        The slope **γ_t** is the weekly momentum factor return — the premium
        earned by high-momentum stocks. Its time-series *t*-statistic tests
        whether momentum commands a significant risk premium.
        """
    )

    # Step 3 — Comomentum
    st.markdown("#### Step 3 · Comomentum Construction")
    st.markdown(
        """
        At each week *t*:

        1. **Decile sort** — all listed stocks with valid momentum are sorted;
           the top 10 % form the *winner* decile and bottom 10 % form the *loser* decile.
        2. **Rolling FF3 residuals** — for each decile stock, regress the last
           52 weeks of returns on Mkt-RF, SMB, HML (Fama–French 3 factors) to
           obtain abnormal returns (residuals). Betas vary over time via a
           rolling window (Lewellen & Nagel, 2006).
        3. **Pairwise correlations** — compute the full K × K correlation matrix
           of residuals within each decile; average all K(K−1)/2 unique pairs.
        4. **Aggregate** — comomentum is the simple average of the winner and
           loser decile correlations:
        """
    )
    st.latex(
        r"\text{CoMOM}_t = \tfrac{1}{2}"
        r"\left(\text{CoMOM}_{W,t} + \text{CoMOM}_{L,t}\right)"
    )
    st.markdown(
        """
        A **high CoMOM** value means extreme momentum stocks are moving together
        beyond what the Fama–French factors explain — a signature of crowded
        trading by momentum arbitrageurs.
        """
    )

    # Step 4 — Adjusted Momentum
    st.markdown("#### Step 4 · Adjusted Momentum Strategy")
    st.markdown(
        """
        We scale the **factor return series** (not the exposures, since
        cross-sectional z-scoring would undo any uniform scaling of exposures):

        1. Compute an **expanding-window percentile rank** of CoMOM at each week
           (using only past data — no look-ahead).
        2. Compute the scaling factor:
        """
    )
    st.latex(r"s_t = 2 - \text{PctRank}_{t-1}(\text{CoMOM})")
    st.markdown(
        """
        This maps into a **[1, 2]** range:
        - Low comomentum (rank ≈ 0) → *s* ≈ 2 → **increase** the momentum bet
        - High comomentum (rank ≈ 1) → *s* ≈ 1 → **maintain at par** (reduce relative exposure)

        3. Apply the scaling to produce adjusted factor returns:
        """
    )
    st.latex(r"\gamma^{\text{adj}}_t = s_t \cdot \gamma_t")

    st.divider()

    # ── Data ─────────────────────────────────────────────────────────
    st.subheader("Data")
    st.markdown(
        """
        | Dataset | Description | Dimensions |
        |---------|-------------|------------|
        | `US_Returns.csv` | Weekly stock returns | 1,513 weeks × 7,261 stocks |
        | `US_live.csv` | Listed/not-listed indicator | 1,513 × 7,261 |
        | `US_Dates.xlsx` | Weekly date labels | 1,513 × 1 |
        | `US_Names.xlsx` | Stock ticker names | 1 × 7,261 |
        | `FamaFrench.csv` | FF3 factors + RF | 1,513 × 5 |

        **Sample period:** January 1992 – December 2020
        """
    )

    st.divider()

    # ── References ───────────────────────────────────────────────────
    st.subheader("References")
    st.markdown(
        """
        - Lou, D. & Polk, C. (2021). *Comomentum: Inferring Arbitrage Activity
          from Return Correlations*. Review of Financial Studies, 35(7), 3272–3302.
        - Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and
          Selling Losers*. Journal of Finance, 48(1), 65–91.
        - Fama, E. & MacBeth, J. (1973). *Risk, Return, and Equilibrium:
          Empirical Tests*. Journal of Political Economy, 81(3), 607–636.
        - Lewellen, J. & Nagel, S. (2006). *The Conditional CAPM Does Not Explain
          Asset-Pricing Anomalies*. Journal of Financial Economics, 82(2), 289–314.
        """
    )
