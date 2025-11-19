import pandas as pd
import numpy as np

def implied_probs_two_way(oddsH, oddsA):
    invH = 1.0 / oddsH
    invA = 1.0 / oddsA
    s = invH + invA
    return invH / s, invA / s 


def kelly_fraction(p, odds, kelly_scale=0.25):
    b = odds - 1.0
    edge = p * odds - 1.0  
    if edge <= 0:
        return 0.0, edge
    f_full = edge / b  
    f = kelly_scale * f_full
    return f, edge


def decide_bets_for_day(
    df_day,
    bankroll,
    kelly_scale=0.25,
    max_risk_per_bet=0.02,
    daily_risk_cap=0.10,
    min_edge=0.01
):
    bets = []

    for idx, row in df_day.iterrows():
        oddsH = row["oddsH"]
        oddsA = row["oddsA"]
        p_h = row["p_hat_h"]
        p_a = 1.0 - p_h

        f_h, edge_h = kelly_fraction(p_h, oddsH, kelly_scale=kelly_scale)
        f_a, edge_a = kelly_fraction(p_a, oddsA, kelly_scale=kelly_scale)

        candidates = []
        if edge_h > min_edge and f_h > 0:
            candidates.append(("home", f_h, edge_h, oddsH, p_h))
        if edge_a > min_edge and f_a > 0:
            candidates.append(("away", f_a, edge_a, oddsA, p_a))

        if not candidates:

        side, f, edge, odds, p_model = max(candidates, key=lambda x: x[2])

        f = min(f, max_risk_per_bet)
        if f <= 0:
            continue

        bets.append(
            {
                "game_index": idx,
                "side": side,
                "f_fraction": f,
                "stake": f * bankroll,
                "odds": odds,
                "p_model": p_model,
                "edge": edge,
            }
        )

    total_f = sum(b["f_fraction"] for b in bets)
    if total_f > daily_risk_cap and total_f > 0:
        scale = daily_risk_cap / total_f
        for b in bets:
            b["f_fraction"] *= scale
            b["stake"] = b["f_fraction"] * bankroll

    return bets

