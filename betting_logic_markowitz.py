import numpy as np
import pandas as pd


def bet_stats(p, odds):
    if odds <= 0 or p <= 0 or p >= 1:
        return 0.0, 0.0

    win_ret = odds - 1.0
    lose_ret = -1.0

    mu = p * odds - 1.0

    er2 = p * (win_ret ** 2) + (1 - p) * (lose_ret ** 2)
    sigma2 = er2 - mu ** 2

    sigma2 = max(sigma2, 1e-12)
    return mu, sigma2


def build_daily_portfolio(df_day,
                          bankroll,
                          min_edge=0.0,
                          daily_risk_cap=0.10):
    candidates = []

    for idx, row in df_day.iterrows():
        oddsH = row["oddsH"]
        oddsA = row["oddsA"]
        p_h = row["p_hat_h"]
        p_a = 1.0 - p_h

        mu_h, sigma2_h = bet_stats(p_h, oddsH)
        if mu_h > min_edge:
            candidates.append({
                "game_index": idx,
                "side": "home",
                "odds": oddsH,
                "p_model": p_h,
                "mu": mu_h,
                "sigma2": sigma2_h,
            })
        
        mu_a, sigma2_a = bet_stats(p_a, oddsA)
        if mu_a > min_edge:
            candidates.append({
                "game_index": idx,
                "side": "away",
                "odds": oddsA,
                "p_model": p_a,
                "mu": mu_a,
                "sigma2": sigma2_a,
            })

    if not candidates:
        return []  

    for c in candidates:
        c["w_raw"] = c["mu"] / c["sigma2"]

    candidates = [c for c in candidates if c["w_raw"] > 0]
    if not candidates:
        return []

    total_w_raw = sum(c["w_raw"] for c in candidates)
    if total_w_raw <= 0:
        return []

    scale = daily_risk_cap / total_w_raw
    bets = []
    for c in candidates:
        w = c["w_raw"] * scale 
        stake = w * bankroll
        bets.append({
            "game_index": c["game_index"],
            "side": c["side"],
            "odds": c["odds"],
            "p_model": c["p_model"],
            "mu": c["mu"],
            "sigma2": c["sigma2"],
            "weight": w,
            "stake": stake,
        })

    return bets


