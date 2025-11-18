import numpy as np
import pandas as pd
from collections import defaultdict


def add_elo_and_form_features(
    df: pd.DataFrame,
    base_rating: float = 1500.0,
    k_factor: float = 20.0,
) -> pd.DataFrame:
    """
    For each game, before updating with this game, we store:
        ELO_H, ELO_A: current Elo ratings of home and away
        H_gp_before, H_gf_mean_before, H_ga_mean_before
        A_gp_before, A_gf_mean_before, A_ga_mean_before
    Then we update:
        Elo ratings based on H/D/A result
        cumulative GF/GA + games for each team
    """

    df = df.copy()

    df["__orig_idx"] = np.arange(len(df))
    df = df.sort_values(["Season", "__orig_idx"]).reset_index(drop=True)

    # elo and form state
    ratings = defaultdict(lambda: base_rating)
    games_played = defaultdict(int)
    gf_sum = defaultdict(float)
    ga_sum = defaultdict(float)

    gf_ema = defaultdict(float)  # goals-for 
    ga_ema = defaultdict(float)  # goals-against
    alpha = 0.2  #MOVING AVERAGe (EMA) smoothing factor - WE WILL BUILD MOVING AVERAGE

    n = len(df)
    ELO_H = np.zeros(n, dtype=float)
    ELO_A = np.zeros(n, dtype=float)

    H_gp_before = np.zeros(n, dtype=float)
    H_gf_mean_before = np.zeros(n, dtype=float)
    H_ga_mean_before = np.zeros(n, dtype=float)

    A_gp_before = np.zeros(n, dtype=float)
    A_gf_mean_before = np.zeros(n, dtype=float)
    A_ga_mean_before = np.zeros(n, dtype=float)

    for i, row in df.iterrows():
        hid = int(row["HID"])
        aid = int(row["AID"])
        hs = float(row["HS"])
        a_s = float(row["AS"])

        # current Elo
        Rh = ratings[hid]
        Ra = ratings[aid]

        # current form
        gh_played = games_played[hid]
        ga_played = games_played[aid]

        #current EMA form for each team before this game
        h_gf_form = gf_ema[hid]  #goals-for
        h_ga_form = ga_ema[hid]  #goals-against
        a_gf_form = gf_ema[aid]
        a_ga_form = ga_ema[aid]

        ELO_H[i] = Rh
        ELO_A[i] = Ra

        H_gp_before[i] = gh_played
        H_gf_mean_before[i] = h_gf_form
        H_ga_mean_before[i] = h_ga_form

        A_gp_before[i] = ga_played
        A_gf_mean_before[i] = a_gf_form
        A_ga_mean_before[i] = a_ga_form
       
        # outcome for Elo
        H_flag = int(row["H"])
        D_flag = int(row["D"])
        A_flag = int(row["A"])

        if H_flag + D_flag + A_flag != 1:
            raise ValueError(f"Invalid H/D/A encoding at row {i}")

        if H_flag == 1:
            s_h, s_a = 1.0, 0.0
        elif D_flag == 1:
            s_h, s_a = 0.5, 0.5
        else:  
            s_h, s_a = 0.0, 1.0

        diff = Rh - Ra
        Eh = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        Ea = 1.0 - Eh

        ratings[hid] = Rh + k_factor * (s_h - Eh)
        ratings[aid] = Ra + k_factor * (s_a - Ea)

        games_played[hid] += 1
        games_played[aid] += 1

        gf_sum[hid] += hs
        ga_sum[hid] += a_s

        gf_sum[aid] += a_s
        ga_sum[aid] += hs

        # EMA update AFTER the game (so next match sees updated form)
        gf_ema[hid] = (1 - alpha) * gf_ema[hid] + alpha * hs
        ga_ema[hid] = (1 - alpha) * ga_ema[hid] + alpha * a_s

        gf_ema[aid] = (1 - alpha) * gf_ema[aid] + alpha * a_s
        ga_ema[aid] = (1 - alpha) * ga_ema[aid] + alpha * hs


    # attach ELO-LIKE features
    df["ELO_H"] = ELO_H
    df["ELO_A"] = ELO_A

    df["H_gp_before"] = H_gp_before
    df["H_gf_mean_before"] = H_gf_mean_before
    df["H_ga_mean_before"] = H_ga_mean_before

    df["A_gp_before"] = A_gp_before
    df["A_gf_mean_before"] = A_gf_mean_before
    df["A_ga_mean_before"] = A_ga_mean_before

    df["ELO_diff"] = df["ELO_H"] - df["ELO_A"]
    df["gf_mean_diff"] = df["H_gf_mean_before"] - df["A_gf_mean_before"]
    df["ga_mean_diff"] = df["H_ga_mean_before"] - df["A_ga_mean_before"]
    df["gp_diff"] = df["H_gp_before"] - df["A_gp_before"]

    current_season = None
    for i, row in df.iterrows():
        season = int(row["Season"])
        if season != current_season:
            current_season = season
            # reset ratings or at least shrink after seasons
            for tid in list(ratings.keys()):
                ratings[tid] = 0.3 * ratings[tid] + 0.7 * base_rating

    # return in original order
    df = df.sort_values("__orig_idx").drop(columns=["__orig_idx"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = pd.read_csv("../data/cleaned_games.csv")
    df = add_elo_and_form_features(df)
    df.to_csv("../data/cleaned_games_with_elo.csv", index=False)
    print("Saved to ../data/cleaned_games_with_elo.csv")
    print(df[[
        "Season", "HID", "AID",
        "ELO_H", "ELO_A",
        "H_gp_before", "H_gf_mean_before", "H_ga_mean_before",
        "A_gp_before", "A_gf_mean_before", "A_ga_mean_before"
    ]].head())

