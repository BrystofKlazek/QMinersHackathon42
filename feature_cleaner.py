import pandas as pd

INPUT_PATH = "data/features_many.csv"
OUTPUT_PATH = "data/features_v1.csv"

meta_and_targets = [
    "orig_index",
    "Season",
    "Date",
    "HID",
    "AID",
    "market_type",
    "y_home_win",
    "y_away_win",
    "y_draw",
]

# V1 feature set - trimmed, no obvious redundancies
feature_cols_v1 = [
    # Odds
    "oddsH", "oddsA",

    # Elo (just the diff, not the elos themselves?)
    "elo_diff",

    # Season standings
    "pts_pg_h", "pts_pg_a",
    "gd_pg_h", "gd_pg_a",
    "games_season_h", "games_season_a",

    # Schedule and fatigue
    "rest_days_h", "rest_days_a", "rest_diff",
    "back_to_back_h", "back_to_back_a",
    "home_streak_h", "away_streak_a",

    # Goals (overall, fast and slow EMAs)
    "h_gf_fast", "h_ga_fast", "h_gf_slow", "h_ga_slow",
    "a_gf_fast", "a_ga_fast", "a_gf_slow", "a_ga_slow",

    # Period scoring pattern  and 3rd-period strength
    "h_gf_p1_share", "h_gf_p2_share", "h_gf_p3_share",
    "a_gf_p1_share", "a_gf_p2_share", "a_gf_p3_share",
    "h_third_period_goal_diff", "a_third_period_goal_diff",
    "third_period_goal_diff_diff",

    # OT/SO behaviour
    "h_ot_rate_fast", "h_ot_win_rate_fast",
    "a_ot_rate_fast", "a_ot_win_rate_fast",

    # Advanced stats – core bits
    # SOG
    "sog_share_fast_h", "sog_share_fast_a", "sog_share_fast_diff",
    # PIM
    "pim_for_fast_h", "pim_for_fast_a", "pim_for_fast_diff",
    # Special teams
    "ppg_fast_h", "ppg_fast_a", "shg_fast_h", "shg_fast_a",
    # Goaltending
    "svpct_fast_h", "svpct_fast_a", "svpct_fast_sum",
    # Faceoffs 
    "fo_share_fast_diff",
]

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)

    missing = [c for c in meta_and_targets + feature_cols_v1 if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INPUT_PATH}: {missing}")

    cols_to_keep = meta_and_targets + feature_cols_v1
    df_v1 = df[cols_to_keep].copy()

    df_v1.to_csv(OUTPUT_PATH, index=False)
    print(f"features_v1 written to {OUTPUT_PATH}")
    print("Shape:", df_v1.shape)

