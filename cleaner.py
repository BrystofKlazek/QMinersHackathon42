import pandas as pd
import numpy as np

# Advanced stats 
ADV_COLS = [
    "H_SOG", "A_SOG",
    "H_BLK_S", "A_BLK_S",
    "H_HIT", "A_HIT",
    "H_BLK", "A_BLK",
    "H_FO", "A_FO",
]

# Goal-related columns 
GOAL_CORE_COLS = [
    "HS", "AS",
    "H_P1", "H_P2", "H_P3",
    "A_P1", "A_P2", "A_P3",
    "H_PPG", "A_PPG",
    "H_SHG", "A_SHG",
]

# Other basic numeric stats we want to keep 
BASIC_NUMERIC_COLS = [
    "H_PEN", "A_PEN",
    "H_MAJ", "A_MAJ",
    "H_SV", "A_SV",
    "H_PIM", "A_PIM",
]

OT_SO_COLS = ["H_OT", "A_OT", "H_SO", "A_SO"]


def classify_market_type(row) -> str:
    #Classify market based on odds -  no_odds, 2way, 3way, weird
    oh, od, oa = row["OddsH"], row["OddsD"], row["OddsA"]
    if oh == 0 and od == 0 and oa == 0:
        return "no_odds"
    if oh > 0 and oa > 0 and od == 0:
        return "2way"
    if oh > 0 and oa > 0 and od > 0:
        return "3way"
    return "weird"


def apply_manual_goal_fixes(df: pd.DataFrame) -> None:
    #Cahnging of the goals as is in Ivan's notebook:
    # Game 1: nb = 5919
    nb = 5919
    if nb in df.index:
        df.loc[nb, "H_P2"] = 1
        df.loc[nb, "A_P2"] = 1
        df.loc[nb, "H_P3"] = 1
        df.loc[nb, "A_P3"] = 1
    #We leave OT/SO as they are -  later we fill NaNs to 0 globally anyway.

    # Game 2: nb = 6487
    nb = 6487
    if nb in df.index:
        df.loc[nb, "H_P3"] = 2
        df.loc[nb, "A_P3"] = 1

    # Game 3: nb = 6939
    nb = 6939
    if nb in df.index:
        df.loc[nb, "H_P2"] = 1
        df.loc[nb, "A_P2"] = 1
        df.loc[nb, "H_P3"] = 1
        df.loc[nb, "A_P3"] = 0

def clean_games(input_path: str,
                out_all_path: str,
                out_two_way_path: str):
    df = pd.read_csv(input_path)

    # Drop obvious junk indices of the dataframe
    for col in ["Unnamed: 0", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Change weird shite to ints
    if "Season" in df.columns:
        df["Season"] = df["Season"].astype(int)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["H", "D", "A"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # apply the  mmanual goal fixes BEFORE we drop rows
    apply_manual_goal_fixes(df)

    # Classify the market type
    df["market_type"] = df.apply(classify_market_type, axis=1)

    # Droping of no-odds and weird-odds games
    no_odds = (df["market_type"] == "no_odds").sum()
    weird = (df["market_type"] == "weird").sum()
    if no_odds:
        print(f"Dropping {no_odds} games with no odds (OddsH=OddsD=OddsA=0).")
    if weird:
        print(f"Dropping {weird} games with weird odds pattern.")

    df = df[df["market_type"].isin(["2way", "3way"])].copy()

    # For 2-way markets, OddsD is structurally absent so I set it to NaN explicitly
    #so if some model accidentaly uses it it screams
    if "OddsD" in df.columns:
        df.loc[df["market_type"] == "2way", "OddsD"] = np.nan


    # ensure OT/SO columns exist and fill NaN to 0 (no OT/SO)
    for col in OT_SO_COLS:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    # This maps special NaN to B (Basic gejm)
    if "Special" in df.columns:
        df["Special"] = df["Special"].fillna("B")
    else:
        df["Special"] = "B"

    #Just keep track of the advancet stuff for 2009+

    for col in ADV_COLS:
        if col not in df.columns:
            print("err in cols you dumb shite")

    df["has_adv_stats"] = df[ADV_COLS].notna().any(axis=1)

    #NaN in goals to zero
    for col in GOAL_CORE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # NaN to 0 for PPG/SHG
    for col in ["H_PPG", "A_PPG", "H_SHG", "A_SHG"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Require HS and AS to exist (we can't use a game without final score)
    before_hsas = len(df)
    df = df.dropna(subset=["HS", "AS"])
    dropped_hsas = before_hsas - len(df)
    if dropped_hsas:
        print(f"Dropped {dropped_hsas} games missing HS/AS.")

    for col in BASIC_NUMERIC_COLS:
        if col not in df.columns:
            print("err in cols you dumb shite")

    numeric_subset = [c for c in BASIC_NUMERIC_COLS if c in df.columns]
    if numeric_subset:
        before_num = len(df)
        df = df.dropna(subset=numeric_subset, how="any")
        dropped_num = before_num - len(df)
        if dropped_num:
            print(f"Dropped {dropped_num} games with missing basic numeric stats.")

    # Sort chronologically
    sort_cols = ["Season"]
    if "Date" in df.columns:
        sort_cols.append("Date")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    print("After cleaning:")
    print("  Total games with odds:", len(df))
    print("  Seasons:", df["Season"].min(), "–", df["Season"].max())
    print("  2-way games:", (df["market_type"] == "2way").sum())
    print("  3-way games:", (df["market_type"] == "3way").sum())
    print("  Games with advanced stats:", df["has_adv_stats"].sum())

    # Save all usable markets (2-way + 3-way)
    df.to_csv(out_all_path, index=False)
    print(f"Saved cleaned all-markets file to: {out_all_path}")

    # Save pure 2-way subset for the main model
    df_two = df[df["market_type"] == "2way"].copy()
    df_two.to_csv(out_two_way_path, index=False)
    print(f"Saved cleaned 2-way file to: {out_two_way_path}")

    return df, df_two


if __name__ == "__main__":
    clean_games(
        input_path="data/games.csv",
        out_all_path="data/cleaned_games_all.csv",
        out_two_way_path="data/cleaned_games_two_way.csv",
    )

