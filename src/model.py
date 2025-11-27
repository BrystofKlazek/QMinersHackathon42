import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn import metrics, datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

class Model:
    def __init__(self):
        # Tady se ukládají data do "self"     
        self.sensitivity = 0.19
        self.epochs = 4
        self.decorrelation_lambda = 0.3
        self.lambda_ =0.002

    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ):
        def clear_data( inc: pd.DataFrame) -> pd.DataFrame:

            
            """pd.reset_option('display.max_rows')
            pd.set_option('display.max_columns', None)"""

             
            df = inc.copy()
            #df.head()

        

            df[["OddsH", "OddsA"]] = df[["OddsH", "OddsA"]].replace(0, np.nan)

           
            df = df.dropna(subset=['OddsH'])
            

             
            #df.head() #dropping null values for odds because we need them for our model

             
            #df.info()

             
            #df = df.drop(columns = ["Date"]) #drop dates of games
            """print('Date' in df.columns)
            df.info()"""

             
            """(df['D'] == True).sum()
            df.info()"""

             
            """draw_list = df.loc[df["D"]]
            df.loc[(df['D'] == True) & (df["Special"].notna()) & (df["H_SO"].notna())] # draw is not clasified as winning in nájezdy
            """

             
            """df.loc[(df['D'] == True) & (df["Special"].notna())] #draws are classified as draw in over time circa 100/500"""

             
            """df.loc[(df['D'] == True)] #& (df["H_SO"].notna())] #aroud 400 draws with no overtime/ NO data on DRAW AND THEN NAJEZDY, #hypothesis: is draw divided with some year (OT and no OT?)"""

             

            #hypothesis: advanced features where measured after some year # there is 236 games where there has been actually najezdy and overtime

             
            """df.loc[df["H_SOG"].notna()]"""

             
            """count = (df.loc[6904:, 'H_SOG'].isna()).sum() #there is sth wrong, because they prob started with new year but hadnt done it perfectly
            print(count)"""

             
            """df.loc[df["H_BLK_S"].notna()]
            change_of_features = 6970"""

             
            """count = (df.loc[change_of_features:, 'H_BLK_S'].isna()).sum() #hypothesis confirmed: season 2009 was start of advanced features
            #split data in this year and do ranksum test separately? yes -> see results and and conclude from rank sum test
            print(count)"""

             
            #použit median misto nan pro svm metodu
            #A_SOG	H_BLK_S	A_BLK_S	H_HIT	A_HIT	H_BLK	A_BLK	H_FO	A_FO
            median_H_SOG = df['H_SOG'].median()
            median_A_SOG = df["A_SOG"].median()
            median_H_BLK_S = df['H_BLK_S'].median()
            median_A_BLK_S = df['A_BLK_S'].median()
            median_H_HIT = df['H_HIT'].median()
            median_A_HIT = df['A_HIT'].median()
            median_H_BLK = df['H_BLK'].median()
            median_A_BLK = df['A_BLK'].median()
            median_H_FO = df['H_FO'].median()
            median_A_FO = df['A_FO'].median()

            df["H_SOG"] = df["H_SOG"].fillna(median_H_SOG)
            df["A_SOG"] = df["A_SOG"].fillna(median_A_SOG)
            df["H_BLK_S"] = df["H_BLK_S"].fillna(median_H_BLK_S)
            df["A_BLK_S"] = df["A_BLK_S"].fillna(median_A_BLK_S)
            df["H_HIT"] = df["H_HIT"].fillna(median_H_HIT)
            df["A_HIT"] = df["A_HIT"].fillna(median_A_HIT)
            df["H_BLK"] = df["H_BLK"].fillna(median_H_BLK)
            df["A_BLK"] = df["A_BLK"].fillna(median_A_BLK)
            df["H_FO"] = df["H_FO"].fillna(median_H_FO)
            df["A_FO"] = df["A_FO"].fillna(median_A_FO)

            #df.info()

             
            #f.loc[(df['H_PPG'].isna())] #cca 20 zapásu chybí udáj o golech v přesilovce, a všechny mají draw true -> 0
            df['H_PPG'] = df['H_PPG'].fillna(0)
            df['A_PPG'] = df['A_PPG'].fillna(0)
            df["H_SHG"] = df["H_SHG"].fillna(0)
            df["A_SHG"] = df["A_SHG"].fillna(0)

             
            #df.info()

             
            H_SV_median = df['H_SV'].median()
            A_SV_median = df['A_SV'].median()
            df['H_SV'] = df['H_SV'].fillna(H_SV_median)
            df['A_SV'] = df['A_SV'].fillna(A_SV_median)
            #df.info()

             
            #df.loc[df["H_P3"].isna()]

            #tohle musím dopnit ručně, aby to sedělo s počtem golu a special eventem, udělám to asi uniformě
            df = df.dropna(subset=['H_P3', 'A_P3'])
            df = df.dropna(subset=['H_P2', 'A_P2'])



             
            #check whether HSO are only when special is PS
            #df.loc[(df['H_SO'].notna()) & (df['Special'] == 'PS')]
            #its ok, sth is contumacy (3 games circa) 
            df["H_SO"] = df["H_SO"].fillna(0)
            df["A_SO"] = df["A_SO"].fillna(0)
            #df.info()


             
            #df.loc[(df['H_OT'].notna()) & (df['Special'].isna())] # 1 game and its ok, other games (notna and special ot + ps and dw) are also ok -> all h_OT not null = 0
            #its ok, sth is contumacy (3 games circa) 
            df["H_OT"] = df["H_OT"].fillna(0)
            df["A_OT"] = df["A_OT"].fillna(0)
            #df.info()



             
            #and special add B jako basic game and use dummy var
            df["Special"] = df["Special"].fillna("B")
            #df.info()

             
            #drop Open column
            df.fillna(0, inplace=True)
            #df.head()

             
            df["H"] = df["H"].astype('category')
            df["A"] = df["A"].astype('category')
            df["D"] = df["D"].astype('category')
            special = df[["Special"]].drop_duplicates().reset_index(drop=True)
            #display(special)

             
            special_category = pd.api.types.CategoricalDtype(categories=special.Special.values, ordered=True)
            df["Special"] = df["Special"].astype(special_category)
            

             
            df[df.select_dtypes(['category']).columns] = df.select_dtypes(['category']).apply(lambda x: x.cat.codes)
            



    



            Q1 = df["H_PIM"].quantile(0.25)
            Q3 = df["H_PIM"].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR

            df["H_PIM"] = df["H_PIM"].clip(upper= upper)


            Q1 = df["A_PIM"].quantile(0.25)
            Q3 = df["A_PIM"].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR

            df["A_PIM"] = df["A_PIM"].clip(upper= upper)


            Q1 = df["H_SV"].quantile(0.25)
            Q3 = df["H_SV"].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR

            df["H_SV"] = df["H_SV"].clip(upper= upper)


            df["H_MAJ"] = df["H_MAJ"].clip(upper= 1)


            df["A_MAJ"] = df["A_MAJ"].clip(upper= 1)


             
            return df

        def build_features(
            games_all: pd.DataFrame,
            elo_K: float = 20.0,
            elo_alpha: float = 0.99,
            elo_beta: float = 400.0 / np.log(10),
            elo_home_adv: float = 50.0,
            lambda_fast: float = 0.8,
            lambda_slow: float = 0.95,
            lambda_h2h: float = 0.2
        ) -> pd.DataFrame:
            df = games_all.copy()
            df["Date"] = pd.to_datetime(df["Date"])

            # Sort chronologically to simulate time passing
            df = df.sort_values(["Date", "Season", "HID", "AID"]).reset_index(drop=False)
            df.rename(columns={"index": "orig_index"}, inplace=True)

            team_ids = pd.unique(pd.concat([df["HID"], df["AID"]]))

            # --- State Initialization ---
            elo = {tid: 0.0 for tid in team_ids}
            last_date = {tid: None for tid in team_ids}
            home_streak = {tid: 0 for tid in team_ids}
            away_streak = {tid: 0 for tid in team_ids}
            
            points = {tid: 0.0 for tid in team_ids}
            games_season = {tid: 0 for tid in team_ids}
            goal_diff_season = {tid: 0.0 for tid in team_ids}
            wins_reg = {tid: 0 for tid in team_ids}

            def init_team_stats():
                return {
                    "gf_fast": 0.0, "ga_fast": 0.0, "gf_slow": 0.0, "ga_slow": 0.0,
                    "gf_p1_fast": 0.0, "ga_p1_fast": 0.0, "gf_p2_fast": 0.0, "ga_p2_fast": 0.0,
                    "gf_p3_fast": 0.0, "ga_p3_fast": 0.0,
                    "gf_p1_slow": 0.0, "ga_p1_slow": 0.0, "gf_p2_slow": 0.0, "ga_p2_slow": 0.0,
                    "gf_p3_slow": 0.0, "ga_p3_slow": 0.0,
                    "ot_rate_fast": 0.0, "ot_win_rate_fast": 0.0, "games_with_periods": 0,
                }
            
            team_stats = {tid: init_team_stats() for tid in team_ids}
            
            def init_adv_stats():
                return {
                    "sog_for_fast": 0.0, "sog_against_fast": 0.0, "sog_for_slow": 0.0, "sog_against_slow": 0.0,
                    "sog_share_fast": 0.5, "sog_share_slow": 0.5,
                    "pim_for_fast": 0.0, "pim_against_fast": 0.0,
                    "ppg_fast": 0.0, "shg_fast": 0.0, "svpct_fast": 0.9,
                    "hits_diff_fast": 0.0, "blocks_diff_fast": 0.0, "fo_share_fast": 0.5, "adv_games": 0,
                }

            adv_stats = {tid: init_adv_stats() for tid in team_ids}
            h2h = {}
            feature_rows = []
            current_season = None

            def compute_scores(row):
                if row["H"] == 1: return 1.0, 0.0
                elif row["A"] == 1: return 0.0, 1.0
                elif row["D"] == 1: return 0.5, 0.5
                return 0.5, 0.5

            # --- Main Loop ---
            for _, row in df.iterrows():
                season = row["Season"]
                date = row["Date"]
                h = row["HID"]
                a = row["AID"]

                # Season Reset
                if current_season is None or season != current_season:
                    current_season = season
                    points = {tid: 0.0 for tid in team_ids}
                    games_season = {tid: 0 for tid in team_ids}
                    goal_diff_season = {tid: 0.0 for tid in team_ids}
                    wins_reg = {tid: 0 for tid in team_ids}

                # Elo Decay
                elo[h] *= elo_alpha
                elo[a] *= elo_alpha

                # Current ELO
                elo_h = elo[h]
                elo_a = elo[a]
                elo_diff = (elo_h + elo_home_adv) - elo_a
                elo_p_h = 1.0 / (1.0 + math.exp(-elo_diff / elo_beta))

                # Fatigue
                last_h = last_date[h]
                last_a = last_date[a]
                rest_h = (date - last_h).days if last_h is not None else np.nan
                rest_a = (date - last_a).days if last_a is not None else np.nan
                back_to_back_h = 1 if rest_h == 1 else 0 if not math.isnan(rest_h) else 0
                back_to_back_a = 1 if rest_a == 1 else 0 if not math.isnan(rest_a) else 0

                # Standings
                pts_pg_h = points[h] / games_season[h] if games_season[h] > 0 else 0.0
                pts_pg_a = points[a] / games_season[a] if games_season[a] > 0 else 0.0
                gd_pg_h = goal_diff_season[h] / games_season[h] if games_season[h] > 0 else 0.0
                gd_pg_a = goal_diff_season[a] / games_season[a] if games_season[a] > 0 else 0.0

                # Stats Features (Calculated using current state)
                def per_team_period_features(ts):
                    gf_tot = ts["gf_p1_slow"] + ts["gf_p2_slow"] + ts["gf_p3_slow"]
                    if gf_tot <= 0:
                         p1s = p2s = p3s = 1.0/3.0
                    else:
                        p1s = ts["gf_p1_slow"]/gf_tot
                        p2s = ts["gf_p2_slow"]/gf_tot
                        p3s = ts["gf_p3_slow"]/gf_tot
                    
                    return {
                        "gf_fast": ts["gf_fast"], "ga_fast": ts["ga_fast"],
                        "gf_slow": ts["gf_slow"], "ga_slow": ts["ga_slow"],
                        "gf_p1_fast": ts["gf_p1_fast"], "gf_p2_fast": ts["gf_p2_fast"], "gf_p3_fast": ts["gf_p3_fast"],
                        "ga_p1_fast": ts["ga_p1_fast"], "ga_p2_fast": ts["ga_p2_fast"], "ga_p3_fast": ts["ga_p3_fast"],
                        "gf_p1_share": p1s, "gf_p2_share": p2s, "gf_p3_share": p3s,
                        "third_period_goal_diff": ts["gf_p3_slow"] - ts["ga_p3_slow"],
                        "ot_rate_fast": ts["ot_rate_fast"], "ot_win_rate_fast": ts["ot_win_rate_fast"]
                    }

                per_h = per_team_period_features(team_stats[h])
                per_a = per_team_period_features(team_stats[a]) 
                adv_h = adv_stats[h]
                adv_a = adv_stats[a]

                # H2H
                key_ha = (h, a)
                if key_ha in h2h:
                    h2h_wr = h2h[key_ha]["win_ema"]
                    h2h_gd = h2h[key_ha]["goal_diff_ema"]
                    h2h_cnt = h2h[key_ha]["count"]
                else:
                    h2h_wr = 0.5
                    h2h_gd = 0.0
                    h2h_cnt = 0

                oddsH = row.get("OddsH")
                oddsA = row.get("OddsA")

                # --- BUILD ROW (Full Feature Set) ---
                feature_rows.append({
                    "orig_index": row["orig_index"],
                    "is_opp": row["is_opp"], 
                    "Season": season, "Date": date, "HID": h, "AID": a,
                    "market_type": row.get("market_type", "2way"),
                    "y_home_win": row.get("H"), "y_away_win": row.get("A"), "y_draw": row.get("D"),
                    "oddsH": oddsH, "oddsA": oddsA,
                    
                    "elo_h": elo_h, "elo_a": elo_a, "elo_diff": elo_diff, "elo_p_h": elo_p_h,
                    
                    "pts_pg_h": pts_pg_h, "pts_pg_a": pts_pg_a,
                    "gd_pg_h": gd_pg_h, "gd_pg_a": gd_pg_a,
                    "games_season_h": games_season[h], "games_season_a": games_season[a],
                    
                    "rest_days_h": rest_h, "rest_days_a": rest_a,
                    "rest_diff": (rest_h - rest_a) if pd.notna(rest_h) and pd.notna(rest_a) else np.nan,
                    "back_to_back_h": back_to_back_h, "back_to_back_a": back_to_back_a,
                    "home_streak_h": home_streak[h], "away_streak_a": away_streak[a],
                    
                    # Unpack raw per-team columns (ALL GF/GA/Period stats)
                    **{f"h_{k}": v for k, v in per_h.items()},
                    **{f"a_{k}": v for k, v in per_a.items()},
                    
                    # Calculated Diffs
                    "gf_fast_diff": per_h["gf_fast"] - per_a["gf_fast"],
                    "ga_fast_diff": per_h["ga_fast"] - per_a["ga_fast"],
                    "gf_slow_diff": per_h["gf_slow"] - per_a["gf_slow"],
                    "ga_slow_diff": per_h["ga_slow"] - per_a["ga_slow"],
                    "gf_p1_share_diff": per_h["gf_p1_share"] - per_a["gf_p1_share"],
                    "gf_p2_share_diff": per_h["gf_p2_share"] - per_a["gf_p2_share"],
                    "gf_p3_share_diff": per_h["gf_p3_share"] - per_a["gf_p3_share"],
                    "third_period_goal_diff_diff": per_h["third_period_goal_diff"] - per_a["third_period_goal_diff"],
                    
                    # Advanced Stats Raw
                    "sog_for_fast_h": adv_h["sog_for_fast"], "sog_against_fast_h": adv_h["sog_against_fast"],
                    "sog_share_fast_h": adv_h["sog_share_fast"],
                    "pim_for_fast_h": adv_h["pim_for_fast"], "pim_against_fast_h": adv_h["pim_against_fast"],
                    "ppg_fast_h": adv_h["ppg_fast"], "shg_fast_h": adv_h["shg_fast"],
                    "svpct_fast_h": adv_h["svpct_fast"],
                    "hits_diff_fast_h": adv_h["hits_diff_fast"], "blocks_diff_fast_h": adv_h["blocks_diff_fast"],
                    "fo_share_fast_h": adv_h["fo_share_fast"],
                    
                    "sog_for_fast_a": adv_a["sog_for_fast"], "sog_against_fast_a": adv_a["sog_against_fast"],
                    "sog_share_fast_a": adv_a["sog_share_fast"],
                    "pim_for_fast_a": adv_a["pim_for_fast"], "pim_against_fast_a": adv_a["pim_against_fast"],
                    "ppg_fast_a": adv_a["ppg_fast"], "shg_fast_a": adv_a["shg_fast"],
                    "svpct_fast_a": adv_a["svpct_fast"],
                    "hits_diff_fast_a": adv_a["hits_diff_fast"], "blocks_diff_fast_a": adv_a["blocks_diff_fast"],
                    "fo_share_fast_a": adv_a["fo_share_fast"],
                    
                    # Advanced Stats Diffs
                    "sog_share_fast_diff": adv_h["sog_share_fast"] - adv_a["sog_share_fast"],
                    "pim_for_fast_diff": adv_h["pim_for_fast"] - adv_a["pim_for_fast"],
                    "svpct_fast_sum": adv_h["svpct_fast"] + adv_a["svpct_fast"],
                    "fo_share_fast_diff": adv_h["fo_share_fast"] - adv_a["fo_share_fast"],
                    
                    # H2H
                    "h2h_win_rate_ha": h2h_wr,
                    "h2h_goal_diff_pg_ha": h2h_gd,
                    "h2h_games_ha": h2h_cnt,
                })

                # --- 5. UPDATE STATE (Only for History games) ---
                if row["is_opp"] == False:
                    # Sanity check for results
                    if pd.isna(row.get("HS")) or pd.isna(row.get("AS")):
                        continue

                    S_h, S_a = compute_scores(row)
                    goal_diff = row["HS"] - row["AS"]
                    margin_factor = 1.0 + 0.5 * math.log(1 + abs(goal_diff)) if goal_diff != 0 else 1.0
                    elo[h] += elo_K * (S_h - elo_p_h) * margin_factor
                    elo[a] += elo_K * (S_a - (1.0 - elo_p_h)) * margin_factor

                    last_date[h] = date; last_date[a] = date
                    home_streak[h] = home_streak.get(h, 0) + 1; away_streak[h] = 0
                    away_streak[a] = away_streak.get(a, 0) + 1; home_streak[a] = 0

                    ph, pa = compute_scores(row)
                    points[h] += ph; points[a] += pa
                    games_season[h] += 1; games_season[a] += 1
                    goal_diff_season[h] += goal_diff; goal_diff_season[a] -= goal_diff
                    if S_h == 1.0 and (row.get("H_OT", 0) == 0 and row.get("H_SO", 0) == 0):
                        wins_reg[h] += 1

                    hs, as_ = row["HS"], row["AS"]
                    ts_h = team_stats[h]
                    ts_a = team_stats[a]

                    def update_ema(old, new, lam): return lam * old + (1-lam) * new

                    ts_h["gf_fast"] = update_ema(ts_h["gf_fast"], hs, lambda_fast)
                    ts_h["ga_fast"] = update_ema(ts_h["ga_fast"], as_, lambda_fast)
                    ts_a["gf_fast"] = update_ema(ts_a["gf_fast"], as_, lambda_fast)
                    ts_a["ga_fast"] = update_ema(ts_a["ga_fast"], hs, lambda_fast)

                    ts_h["gf_slow"] = update_ema(ts_h["gf_slow"], hs, lambda_slow)
                    ts_h["ga_slow"] = update_ema(ts_h["ga_slow"], as_, lambda_slow)
                    ts_a["gf_slow"] = update_ema(ts_a["gf_slow"], as_, lambda_slow)
                    ts_a["ga_slow"] = update_ema(ts_a["ga_slow"], hs, lambda_slow)

                    # Period stats update
                    for side, tid, ts in [("H", h, ts_h), ("A", a, ts_a)]:
                        gf1 = row.get(f"{side}_P1", 0); gf2 = row.get(f"{side}_P2", 0); gf3 = row.get(f"{side}_P3", 0)
                        o_side = "A" if side == "H" else "H"
                        ga1 = row.get(f"{o_side}_P1", 0); ga2 = row.get(f"{o_side}_P2", 0); ga3 = row.get(f"{o_side}_P3", 0)

                        ts["gf_p1_fast"] = update_ema(ts["gf_p1_fast"], gf1, lambda_fast)
                        ts["gf_p2_fast"] = update_ema(ts["gf_p2_fast"], gf2, lambda_fast)
                        ts["gf_p3_fast"] = update_ema(ts["gf_p3_fast"], gf3, lambda_fast)
                        ts["ga_p1_fast"] = update_ema(ts["ga_p1_fast"], ga1, lambda_fast)
                        ts["ga_p2_fast"] = update_ema(ts["ga_p2_fast"], ga2, lambda_fast)
                        ts["ga_p3_fast"] = update_ema(ts["ga_p3_fast"], ga3, lambda_fast)

                        ts["gf_p1_slow"] = update_ema(ts["gf_p1_slow"], gf1, lambda_slow)
                        ts["gf_p2_slow"] = update_ema(ts["gf_p2_slow"], gf2, lambda_slow)
                        ts["gf_p3_slow"] = update_ema(ts["gf_p3_slow"], gf3, lambda_slow)
                        ts["ga_p1_slow"] = update_ema(ts["ga_p1_slow"], ga1, lambda_slow)
                        ts["ga_p2_slow"] = update_ema(ts["ga_p2_slow"], ga2, lambda_slow)
                        ts["ga_p3_slow"] = update_ema(ts["ga_p3_slow"], ga3, lambda_slow)

                    # OT/SO update
                    for side, tid, ts in [("H", h, ts_h), ("A", a, ts_a)]:
                        is_ot = (row.get("H_OT",0) + row.get("A_OT",0) + row.get("H_SO",0) + row.get("A_SO",0)) > 0
                        ot_val = 1.0 if is_ot else 0.0
                        win_ots = 0.0
                        if is_ot:
                            if side == "H": win_ots = 1.0 if (row.get("H_OT",0) + row.get("H_SO",0)) > 0 else 0.0
                            else: win_ots = 1.0 if (row.get("A_OT",0) + row.get("A_SO",0)) > 0 else 0.0
                        
                        ts["ot_rate_fast"] = update_ema(ts["ot_rate_fast"], ot_val, lambda_fast)
                        ts["ot_win_rate_fast"] = update_ema(ts["ot_win_rate_fast"], win_ots, lambda_fast)

                    # Advanced Stats Update
                    if row.get("has_adv_stats", False) or row.get("H_SOG", 0) > 0:
                        h_sog, a_sog = row.get("H_SOG",0), row.get("A_SOG",0)
                        tot_sog = h_sog + a_sog if (h_sog+a_sog)>0 else 1
                        
                        # SOG
                        for side, tid, adv, sog_for, sog_against, sog_share in [
                            ("H", h, adv_h, h_sog, a_sog, h_sog/tot_sog),
                            ("A", a, adv_a, a_sog, h_sog, a_sog/tot_sog),
                        ]:
                            adv["sog_for_fast"] = update_ema(adv["sog_for_fast"], sog_for, lambda_fast)
                            adv["sog_against_fast"] = update_ema(adv["sog_against_fast"], sog_against, lambda_fast)
                            adv["sog_for_slow"] = update_ema(adv["sog_for_slow"], sog_for, lambda_slow)
                            adv["sog_against_slow"] = update_ema(adv["sog_against_slow"], sog_against, lambda_slow)
                            adv["sog_share_fast"] = update_ema(adv["sog_share_fast"], sog_share, lambda_fast)

                        # PIM, PPG, SHG
                        H_PIM = row.get("H_PIM", 0); A_PIM = row.get("A_PIM", 0)
                        adv_h["pim_for_fast"] = update_ema(adv_h["pim_for_fast"], H_PIM, lambda_fast)
                        adv_h["pim_against_fast"] = update_ema(adv_h["pim_against_fast"], A_PIM, lambda_fast)
                        adv_a["pim_for_fast"] = update_ema(adv_a["pim_for_fast"], A_PIM, lambda_fast)
                        adv_a["pim_against_fast"] = update_ema(adv_a["pim_against_fast"], H_PIM, lambda_fast)

                        H_PPG = row.get("H_PPG", 0); A_PPG = row.get("A_PPG", 0)
                        H_SHG = row.get("H_SHG", 0); A_SHG = row.get("A_SHG", 0)
                        adv_h["ppg_fast"] = update_ema(adv_h["ppg_fast"], H_PPG, lambda_fast)
                        adv_a["ppg_fast"] = update_ema(adv_a["ppg_fast"], A_PPG, lambda_fast)
                        adv_h["shg_fast"] = update_ema(adv_h["shg_fast"], H_SHG, lambda_fast)
                        adv_a["shg_fast"] = update_ema(adv_a["shg_fast"], A_SHG, lambda_fast)

                        # SV%
                        H_SV = row.get("H_SV", 0); A_SV = row.get("A_SV", 0)
                        sa_h = a_sog + as_; sa_a = h_sog + hs
                        sv_h = H_SV/sa_h if sa_h > 0 else adv_h["svpct_fast"]
                        sv_a = A_SV/sa_a if sa_a > 0 else adv_a["svpct_fast"]
                        adv_h["svpct_fast"] = update_ema(adv_h["svpct_fast"], sv_h, lambda_fast)
                        adv_a["svpct_fast"] = update_ema(adv_a["svpct_fast"], sv_a, lambda_fast)

                        # Hits/Blk
                        H_HIT = row.get("H_HIT", 0); A_HIT = row.get("A_HIT", 0)
                        H_BLK = row.get("H_BLK", 0); A_BLK = row.get("A_BLK", 0)
                        adv_h["hits_diff_fast"] = update_ema(adv_h["hits_diff_fast"], H_HIT-A_HIT, lambda_fast)
                        adv_a["hits_diff_fast"] = update_ema(adv_a["hits_diff_fast"], A_HIT-H_HIT, lambda_fast)
                        adv_h["blocks_diff_fast"] = update_ema(adv_h["blocks_diff_fast"], H_BLK-A_BLK, lambda_fast)
                        adv_a["blocks_diff_fast"] = update_ema(adv_a["blocks_diff_fast"], A_BLK-H_BLK, lambda_fast)

                        # FO
                        H_FO = row.get("H_FO", 0); A_FO = row.get("A_FO", 0)
                        tot_fo = H_FO + A_FO if (H_FO+A_FO)>0 else 1
                        adv_h["fo_share_fast"] = update_ema(adv_h["fo_share_fast"], H_FO/tot_fo, lambda_fast)
                        adv_a["fo_share_fast"] = update_ema(adv_a["fo_share_fast"], A_FO/tot_fo, lambda_fast)

                        adv_h["adv_games"] += 1; adv_a["adv_games"] += 1

                    # H2H Update
                    if key_ha not in h2h: h2h[key_ha] = {"win_ema": 0.5, "goal_diff_ema": 0.0, "count": 0}
                    h2h[key_ha]["win_ema"] = update_ema(h2h[key_ha]["win_ema"], S_h, lambda_h2h)
                    h2h[key_ha]["goal_diff_ema"] = update_ema(h2h[key_ha]["goal_diff_ema"], goal_diff, lambda_h2h)
                    h2h[key_ha]["count"] += 1

            feat_df = pd.DataFrame(feature_rows)
            feat_df = feat_df.set_index("orig_index").sort_index()
            return feat_df 

        def clear_oops ( oops: pd.DataFrame) -> pd.DataFrame:
            df = oops.copy()
            df.drop(columns=["BetH", "BetA", "BetD"], inplace=True)
            df = df.replace([np.inf, -np.inf], np.nan)
            df.fillna(0, inplace=True)
            return df
        #nn return dates and probs for them, because I needed to get rid of draw opp
        def nn_train_and_predict( builded_ft_with_oops: pd.DataFrame) -> dict: 
            df = builded_ft_with_oops.copy()
            # --- DEBUG START ---
            print("\n--- DEBUGGING DATA SHAPE ---")
            print(f"1. Total Rows: {len(df)}")
            print(f"2. Rows with is_opp=False (History): {len(df[df['is_opp'] == False])}")
            print(f"3. Rows with is_opp=True (Future): {len(df[df['is_opp'] == True])}")
            
            if "y_draw" in df.columns:
                print(f"4. Unique values in 'y_draw': {df['y_draw'].unique()}")
                print(f"5. Rows after y_draw==0 filter: {len(df[(df['is_opp'] == False) & (df['y_draw'] == 0)])}")
            else:
                print("4. ERROR: 'y_draw' column is missing!")
            print("----------------------------\n")
            # --- DEBUG END ---
                                         
            # Copy to avoid modifying original
            
            
            # --- FIX: Split History vs Opportunities FIRST ---
            # 1. Extract Opportunities (Future Games)
            # We don't filter these by 'y_draw' because they don't have results yet!
            df_oops = df[df["is_opp"] == True].copy()
            
            # 2. Extract History (Past Games)
            df_hist = df[df["is_opp"] == False].copy()

            # --- Now Clean the History Data ---
            # 3. Filter Draws only from History
            if "y_draw" in df_hist.columns:
                df_hist = df_hist[df_hist["y_draw"] == 0]
            #PROJET PORADNE JESTLI != DF NEDELA BORDEL
            # 4. Calculate Bookmaker Probabilities for both
            # (Handle NaNs carefully)
            for d in [df_oops, df_hist]:
                d.replace(np.nan, 0, inplace=True)
                # Avoid division by zero
                mask = (d["oddsH"] != 0) & (d["oddsA"] != 0)
                d.loc[mask, "Bookmaker_prob"] = (1/d.loc[mask, "oddsH"]) / (1/d.loc[mask, "oddsH"] + 1/d.loc[mask, "oddsA"])
                d["Bookmaker_prob"] = d["Bookmaker_prob"].fillna(0)

            # 5. Clean up Prediction Data (df_oops)
            # Filter rows with valid odds if needed (e.g. oddsD != 0 check)
            if "OddsD" in df_oops.columns:
                 df_oops = df_oops[df_oops["OddsD"] == 0]

            # Safe Date Extraction
            if not df_oops.empty:
                date_key = tuple(df_oops["Date"].tolist())
            else:
                date_key = ("No_Games",)

            # Drop non-feature columns
            drop_cols_predict = ["Season", "Date", "y_draw", "elo_p_h", "market_type", "y_away_win", "is_opp", "is_oop"]
            df_oops.drop(columns=drop_cols_predict, inplace=True, errors='ignore')
            
            # Separate Features vs Meta
            X_predict = df_oops.drop(columns=["Bookmaker_prob", "y_home_win"], errors='ignore').values
            bookmaker_prob_predict = df_oops["Bookmaker_prob"].values.tolist()

            # 6. Clean up Training Data (df_hist)
            drop_cols_train = ["Season", "Date", "y_draw", "elo_p_h", "market_type", "y_away_win", "is_opp", "is_oop"]
            df_hist.drop(columns=drop_cols_train, inplace=True, errors='ignore')

            Xtrain_data = df_hist.drop(columns=["y_home_win", "Bookmaker_prob"], errors='ignore').values
            ytrain_data = df_hist["y_home_win"].values
            bookmaker_prob_train = df_hist["Bookmaker_prob"].values

            # --- CRITICAL SAFETY CHECK ---
            if len(Xtrain_data) == 0:
                raise ValueError("Error: Training Set is empty. Check if 'y_draw' filtering removed all rows.")
            if len(X_predict) == 0:
                # If we have no games to predict, return empty result gracefully
                print("Warning: No opportunities found to predict.")
                return {date_key: np.array([])}

            # 7. Scaling
            scaler = StandardScaler()
            Xtrain_scaled = scaler.fit_transform(Xtrain_data)
            X_predict_scaled = scaler.transform(X_predict)

            # 8. Tensor Conversion
            X_train = torch.tensor(Xtrain_scaled, dtype=torch.float32)
            bookmaker_prob_train = torch.tensor(bookmaker_prob_train, dtype=torch.float32).unsqueeze(1)
            y_train = torch.tensor(ytrain_data, dtype=torch.float32).unsqueeze(1)

            X_predict = torch.tensor(X_predict_scaled, dtype=torch.float32)
            bookmaker_prob_predict = torch.tensor(bookmaker_prob_predict, dtype=torch.float32).unsqueeze(1)

            # 9. DataLoaders
            train_dataset = TensorDataset(X_train, bookmaker_prob_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 

            pred_dataset = TensorDataset(X_predict, bookmaker_prob_predict)
            pred_loader = DataLoader(pred_dataset, batch_size=64, shuffle=False)

            # --- Define Model (Inner Class) ---
            class ProbabilityEstimatorNN(nn.Module):
                def __init__(self, input_dim, hidden_dim=64):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                    self.output = nn.Linear(hidden_dim, 1)
                    nn.init.xavier_uniform_(self.fc1.weight)
                    nn.init.xavier_uniform_(self.fc2.weight)
                    nn.init.zeros_(self.fc1.bias)
                    nn.init.zeros_(self.fc2.bias)
                    nn.init.uniform_(self.output.weight, -0.1, 0.1)
                    nn.init.zeros_(self.output.bias)

                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = self.output(x)
                    return x

            # Training Helpers
            def decorrelation_loss(outputs, bookmaker_prob, lambda_decorr):
                outputs = outputs.view(-1)
                bookmaker_prob = bookmaker_prob.view(-1)
                cov = torch.mean((outputs - outputs.mean()) * (bookmaker_prob - bookmaker_prob.mean()))
                return lambda_decorr * cov**2

            def l2_regularization(model, lambda_):
                l2_norm = sum(torch.sum(p ** 2) for p in model.parameters() if p.requires_grad and p.dim() > 1)
                return lambda_ * l2_norm

            def train(model, optimizer, criterion, lambda_decorr, epochs, loader, lambda_reg):
                model.train()
                for _ in range(epochs):
                    for X_b, bm_b, y_b in loader:
                        optimizer.zero_grad()
                        out = model(X_b)
                        loss = criterion(out, y_b) + decorrelation_loss(bm_b, out, lambda_decorr) + l2_regularization(model, lambda_reg)
                        loss.backward()
                        optimizer.step()

            def predict(model, loader):
                model.eval()
                preds = []
                with torch.no_grad():
                    for X_b, _ in loader:
                        out = model(X_b)
                        preds.append(torch.sigmoid(out).view(-1).cpu())
                return torch.cat(preds).numpy() if preds else np.array([])

            # --- Execution ---
            feature_dim = Xtrain_data.shape[1]
            model = ProbabilityEstimatorNN(feature_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()
            
            train(model, optimizer, criterion, self.decorrelation_lambda, self.epochs, train_loader, self.lambda_)
            predictions = predict(model, pred_loader)
            
            return {date_key: predictions}
        
        def generate_bets( outcome:pd.DataFrame, sens:float, bookmaker_odds_H:np.array, bookmaker_odds_A:np.array, wealth:float) -> dict:

            """
            Maximize Sharpe ratio given model probabilities and bookmaker odds.
            """
            
            # -----------------------------
            # 1. CLASSIFICATION
            # -----------------------------
            def classify_probabilities(outcome, sens):
                result = {}
                for i, p in enumerate(outcome):
                    if p > 0.5 + sens:
                        result[i] = ("H", (p,bookmaker_odds_H[i]))
                    elif p < 0.5 - sens:
                        result[i] = ("A", (1 - p, bookmaker_odds_A[i]))
                    else:
                        result[i] = ("D", 0.0)
                return result

            classified = classify_probabilities(outcome, sens)

            # Extract only usable probabilities (H/A bets)

            # -----------------------------
            # 2. EXPECTED RETURN & VARIANCE
            # -----------------------------
            #exp_ret = np.maximum(probs * bookmaker_odds - 1,0) 
            indices = list(classified.keys())
            labels = []
            p_list = []
            odds_list = []

            for i, (label, info) in classified.items():
                labels.append(label)
                if label == "D":
                    p_list.append(0.0)
                    odds_list.append(0.0)
                else:
                    p, o = info   # info = (p, odds)
                    p_list.append(p)
                    odds_list.append(o)
            #print(odds_list)
            probs = np.array(p_list)        # shape (n,)
            odds  = np.array(odds_list)     # shape (n,)
            labels = np.array(labels) 

            exp_ret = probs*odds - 1 # μ_i = p_i * o_i - 1

            var_ret = probs * (1 - probs) * (odds ** 2) + 1e-8 
                # μ_i = p_i * o_i - 1        
                # σ_i² = p(1-p)o²
        
            # -----------------------------
            # 3. INITIAL GUESS
            # -----------------------------
            
            b0 = np.clip(probs, 1e-3, wealth)  # start with 0 bets
            

            # -----------------------------
            # 4. BOUNDS: zero-prob → force no bet
            # -----------------------------
            bounds = []
            for p in probs:
                if p == 0:
                    bounds.append((0, 0))           # lock bet = 0
                else:
                    bounds.append((0, wealth))      # normal range

            # -----------------------------
            # 5. Constraint: sum(bets) ≤ wealth
            # -----------------------------
            constraints = [
                {'type': 'ineq', 'fun': lambda b: wealth - np.sum(b)}
            ]

            # -----------------------------
            # 6. Sharpe ratio (negative for minimizer)
            # -----------------------------
            def sharpe_negative(b, exp_ret, var_ret):
                expected_return = b @ exp_ret
                variance = np.sum((b ** 2) * var_ret)

                #print("expected_return:", expected_return, "shape:", np.shape(expected_return))
                #print("variance:", variance, "shape:", np.shape(variance))

                if np.isscalar(expected_return) == False or np.isscalar(variance) == False:
                    raise ValueError("Expected return and variance must be scalars.")

                if variance <= 0:
                    return np.inf

                return -(expected_return / np.sqrt(variance))
            # -----------------------------
            # 7. Optimization
            # -----------------------------
            result = minimize(
                fun=sharpe_negative,
                x0=b0,
                args=(exp_ret, var_ret),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            optimal_bets = result.x

            # -----------------------------
            # 8. Attach to dictionary
            # -----------------------------
            merged = {}
            for k, i in enumerate(indices):
                label, info = classified[i]
                merged[i] = {
                    "label": label,
                    "prob": info,
                    "bet": float(optimal_bets[k])
                }

            return merged

        # --- MAIN FUNCTION LOGIC ---
        
        wealth = summary.iloc[0]["Bankroll"]
        #print(f"\n1.fd Total Rows: {len(inc)}\n ")
        

        clear = clear_data(inc)
        oops_ext = clear_oops(opps)
        id = oops_ext.index
        ids = id.values   
        Bets = pd.DataFrame({
            'ID': ids,
            'BetH': 0.0,
            'BetA': 0.0,
            'BetD': 0.0
        })

        #is there opp for my model?
        opps_ND = opps[opps["OddsD"]==0]
        print(f"-- opps {len(opps_ND)} ---")
        train = clear[clear["OddsD"]==0]
        print(f"-- clear {len(train)} ---")

        if len(opps_ND) == 0:
            print(f"No opp for my model")
            return Bets
        elif len(train) == 0:
            print(f"No valid training")
            return Bets

        clear["is_opp"] = False
        oops_ext["is_opp"] = True
        

        #creates min and max bet arrays for each day in opps
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]


        games_all = pd.concat([clear, oops_ext], axis=0, sort=False)

        feat_all = build_features(games_all)

        predictions = nn_train_and_predict(feat_all)

        # --- NEW CODE (USE THIS) ---
        # 1. Extract the arrays from the dictionary values
        arrays = list(predictions.values())

        # 2. Concatenate them into a single numpy array (handles single or multiple dates)
        probs = np.concatenate(arrays)
        # 3. Flatten ensure it is 1D: [0.5, 0.6, ...] instead of [[0.5], [0.6]...]
        probs = probs.flatten()   

        date_list = list(predictions.keys())

        oopsNotD = oops_ext
        bookmaker_odds_H = oopsNotD["OddsH"].values
        bookmaker_odds_A = oopsNotD["OddsA"].values

        bets = generate_bets(
            outcome=probs,
            sens=self.sensitivity,
            bookmaker_odds_H=bookmaker_odds_H,
            bookmaker_odds_A=bookmaker_odds_A,
            wealth=wealth
        )

        #creating apropriate bets for all dates in opps


        


        if summary["Date"].iloc[0] == date_list[0]:
            if bets[0]["bet"] > min_bet:
                if bets[0]["label"] == "H":
                    Bets.at[0, "BetH"] = min(max_bet, bets[0]["bet"])
                elif bets[0]["label"] == "A":
                    Bets.at[0, "BetA"] = min(max_bet, bets[0]["bet"])
        
        return Bets


