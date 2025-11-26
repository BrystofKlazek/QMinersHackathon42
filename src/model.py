import numpy as np
import pandas as pd

class Model:

    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ):
        def clear_data(data: pd.DataFrame) -> pd.DataFrame:
            import pandas as pd
            import numpy as np

            
            """pd.reset_option('display.max_rows')
            pd.set_option('display.max_columns', None)"""

             
            df = pd.read_csv("data/games.csv")
            #df.head()


        

            df[["OddsH", "OddsA"]] = df[["OddsH", "OddsA"]].replace(0, np.nan)


             
            df = df.dropna(subset=['OddsH'])

             
            df = df.drop("Unnamed: 0", axis = 1)

             
            #df.head() #dropping null values for odds because we need them for our model

             
            #df.info()

             
            df = df.drop(columns = ["Date"]) #drop dates of games
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
            df["H_BLK_S"] = df["H_BLK_S"].fillna(median_A_BLK_S)
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
            df = df.drop("Open", axis=1)
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
            elo_K: float = 20.0,                  # Parameter concerning speed of ELO updates
            elo_alpha: float = 0.99,              # Elo decay per game
            elo_beta: float = 400.0 / np.log(10), # scale for logistic link
            elo_home_adv: float = 50.0,           # home advantage in Elo points
            lambda_fast: float = 0.8,             # fast EMA
            lambda_slow: float = 0.95,             # slow EMA
            lambda_h2h: float = 0.2
        ) -> pd.DataFrame:
            df = games_all.copy()
            df["Date"] = pd.to_datetime(df["Date"])

            # Sort in time order -  preserve original index
            df = df.sort_values(["Date", "Season", "HID", "AID"]).reset_index(drop=False)
            df.rename(columns={"index": "orig_index"}, inplace=True)

            team_ids = pd.unique(pd.concat([df["HID"], df["AID"]]))

            # Elo ratings for each team id - build a DICT so it is easily accesible
            elo = {tid: 0.0 for tid in team_ids}

            # Last game date and also streaks - streak I think could beand also streaks - streak I think could be valuablee
            last_date = {tid: None for tid in team_ids}
            home_streak = {tid: 0 for tid in team_ids}
            away_streak = {tid: 0 for tid in team_ids}

            #Standing in current season - also DICTs so that it is easily accesible 
            points = {tid: 0.0 for tid in team_ids}
            games_season = {tid: 0 for tid in team_ids}
            goal_diff_season = {tid: 0.0 for tid in team_ids}
            wins_reg = {tid: 0 for tid in team_ids}

            # Per-team goal and period stats (EMAs)
            # GF - goals for, GA -goals against, p1 - part 1, OT - overtime, 
            def init_team_stats():
                return {
                    "gf_fast": 0.0,
                    "ga_fast": 0.0,
                    "gf_slow": 0.0,
                    "ga_slow": 0.0,
                    "gf_p1_fast": 0.0,
                    "ga_p1_fast": 0.0,
                    "gf_p2_fast": 0.0,
                    "ga_p2_fast": 0.0,
                    "gf_p3_fast": 0.0,
                    "ga_p3_fast": 0.0,
                    "gf_p1_slow": 0.0,
                    "ga_p1_slow": 0.0,
                    "gf_p2_slow": 0.0,
                    "ga_p2_slow": 0.0,
                    "gf_p3_slow": 0.0,
                    "ga_p3_slow": 0.0,
                    "ot_rate_fast": 0.0,
                    "ot_win_rate_fast": 0.0,
                    "games_with_periods": 0,
                }

            team_stats = {tid: init_team_stats() for tid in team_ids}

            # Advanced stats EMAs
            # SOG - shots at goalie, PIM - penalties in minutes, PPG - power play goals, 
            # SHG - short handed goals, SVPCT - save procentage, FO_share - face off share 
            def init_adv_stats():
                return {
                    "sog_for_fast": 0.0,
                    "sog_against_fast": 0.0,
                    "sog_for_slow": 0.0,
                    "sog_against_slow": 0.0,
                    "sog_share_fast": 0.5,
                    "sog_share_slow": 0.5,
                    "pim_for_fast": 0.0,
                    "pim_against_fast": 0.0,
                    "ppg_fast": 0.0,
                    "shg_fast": 0.0,
                    "svpct_fast": 0.9,
                    "hits_diff_fast": 0.0,
                    "blocks_diff_fast": 0.0,
                    "fo_share_fast": 0.5,
                    "adv_games": 0,
                }

            adv_stats = {tid: init_adv_stats() for tid in team_ids}

            # Head-to-head stats keyed by (home_team, away_team) so we know history
            # of team vs team
            h2h = {}

            feature_rows = []
            current_season = None

            def compute_scores(row):
                #Home and away scores for Elo, distributed in [0,1]
                if row["H"] == 1:
                    return 1.0, 0.0
                elif row["A"] == 1:
                    return 0.0, 1.0
                elif row["D"] == 1:
                    return 0.5, 0.5

            # now iterate over rows and seasons, h, a...
            for _, row in df.iterrows():
                season = row["Season"]
                date = row["Date"]
                h = row["HID"]
                a = row["AID"]

                # If season changes - reset season standings?
                if current_season is None or season != current_season:
                    # If we arrive into a larger number in dataframe,  startt a new season
                    current_season = season
                    points = {tid: 0.0 for tid in team_ids}
                    games_season = {tid: 0 for tid in team_ids}
                    goal_diff_season = {tid: 0.0 for tid in team_ids}
                    wins_reg = {tid: 0 for tid in team_ids}

                # Elo decay towards 0 for now (maybe for mean later)? 
                elo[h] *= elo_alpha
                elo[a] *= elo_alpha

                # Pre-match Elo and Elo-probability
                elo_h = elo[h]
                elo_a = elo[a]
                elo_diff = (elo_h + elo_home_adv) - elo_a
                elo_p_h = 1.0 / (1.0 + math.exp(-elo_diff / elo_beta))

                # Number of rest days or back-to-back fights (could provide useful info? IDK)
                last_h = last_date[h]
                last_a = last_date[a]
                rest_h = (date - last_h).days if last_h is not None else np.nan
                rest_a = (date - last_a).days if last_a is not None else np.nan
                back_to_back_h = 1 if rest_h == 1 else 0 if not math.isnan(rest_h) else 0
                back_to_back_a = 1 if rest_a == 1 else 0 if not math.isnan(rest_a) else 0

                # Now set season standings  
                pts_h = points[h]
                pts_a = points[a]
                gms_h = games_season[h]
                gms_a = games_season[a]
                gd_h = goal_diff_season[h]
                gd_a = goal_diff_season[a]

                pts_pg_h = pts_h / gms_h if gms_h > 0 else 0.0
                pts_pg_a = pts_a / gms_a if gms_a > 0 else 0.0
                gd_pg_h = gd_h / gms_h if gms_h > 0 else 0.0
                gd_pg_a = gd_a / gms_a if gms_a > 0 else 0.0

                # Pre-match stats
                def per_team_period_features(ts):
                    gf_tot_slow = ts["gf_p1_slow"] + ts["gf_p2_slow"] + ts["gf_p3_slow"]
                    if gf_tot_slow <= 0:
                        # if we don't know the distribution of goals set them uniformly distributed
                        # Before the first match with each goal or something?
                        p1_share = p2_share = p3_share = 1.0 / 3.0 
                    else:
                        p1_share = ts["gf_p1_slow"] / gf_tot_slow
                        p2_share = ts["gf_p2_slow"] / gf_tot_slow
                        p3_share = ts["gf_p3_slow"] / gf_tot_slow
                    third_period_diff = ts["gf_p3_slow"] - ts["ga_p3_slow"]
                    return {
                        "gf_fast": ts["gf_fast"],
                        "ga_fast": ts["ga_fast"],
                        "gf_slow": ts["gf_slow"],
                        "ga_slow": ts["ga_slow"],
                        "gf_p1_fast": ts["gf_p1_fast"],
                        "gf_p2_fast": ts["gf_p2_fast"],
                        "gf_p3_fast": ts["gf_p3_fast"],
                        "ga_p1_fast": ts["ga_p1_fast"],
                        "ga_p2_fast": ts["ga_p2_fast"],
                        "ga_p3_fast": ts["ga_p3_fast"],
                        "gf_p1_share": p1_share,
                        "gf_p2_share": p2_share,
                        "gf_p3_share": p3_share,
                        #Chat says the most useful is third period goal difference and that
                        #Setting it explicitly for now might not hurt
                        "third_period_goal_diff": third_period_diff,
                        "ot_rate_fast": ts["ot_rate_fast"],
                        "ot_win_rate_fast": ts["ot_win_rate_fast"],
                    }

                ts_h = team_stats[h]
                ts_a = team_stats[a]
                #Stats per team for easy access later
                per_h = per_team_period_features(ts_h)
                per_a = per_team_period_features(ts_a) 

                # Advanced stats pre game
                adv_h = adv_stats[h]
                adv_a = adv_stats[a]

                # Head-to-heads pre game 
                key_ha = (h, a)
                if key_ha in h2h:
                    h2h_entry = h2h[key_ha]
                    h2h_win_rate = h2h_entry["win_ema"]
                    h2h_goal_diff_pg = h2h_entry["goal_diff_ema"]
                    h2h_games = h2h_entry["count"]
                else:
                    h2h_win_rate = 0.5
                    h2h_goal_diff_pg = 0.0
                    h2h_games = 0

                # Implied market probabilities from 2-way odds
                oddsH = row["OddsH"]
                oddsA = row["OddsA"]

                # Build feature row BEFORE updating states
                feature_rows.append(
                    {
                        "orig_index": row["orig_index"],
                        "Season": season,
                        "Date": date,
                        "HID": h,
                        "AID": a,
                        "market_type": row["market_type"],
                        "y_home_win": row["H"],
                        "y_away_win": row["A"],
                        "y_draw": row["D"],
                        # Odds features
                        "oddsH": oddsH,
                        "oddsA": oddsA,
                        # Elo
                        "elo_h": elo_h,
                        "elo_a": elo_a,
                        "elo_diff": elo_diff,
                        "elo_p_h": elo_p_h,
                        # Season standings
                        "pts_pg_h": pts_pg_h,
                        "pts_pg_a": pts_pg_a,
                        "gd_pg_h": gd_pg_h,
                        "gd_pg_a": gd_pg_a,
                        "games_season_h": gms_h,
                        "games_season_a": gms_a,
                        # Schedule or fatigue or smth
                        "rest_days_h": rest_h,
                        "rest_days_a": rest_a,
                        # Who is more rested
                        "rest_diff": (
                            rest_h - rest_a
                            if (not math.isnan(rest_h) and not math.isnan(rest_a))
                            else np.nan
                        ),
                        "back_to_back_h": back_to_back_h,
                        "back_to_back_a": back_to_back_a,
                        "home_streak_h": home_streak[h],
                        "away_streak_a": away_streak[a],
                        # Per-period & goals: home and away (prefixed h_/a_)
                        **{f"h_{k}": v for k, v in per_h.items()},
                        **{f"a_{k}": v for k, v in per_a.items()},
                        # Differences
                        "gf_fast_diff": per_h["gf_fast"] - per_a["gf_fast"],
                        "ga_fast_diff": per_h["ga_fast"] - per_a["ga_fast"],
                        "gf_slow_diff": per_h["gf_slow"] - per_a["gf_slow"],
                        "ga_slow_diff": per_h["ga_slow"] - per_a["ga_slow"],
                        "gf_p1_share_diff": per_h["gf_p1_share"]
                        - per_a["gf_p1_share"],
                        "gf_p2_share_diff": per_h["gf_p2_share"]
                        - per_a["gf_p2_share"],
                        "gf_p3_share_diff": per_h["gf_p3_share"]
                        - per_a["gf_p3_share"],
                        "third_period_goal_diff_diff": per_h[
                            "third_period_goal_diff"
                        ]
                        - per_a["third_period_goal_diff"],
                        # Advanced stats - hope that this is correclty  
                        "sog_for_fast_h": adv_h["sog_for_fast"],
                        "sog_against_fast_h": adv_h["sog_against_fast"],
                        "sog_share_fast_h": adv_h["sog_share_fast"],
                        "pim_for_fast_h": adv_h["pim_for_fast"],
                        "pim_against_fast_h": adv_h["pim_against_fast"],
                        "ppg_fast_h": adv_h["ppg_fast"],
                        "shg_fast_h": adv_h["shg_fast"],
                        "svpct_fast_h": adv_h["svpct_fast"],
                        "hits_diff_fast_h": adv_h["hits_diff_fast"],
                        "blocks_diff_fast_h": adv_h["blocks_diff_fast"],
                        "fo_share_fast_h": adv_h["fo_share_fast"],
                        "sog_for_fast_a": adv_a["sog_for_fast"],
                        "sog_against_fast_a": adv_a["sog_against_fast"],
                        "sog_share_fast_a": adv_a["sog_share_fast"],
                        "pim_for_fast_a": adv_a["pim_for_fast"],
                        "pim_against_fast_a": adv_a["pim_against_fast"],
                        "ppg_fast_a": adv_a["ppg_fast"],
                        "shg_fast_a": adv_a["shg_fast"],
                        "svpct_fast_a": adv_a["svpct_fast"],
                        "hits_diff_fast_a": adv_a["hits_diff_fast"],
                        "blocks_diff_fast_a": adv_a["blocks_diff_fast"],
                        "fo_share_fast_a": adv_a["fo_share_fast"],
                        "sog_share_fast_diff": adv_h["sog_share_fast"]
                        - adv_a["sog_share_fast"],
                        "pim_for_fast_diff": adv_h["pim_for_fast"]
                        - adv_a["pim_for_fast"],
                        "svpct_fast_sum": adv_h["svpct_fast"]
                        + adv_a["svpct_fast"],
                        "fo_share_fast_diff": adv_h["fo_share_fast"]
                        - adv_a["fo_share_fast"],
                        # Head-to-head stats
                        "h2h_win_rate_ha": h2h_win_rate,
                        "h2h_goal_diff_pg_ha": h2h_goal_diff_pg,
                        "h2h_games_ha": h2h_games,
                    }
                )

                # --------- STAT UPDATE AFTER GAME ---------#

                # Elo updat updatee
                S_h, S_a = compute_scores(row)
                goal_diff = (row["HS"] or 0) - (row["AS"] or 0)
                margin_factor = 1.0
                if goal_diff != 0:
                    margin_factor = 1.0 + 0.5 * math.log(1 + abs(goal_diff))

                elo[h] = elo_h + elo_K * (S_h - elo_p_h) * margin_factor
                elo[a] = elo_a + elo_K * (S_a - (1.0 - elo_p_h)) * margin_factor

                # calculate streaks etc from last date played
                last_date[h] = date
                last_date[a] = date

                home_streak[h] = home_streak.get(h, 0) + 1
                away_streak[h] = 0
                away_streak[a] = away_streak.get(a, 0) + 1
                home_streak[a] = 0

                # new standings (NOT ELO! goals etc.)
                ph, pa = compute_scores(row)
                points[h] += ph
                points[a] += pa
                games_season[h] += 1
                games_season[a] += 1
                goal_diff_season[h] += goal_diff
                goal_diff_season[a] -= goal_diff
                if S_h == 1.0 and (row["H_OT"] == 0 and row["H_SO"] == 0):
                    wins_reg[h] += 1

                # goals - per-period EMAs
                hs = 0 if pd.isna(row["HS"]) else row["HS"]
                ars = 0 if pd.isna(row["AS"]) else row["AS"]

                ts_h["gf_fast"] = lambda_fast * ts_h["gf_fast"] + (1 - lambda_fast) * hs
                ts_h["ga_fast"] = lambda_fast * ts_h["ga_fast"] + (1 - lambda_fast) * ars
                ts_a["gf_fast"] = lambda_fast * ts_a["gf_fast"] + (1 - lambda_fast) * ars
                ts_a["ga_fast"] = lambda_fast * ts_a["ga_fast"] + (1 - lambda_fast) * hs

                ts_h["gf_slow"] = lambda_slow * ts_h["gf_slow"] + (1 - lambda_slow) * hs
                ts_h["ga_slow"] = lambda_slow * ts_h["ga_slow"] + (1 - lambda_slow) * ars
                ts_a["gf_slow"] = lambda_slow * ts_a["gf_slow"] + (1 - lambda_slow) * ars
                ts_a["ga_slow"] = lambda_slow * ts_a["ga_slow"] + (1 - lambda_slow) * hs

                # new prematch stats for both sides
                for side, tid, ts in [("H", h, ts_h), ("A", a, ts_a)]:
                    gf1 = 0 if pd.isna(row[f"{side}_P1"]) else row[f"{side}_P1"]
                    gf2 = 0 if pd.isna(row[f"{side}_P2"]) else row[f"{side}_P2"]
                    gf3 = 0 if pd.isna(row[f"{side}_P3"]) else row[f"{side}_P3"]

                    o_side = "A" if side == "H" else "H"
                    ga1 = 0 if pd.isna(row[f"{o_side}_P1"]) else row[f"{o_side}_P1"]
                    ga2 = 0 if pd.isna(row[f"{o_side}_P2"]) else row[f"{o_side}_P2"]
                    ga3 = 0 if pd.isna(row[f"{o_side}_P3"]) else row[f"{o_side}_P3"]

                    ts["gf_p1_fast"] = lambda_fast * ts["gf_p1_fast"] + (
                        1 - lambda_fast
                    ) * gf1
                    ts["gf_p2_fast"] = lambda_fast * ts["gf_p2_fast"] + (
                        1 - lambda_fast
                    ) * gf2
                    ts["gf_p3_fast"] = lambda_fast * ts["gf_p3_fast"] + (
                        1 - lambda_fast
                    ) * gf3
                    ts["ga_p1_fast"] = lambda_fast * ts["ga_p1_fast"] + (
                        1 - lambda_fast
                    ) * ga1
                    ts["ga_p2_fast"] = lambda_fast * ts["ga_p2_fast"] + (
                        1 - lambda_fast
                    ) * ga2
                    ts["ga_p3_fast"] = lambda_fast * ts["ga_p3_fast"] + (
                        1 - lambda_fast
                    ) * ga3

                    ts["gf_p1_slow"] = lambda_slow * ts["gf_p1_slow"] + (
                        1 - lambda_slow
                    ) * gf1
                    ts["gf_p2_slow"] = lambda_slow * ts["gf_p2_slow"] + (
                        1 - lambda_slow
                    ) * gf2
                    ts["gf_p3_slow"] = lambda_slow * ts["gf_p3_slow"] + (
                        1 - lambda_slow
                    ) * gf3
                    ts["ga_p1_slow"] = lambda_slow * ts["ga_p1_slow"] + (
                        1 - lambda_slow
                    ) * ga1
                    ts["ga_p2_slow"] = lambda_slow * ts["ga_p2_slow"] + (
                        1 - lambda_slow
                    ) * ga2
                    ts["ga_p3_slow"] = lambda_slow * ts["ga_p3_slow"] + (
                        1 - lambda_slow
                    ) * ga3

                # OT and SO behaviour
                for side, tid, ts in [("H", h, ts_h), ("A", a, ts_a)]:
                    ot_game = (
                        row["H_OT"] + row["A_OT"] + row["H_SO"] + row["A_SO"]
                    ) > 0
                    ot_game = 1.0 if ot_game else 0.0
                    if ot_game:
                        if side == "H":
                            win_ots = 1.0 if (row["H_OT"] + row["H_SO"]) > 0 else 0.0
                        else:
                            win_ots = 1.0 if (row["A_OT"] + row["A_SO"]) > 0 else 0.0
                    else:
                        win_ots = 0.0
                    ts["ot_rate_fast"] = lambda_fast * ts["ot_rate_fast"] + (
                        1 - lambda_fast
                    ) * ot_game
                    ts["ot_win_rate_fast"] = lambda_fast * ts["ot_win_rate_fast"] + (
                        1 - lambda_fast
                    ) * win_ots

                # Advanced stats computation
                if row["has_adv_stats"]:
                    # Calculate SOG
                    H_SOG = 0 if pd.isna(row["H_SOG"]) else row["H_SOG"]
                    A_SOG = 0 if pd.isna(row["A_SOG"]) else row["A_SOG"]
                    total_sog = H_SOG + A_SOG if H_SOG + A_SOG > 0 else 1.0
                    sog_share_h = H_SOG / total_sog
                    sog_share_a = A_SOG / total_sog

                    for side, tid, adv, sog_for, sog_against, sog_share in [
                        ("H", h, adv_h, H_SOG, A_SOG, sog_share_h),
                        ("A", a, adv_a, A_SOG, H_SOG, sog_share_a),
                    ]:
                        adv["sog_for_fast"] = lambda_fast * adv["sog_for_fast"] + (
                            1 - lambda_fast
                        ) * sog_for
                        adv["sog_against_fast"] = lambda_fast * adv[
                            "sog_against_fast"
                        ] + (1 - lambda_fast) * sog_against
                        adv["sog_for_slow"] = lambda_slow * adv["sog_for_slow"] + (
                            1 - lambda_slow
                        ) * sog_for
                        adv["sog_against_slow"] = lambda_slow * adv[
                            "sog_against_slow"
                        ] + (1 - lambda_slow) * sog_against
                        adv["sog_share_fast"] = lambda_fast * adv[
                            "sog_share_fast"
                        ] + (1 - lambda_fast) * sog_share

                    # Calculate PIM
                    H_PIM = 0 if pd.isna(row["H_PIM"]) else row["H_PIM"]
                    A_PIM = 0 if pd.isna(row["A_PIM"]) else row["A_PIM"]
                    adv_h["pim_for_fast"] = lambda_fast * adv_h["pim_for_fast"] + (
                        1 - lambda_fast
                    ) * H_PIM
                    adv_h["pim_against_fast"] = lambda_fast * adv_h[
                        "pim_against_fast"
                    ] + (1 - lambda_fast) * A_PIM
                    adv_a["pim_for_fast"] = lambda_fast * adv_a["pim_for_fast"] + (
                        1 - lambda_fast
                    ) * A_PIM
                    adv_a["pim_against_fast"] = lambda_fast * adv_a[
                        "pim_against_fast"
                    ] + (1 - lambda_fast) * H_PIM

                    # Calculate PPG and SHG
                    H_PPG = 0 if pd.isna(row["H_PPG"]) else row["H_PPG"]
                    A_PPG = 0 if pd.isna(row["A_PPG"]) else row["A_PPG"]
                    H_SHG = 0 if pd.isna(row["H_SHG"]) else row["H_SHG"]
                    A_SHG = 0 if pd.isna(row["A_SHG"]) else row["A_SHG"]
                    adv_h["ppg_fast"] = lambda_fast * adv_h["ppg_fast"] + (
                        1 - lambda_fast
                    ) * H_PPG
                    adv_a["ppg_fast"] = lambda_fast * adv_a["ppg_fast"] + (
                        1 - lambda_fast
                    ) * A_PPG
                    adv_h["shg_fast"] = lambda_fast * adv_h["shg_fast"] + (
                        1 - lambda_fast
                    ) * H_SHG
                    adv_a["shg_fast"] = lambda_fast * adv_a["shg_fast"] + (
                        1 - lambda_fast
                    ) * A_SHG

                    # Save percentage calculation (VERY CRUDE! shots against ROUGHLY EQUALS opp SOG + goals conceded)
                    # IDK IF THIS IS CORRECT
                    H_SV = 0 if pd.isna(row["H_SV"]) else row["H_SV"]
                    A_SV = 0 if pd.isna(row["A_SV"]) else row["A_SV"]
                    sa_h = A_SOG + ars
                    sa_a = H_SOG + hs
                    svpct_h_game = H_SV / sa_h if sa_h > 0 else adv_h["svpct_fast"]
                    svpct_a_game = A_SV / sa_a if sa_a > 0 else adv_a["svpct_fast"]
                    adv_h["svpct_fast"] = lambda_fast * adv_h["svpct_fast"] + (
                        1 - lambda_fast
                    ) * svpct_h_game
                    adv_a["svpct_fast"] = lambda_fast * adv_a["svpct_fast"] + (
                        1 - lambda_fast
                    ) * svpct_a_game

                    # Hits and blocks (Physicallity)
                    H_HIT = 0 if pd.isna(row["H_HIT"]) else row["H_HIT"]
                    A_HIT = 0 if pd.isna(row["A_HIT"]) else row["A_HIT"]
                    H_BLK = 0 if pd.isna(row["H_BLK"]) else row["H_BLK"]
                    A_BLK = 0 if pd.isna(row["A_BLK"]) else row["A_BLK"]
                    adv_h["hits_diff_fast"] = lambda_fast * adv_h[
                        "hits_diff_fast"
                    ] + (1 - lambda_fast) * (H_HIT - A_HIT)
                    adv_a["hits_diff_fast"] = lambda_fast * adv_a[
                        "hits_diff_fast"
                    ] + (1 - lambda_fast) * (A_HIT - H_HIT)
                    adv_h["blocks_diff_fast"] = lambda_fast * adv_h[
                        "blocks_diff_fast"
                    ] + (1 - lambda_fast) * (H_BLK - A_BLK)
                    adv_a["blocks_diff_fast"] = lambda_fast * adv_a[
                        "blocks_diff_fast"
                    ] + (1 - lambda_fast) * (A_BLK - H_BLK)

                    # Faceoff stats now
                    H_FO = 0 if pd.isna(row["H_FO"]) else row["H_FO"]
                    A_FO = 0 if pd.isna(row["A_FO"]) else row["A_FO"]
                    total_fo = H_FO + A_FO if H_FO + A_FO > 0 else 1.0
                    fo_share_h = H_FO / total_fo
                    fo_share_a = A_FO / total_fo
                    adv_h["fo_share_fast"] = lambda_fast * adv_h["fo_share_fast"] + (
                        1 - lambda_fast
                    ) * fo_share_h
                    adv_a["fo_share_fast"] = lambda_fast * adv_a["fo_share_fast"] + (
                        1 - lambda_fast
                    ) * fo_share_a

                    adv_h["adv_games"] += 1
                    adv_a["adv_games"] += 1

                # Head-to-head updates
                key_ha = (h, a)
                if key_ha not in h2h:
                    h2h[key_ha] = {"win_ema": 0.5, "goal_diff_ema": 0.0, "count": 0}
                entry = h2h[key_ha]
                entry["win_ema"] = lambda_h2h * entry["win_ema"] + (
                        1 - lambda_h2h) * S_h
                entry["goal_diff_ema"] = lambda_h2h * entry["goal_diff_ema"] + (
                        1 - lambda_h2h) * goal_diff
                entry["count"] += 1

            feat_df = pd.DataFrame(feature_rows)
            feat_df = feat_df.set_index("orig_index").sort_index()
            return feat_df 
        
        min_bet = summary.iloc[0]["Min_bet"]
        N = len(opps)

        bets = np.zeros((N, 3))
        bets[np.arange(N), np.random.choice([0, 1, 2])] = min_bet
        bets = pd.DataFrame(
            data=bets, columns=["BetH", "BetA", "BetD"], index=opps.index
        )
        return bets
