import numpy as np
import pandas as pd

from preprocess import Preprocessor
from model_deutschar_1 import Model1Online


def settle_bets(bets_df: pd.DataFrame, bankroll: float):
    bets_df = bets_df.copy()

    oddsH = bets_df["OddsH"].astype(float).values
    oddsD = bets_df["OddsD"].astype(float).values
    oddsA = bets_df["OddsA"].astype(float).values

    H_true = bets_df["H"].values
    D_true = bets_df["D"].values
    A_true = bets_df["A"].values

    stakeH = bets_df["StakeH"].astype(float).values
    stakeD = bets_df["StakeD"].astype(float).values
    stakeA = bets_df["StakeA"].astype(float).values

    n_bets = 0
    total_staked = 0.0

    for i in range(len(bets_df)):
        sH, sD, sA = stakeH[i], stakeD[i], stakeA[i]
        total = sH + sD + sA
        if total > 0:
            bankroll -= total
            total_staked += total
            n_bets += 1

            if H_true[i] == 1:
                bankroll += sH * oddsH[i]
            elif D_true[i] == 1:
                bankroll += sD * oddsD[i]
            elif A_true[i] == 1:
                bankroll += sA * oddsA[i]

    return bankroll, n_bets, total_staked


def evaluate_model1_on_full_dataset(
    csv_path: str = "../data/cleaned_games_with_elo.csv",
    initial_bankroll: float = 1000.0,
    initial_hist_n: int = 2500,
    block_size: int = 1,
):
    #Here is the test offline evaluation 

    df_all = pd.read_csv(csv_path)
    df_all = df_all.sort_values(["Season"]).reset_index(drop=True)

    print(f"Total games: {len(df_all)}")

    #Split into training and the rest
    if initial_hist_n >= len(df_all):
        raise ValueError("initial_hist_n >= total number of games")

    df_initial = df_all.iloc[:initial_hist_n].copy()
    df_rest = df_all.iloc[initial_hist_n:].copy()

    print(f"Training games: {len(df_initial)}")
    print(f"Remaining games: {len(df_rest)}")

    #Model init
    model = Model1Online(
        lambda_corr=0.002,       #CORRELATION correction
        lambda_stats=0.05,      #STAT correction
        kelly_fraction=0.1,    #Hom much to bet from kelly
        edge_threshold=0.10,    #When to bet
        max_window_size=3000,   # rolling training window
        retrain_min_inc=500,    #When to retrain
        num_epochs=4,           #NN PARAMS FROM DOWN HERE
        batch_size=256,
        lr=1e-3,
    )

    bankroll = initial_bankroll
    total_bets = 0
    total_staked = 0.0

    print("First training...")
    model.update_with_inc(df_initial)

    start = 0
    while start < len(df_rest):
        end = min(start + block_size, len(df_rest))
        opps = df_rest.iloc[start:end].copy()

        if len(opps) == 0:
            break

        #Decide if bet or not
        bets_df = model.make_bets(opps, bankroll=bankroll)

        # Settle bets using actual outcomes
        bankroll, n_bets_block, staked_block = settle_bets(bets_df, bankroll)
        total_bets += n_bets_block
        total_staked += staked_block

        # Now these games are finished, so I can feed them feed them as inc to the model
        model.update_with_inc(opps)

        probs = model.predict_probs(opps)
        oddsH = opps["OddsH"].values
        edgeH = probs[:, 0] * oddsH - 1.0
        print("edge home:", edgeH)

        print(
            f"Game number {start}: bets={n_bets_block}, "
            f"staked={staked_block:.2f}, bankroll={bankroll:.2f}"
        )

        start = end

    # Final stats
    roi = (bankroll / initial_bankroll - 1.0) * 100.0 if initial_bankroll > 0 else 0.0
    avg_stake = total_staked / total_bets if total_bets > 0 else 0.0

    print("\n RESULTS TABLE")
    print(f"Initial bankroll : {initial_bankroll:.2f}")
    print(f"Final bankroll   : {bankroll:.2f}")
    print(f"ROI              : {roi:.2f}%")
    print(f"Total bets       : {total_bets}")
    print(f"Total staked     : {total_staked:.2f}")
    print(f"Average stake    : {avg_stake:.2f}")

    return {
        "initial_bankroll": initial_bankroll,
        "final_bankroll": bankroll,
        "roi_percent": roi,
        "total_bets": total_bets,
        "total_staked": total_staked,
        "avg_stake": avg_stake,
    }


if __name__ == "__main__":
    results = evaluate_model1_on_full_dataset()
    print("\nResults dict:", results)

