import numpy as np

def compute_bets_kelly(
    probs: np.ndarray,
    oddsH: np.ndarray,
    oddsD: np.ndarray,
    oddsA: np.ndarray,
    bankroll: float,
    edge_threshold: float = 0.50,
    kelly_fraction: float = 0.15,
    max_total_fraction_per_game: float = 0.15,
):
    """
    Compute stakes for each game and outcome using (fractional) Kelly.

    Inputs
    ------
    probs: [N, 3] array of model probabilities (pH, pD, pA) per game.
    oddsH, oddsD, oddsA: [N] arrays of decimal odds for home/draw/away.
    bankroll: current bankroll (scalar).
    edge_threshold: minimum edge p*O - 1 required to place a bet.
    kelly_fraction: fraction of Kelly stake to actually bet (0 < f <= 1).
    max_total_fraction_per_game: cap of total stake per game as a fraction of bankroll.

    Returns
    -------
    stakeH, stakeD, stakeA: [N] arrays of stakes for each outcome.
    """

    N = probs.shape[0]
    pH = probs[:, 0]
    pD = probs[:, 1]
    pA = probs[:, 2]

    stakeH = np.zeros(N, dtype=float)
    stakeD = np.zeros(N, dtype=float)
    stakeA = np.zeros(N, dtype=float)

    def kelly_fraction_outcome(p, O):
        """
        Full Kelly fraction for one outcome with prob p and odds O.

        Kelly formula for a single fair-bet outcome:
            f* = (p*O - 1) / (O - 1)

        We only bet if the edge (p*O - 1) is above `edge_threshold`.
        """
        if O <= 1.0:
            return 0.0
        edge = p * O - 1.0
        if edge <= edge_threshold:
            return 0.0
        return edge / (O - 1.0)

    for i in range(N):
        fH = kelly_fraction_outcome(pH[i], oddsH[i])
        fD = kelly_fraction_outcome(pD[i], oddsD[i])
        fA = kelly_fraction_outcome(pA[i], oddsA[i])

        # fractional Kelly (e.g. 10% of full Kelly)
        fH *= kelly_fraction
        fD *= kelly_fraction
        fA *= kelly_fraction

        # cap total risk per game
        total_f = fH + fD + fA
        if total_f > max_total_fraction_per_game and total_f > 0:
            scale = max_total_fraction_per_game / total_f
            fH *= scale
            fD *= scale
            fA *= scale

        stakeH[i] = fH * bankroll
        stakeD[i] = fD * bankroll
        stakeA[i] = fA * bankroll

    return stakeH, stakeD, stakeA

