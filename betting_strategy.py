import numpy as np
from scipy.optimize import minimize
#maybe do this for known number of games (bet half wealth?)-> actualize wealth


import numpy as np
from scipy.optimize import minimize


def betting(bookmaker_odds, outcome, sens):
    """
    Maximize Sharpe ratio given model probabilities and bookmaker odds.
    """
    print("ok")
    # -----------------------------
    # 1. CLASSIFICATION
    # -----------------------------
    def classify_probabilities(outcome, sens):
        result = {}
        for i, p in enumerate(outcome):
            if p > 0.5 + sens:
                result[i] = ("H", p)
            elif p < 0.5 - sens:
                result[i] = ("A", 1 - p)
            else:
                result[i] = ("D", 0.0)
        return result

    classified = classify_probabilities(outcome, sens)

    # Extract only usable probabilities (H/A bets)
    probs = np.array([v[1] for v in classified.values()])

    # -----------------------------
    # 2. EXPECTED RETURN & VARIANCE
    # -----------------------------
    exp_ret = np.maximum(probs * bookmaker_odds - 1, 0) 
                          # μ_i = p_i * o_i - 1
    var_ret = probs * (1 - probs) * (bookmaker_odds ** 2) + 1e-8         # σ_i² = p(1-p)o²

    # -----------------------------
    # 3. INITIAL GUESS
    # -----------------------------
    b0 = np.clip(probs, 1e-3, wealth)  # start with 0 bets
    wealth = 1.0

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

        if variance <= 0:
            return np.inf

        return -(expected_return / np.sqrt(variance))
    print("exp_ret:", exp_ret)
    print("var_ret:", var_ret)
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
    for i, (label, prob) in classified.items():
        merged[i] = {
            "label": label,
            "prob": prob,
            "bet": float(optimal_bets[i])
        }

    return merged, -result.fun
