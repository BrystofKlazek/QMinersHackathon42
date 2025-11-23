import numpy as np
from scipy.optimize import minimize
#maybe do this for known number of games (bet half wealth?)-> actualize wealth

def betting_strategy(bookmaker_odds = np.array , outcome = np.array):
    """
    Optimize betting strategy to maximize Sharpe ratio given estimated probabilities and bookmaker odds."""

    


    # Estimated expected returns vector (example)
    #here I need (p_io_i - 1)
    exp_ret = outcome*bookmaker_odds - 1 #but I need it to be vector

    #variance 
    #(1-p_i)p_io_i^2
    #make sure that im always betting only on one team in game
    var_ret = (1-outcome)*outcome*bookmaker_odds**2 #again I need vector (for every i)

    #if I add proper values into these I can multiplicate accordingly instead of r and sigma
   
    # Initial guess for bets (uniform distribution or zeros)
    b0 = np.ones_like(exp_ret) / len(exp_ret)

    # Wealth available for betting
    wealth = 1.0

    # Constraints

    def sum_bets_constraint(b):
        return wealth - np.sum(b)

    """def zero_both_complementary_constraint(b, pairs):
        # pairs: list of tuples with indices of complementary outcomes
        return np.array([b[i] * b[j] for i, j in pairs])""" # i think this is not needed, for items shall come in pairs

    # Bounds: bets between 0 and wealth (no short-selling)
    bounds = [(0, wealth) for _ in range(len(exp_ret))]


    """# Example complementary pairs (you must define based on your matches)
    complementary_pairs = [(0, 1), (2, 3), ...]  """

    # Constraint dicts for scipy
    constraints = [
        {'type': 'ineq', 'fun': sum_bets_constraint},   # sum(bets) <= wealth
    ]

    # Add constraints for complementary bets (at most one positive in pair)
    # Here, instead of product constraints (nonconvex), constrain sum <= wealth per pair
    for (i, j) in complementary_pairs:
        constraints.append({
            'type': 'ineq',
            'fun': lambda b, i=i, j=j: wealth - (b[i] + b[j])  # sum of pair bets <= wealth
        })

    # Objective: negative Sharpe ratio (to minimize)
    def sharpe_negative(b, r, Sigma):
        expected_return = b.T @ r
        variance = b.T @ Sigma @ b
        if variance <= 0:
            return np.inf  # prevent invalid square root
        sharpe = expected_return / np.sqrt(variance)
        return -sharpe

    # Optimize using Sequential Quadratic Programming (SLSQP)
    result = minimize(
        fun=sharpe_negative,
        x0=b0,
        args=(r, Sigma),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True}
    )

    optimal_bets = result.x  # optimal bet distribution

    print("Optimal bet distribution:", optimal_bets)
    print("Max Sharpe ratio:", -result.fun)
