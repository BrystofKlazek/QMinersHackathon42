import numpy as np
from scipy.optimize import minimize
#maybe do this for known number of games (bet half wealth?)-> actualize wealth
df = pd.read("")

bookmaker_odds = df["OddsH"].values

#model estimated probs for apropriate batch (when evaluating maybe count number of known games and count batch from it)
outcomes = model(Xtrain)

# Estimated expected returns vector (example)
#here I need (p_io_i - 1)
exp_ret = outcomes*bookmaker_odds - 1 #but I need it to be vector

#variance 
#(1-p_i)p_io_i^2
#make sure that im always betting only on one team in game
var_ret = (1-outcome)*outcome*bookmaker_odds**2 #again I need vector (for every i)

#if I add proper values into these I can multiplicate accordingly instead of r and sigma
"""
r = np.array([0.05, 0.1, 0.07, ...])  # shape (n,)

# Covariance matrix of returns (n x n)
Sigma = np.array([
    [...],
    [...],
])
"""
# Initial guess for bets (uniform distribution or zeros)
b0 = np.ones_like(r) / len(r)

# Wealth available for betting
wealth = 1.0

# Constraints

def sum_bets_constraint(b):
    return wealth - np.sum(b)

def zero_both_complementary_constraint(b, pairs):
    # pairs: list of tuples with indices of complementary outcomes
    return np.array([b[i] * b[j] for i, j in pairs])

# Bounds: bets between 0 and wealth (no short-selling)
bounds = [(0, wealth) for _ in r]

# Example complementary pairs (you must define based on your matches)
complementary_pairs = [(0, 1), (2, 3), ...]  

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
