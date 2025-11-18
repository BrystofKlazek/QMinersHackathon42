import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from preprocess import Preprocessor
import betting_logic as bl


#Neural net for my first model

class Model1Net(nn.Module):
    def __init__(
        self,
        num_numeric_features: int,
        num_teams: int,
        team_emb_dim: int = 32,
        hidden_dim: int = 512,
        num_stat_targets: int = 35,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.team_emb = nn.Embedding(num_teams, team_emb_dim)

        input_dim = num_numeric_features + 2 * team_emb_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.out_main = nn.Linear(hidden_dim, 3)

        self.num_stat_targets = num_stat_targets
        self.out_stats = (
            nn.Linear(hidden_dim, num_stat_targets)
            if num_stat_targets > 0
            else None
        )

    def forward(self, X_num, home_ids, away_ids):
        home_e = self.team_emb(home_ids)  
        away_e = self.team_emb(away_ids)   
        x = torch.cat([X_num, home_e, away_e], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        logits = self.out_main(x)
        stats_pred = self.out_stats(x) if self.out_stats is not None else None
        return logits, stats_pred


def model1_loss(
    logits,
    y,
    q_book,
    stats_pred=None,
    y_stats=None,
    lambda_corr: float = 0.05,
    lambda_stats: float = 0.05,
):
    # main CE loss - this is the main to train
    ce = F.cross_entropy(logits, y)

    # decorrelation term vs bookmaker on the true outcome
    probs = F.softmax(logits, dim=1)
    batch_idx = torch.arange(y.shape[0], device=logits.device)
    p_true = probs[batch_idx, y]
    q_true = q_book[batch_idx, y]

    eps = 1e-8
    p_mean = p_true.mean()
    q_mean = q_true.mean()
    p_centered = p_true - p_mean
    q_centered = q_true - q_mean
    cov = (p_centered * q_centered).mean()
    p_std = p_true.std(unbiased=False)
    q_std = q_true.std(unbiased=False)
    corr = cov / (p_std * q_std + eps)

    loss = ce + lambda_corr * (corr ** 2)

    # auxiliary stats loss - to also take in the post game stats
    if (
        lambda_stats > 0.0
        and stats_pred is not None
        and y_stats is not None
        and y_stats.numel() > 0
    ):
        stats_loss = F.mse_loss(stats_pred, y_stats)
        loss = loss + lambda_stats * stats_loss

    return loss, ce.detach(), corr.detach()


class HockeyDataset(Dataset):
    def __init__(self, data_dict):
        self.X_num = data_dict["X_num"]
        self.home_ids = data_dict["home_ids"]
        self.away_ids = data_dict["away_ids"]
        self.q_book = data_dict["q_book"]
        self.y = data_dict["y"]
        self.y_stats = data_dict.get("y_stats", None)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        X_num = torch.from_numpy(self.X_num[idx])
        home_id = torch.tensor(self.home_ids[idx], dtype=torch.long)
        away_id = torch.tensor(self.away_ids[idx], dtype=torch.long)
        q_book = torch.from_numpy(self.q_book[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)

        if self.y_stats is not None:
            y_stats = torch.from_numpy(self.y_stats[idx])
        else:
            # empty tensor if we somehow have no stats
            y_stats = torch.zeros(0, dtype=torch.float32)

        return X_num, home_id, away_id, q_book, y, y_stats

#HERE I ASKED CHAT TO GENERATE THE MODEL WRAPPER

class Model1Online:
    def __init__(
        self,
        lambda_corr: float = 0.05,
        lambda_stats: float = 0.05,
        kelly_fraction: float = 0.03,
        edge_threshold: float = 0.07,
        max_window_size: int = 2000,
        retrain_min_inc: int = 300,
        num_epochs: int = 4,
        batch_size: int = 256,
        lr: float = 1e-3,
        device=None,
    ):
        """
        lambda_corr:     weight for decorrelation penalty in loss
        lambda_stats:    weight for auxiliary stats MSE loss
        kelly_fraction:  fractional Kelly multiplier for staking
        edge_threshold:  minimum edge p*O - 1 to place a bet
        max_window_size: number of most recent games to train on
        retrain_min_inc: how many new games to accumulate before retraining
        """
        self.lambda_corr = lambda_corr
        self.lambda_stats = lambda_stats
        self.kelly_fraction = kelly_fraction
        self.edge_threshold = edge_threshold
        self.max_window_size = max_window_size
        self.retrain_min_inc = retrain_min_inc
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # State that evolves over time
        self.history_df = None         # all games seen so far
        self.prep = None               # Preprocessor fitted on current window
        self.model = None              # Model1Net trained on current window
        self._num_numeric_features = None
        self._num_teams = None
        self._num_stat_targets = None
        self._untrained = True         # flag to know when to (re)train
        self._since_last_train = 0     # count new games since last train

    # ----- internal training helpers -----

    def _train_on_window(self, window_df: pd.DataFrame):
        # Fit preprocessor
        self.prep = Preprocessor()
        self.prep.fit(window_df)
        train_data = self.prep.transform(
            window_df,
            with_labels=True,
            with_stat_targets=True,
        )

        self._num_numeric_features = train_data["X_num"].shape[1]
        self._num_teams = self.prep.max_team_id_ + 1
        self._num_stat_targets = train_data["y_stats"].shape[1]

        if self.model is None:
            self.model = Model1Net(
            num_numeric_features=self._num_numeric_features,
            num_teams=self._num_teams,
            team_emb_dim=16,
            hidden_dim=128,
            num_stat_targets=self._num_stat_targets,
            dropout_p=0.3,
            ).to(self.device)
        else:
            pass
        
        dataset = HockeyDataset(train_data)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_ce = 0.0
            total_corr = 0.0
            n_batches = 0

            for X_num, home_ids, away_ids, q_book, y, y_stats in loader:
                X_num = X_num.to(self.device)
                home_ids = home_ids.to(self.device)
                away_ids = away_ids.to(self.device)
                q_book = q_book.to(self.device)
                y = y.to(self.device)
                y_stats = y_stats.to(self.device)

                optimizer.zero_grad()
                logits, stats_pred = self.model(X_num, home_ids, away_ids)
                loss, ce, corr = model1_loss(
                    logits,
                    y,
                    q_book,
                    stats_pred=stats_pred,
                    y_stats=y_stats,
                    lambda_corr=self.lambda_corr,
                    lambda_stats=self.lambda_stats,
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_ce += ce.item()
                total_corr += corr.item()
                n_batches += 1

            print(
                f"[Model1] Epoch {epoch+1}/{self.num_epochs} "
                f"loss={total_loss/n_batches:.4f} "
                f"CE={total_ce/n_batches:.4f} "
                f"corr={total_corr/n_batches:.4f}"
            )

        self._untrained = False
        self._since_last_train = 0

    #Here is Api for updates etc.
    #This updates the model when enough new games have came
    def update_with_inc(self, inc_df: pd.DataFrame):
        if inc_df is None or len(inc_df) == 0:
            return

        inc_df = inc_df.copy()
        inc_df = inc_df.sort_values(["Season"]).reset_index(drop=True)

        if self.history_df is None:
            self.history_df = inc_df
        else:
            self.history_df = pd.concat([self.history_df, inc_df], ignore_index=True)

        # keep only last max_window_size games
        if len(self.history_df) > self.max_window_size:
            self.history_df = self.history_df.iloc[-self.max_window_size:].reset_index(drop=True)

        self._since_last_train += len(inc_df)

        # train or retrain when enough new data
        if self._untrained or self._since_last_train >= self.retrain_min_inc:
            print(
                f"Retraining on last {len(self.history_df)} games ")
            self._train_on_window(self.history_df)

    def predict_probs(self, opps_df: pd.DataFrame) -> np.ndarray:
        #This returns finally the probabilities

        opps_proc = self.prep.transform(opps_df, 
                                        with_labels=False, with_stat_targets=False)
        X_num = torch.from_numpy(opps_proc["X_num"]).to(self.device)
        home_ids = torch.tensor(opps_proc["home_ids"], dtype=torch.long, 
                                device=self.device)
        away_ids = torch.tensor(opps_proc["away_ids"], dtype=torch.long, 
                                device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(X_num, home_ids, away_ids)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        return probs

    def make_bets(self, opps_df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
        #Places bets according to betting rules logic
        if len(opps_df) == 0:
            return opps_df

        df = opps_df.copy()

        #First we  get our probabilities
        probs = self.predict_probs(df)  # [N, 3]

        #then we extract odds as numpy arrays
        oddsH = df["OddsH"].values.astype(float)
        oddsD = df["OddsD"].values.astype(float)
        oddsA = df["OddsA"].values.astype(float)

        # and then we compute stakes with our betting logic
        stakeH, stakeD, stakeA = bl.compute_bets_kelly(
            probs=probs,
            oddsH=oddsH,
            oddsD=oddsD,
            oddsA=oddsA,
            bankroll=bankroll,
            edge_threshold=self.edge_threshold,
            kelly_fraction=self.kelly_fraction,
            max_total_fraction_per_game=0.05,  # a bit more conservative
        )

        df["StakeH"] = stakeH
        df["StakeD"] = stakeD
        df["StakeA"] = stakeA

        return df
