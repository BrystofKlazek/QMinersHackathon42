import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple



@dataclass
class Preprocessor:
    """
    This is my attempt at a preprocessor for the hockey games.

    Inputs for predictions are:
      Season - numeric and standardized (tried to do that but didnt help much)
      Team IDs - HID and AID
      Bookmaker probs from OddsH/OddsD/OddsA

    These are inputs only for learning, it is post-game info, so it would leak:
      Special (OT/SO etc.)
      HS/AS (goals)
      all H_* / A_* stat columns from STAT_NAMES (PEN, SOG, P1, P2, ...)
      diff_* versions of those stats (tried to add diferences)
      H/D/A outcome columns via one hot encoding

    """

    # use default_factory to avoid dataclass mutable-default issue
    stat_names: List[str] = field(default_factory=lambda: STAT_NAMES.copy())

    # numeric input columns (for X_num)
    numeric_cols_: List[str] = field(default_factory=list)
    means_: Dict[str, float] = field(default_factory=dict)
    stds_: Dict[str, float] = field(default_factory=dict)

    # stat target columns (for y_stats)
    stat_target_cols_: List[str] = field(default_factory=list)
    stat_means_: Dict[str, float] = field(default_factory=dict)
    stat_stds_: Dict[str, float] = field(default_factory=dict)

    max_team_id_: int = 0
    use_unknown_team_: bool = False
    unk_team_id_: Optional[int] = None

    # ---------- internals for diff_* ----------

    def _ensure_diff_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        # This adds diff_X = H_X - A_X columns in-place and returns the df
        # We only use these as auxiliary targets (not inputs).
        for stat in self.stat_names:
            h_col = f"H_{stat}"
            a_col = f"A_{stat}"
            diff_col = f"diff_{stat}"
            if diff_col not in df.columns:
                df[diff_col] = df[h_col] - df[a_col]
        return df

    # ---------- core API ----------

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """
        Fit numeric scaling for INPUT features and stat targets, and
        infer team ID range, on a training dataframe.

        IMPORTANT: fit is called on finished Games data, so all post-match
        stats/labels are present here.
        """
        df = df.copy()
        df = self._ensure_diff_cols(df)

        # 1) INPUT NUMERIC FEATURES (leak-free)
        #    Season + ELO + simple form features.
        num_cols: List[str] = ["Season"]

        # Optional extra numeric features, only if they exist in df
        extra_cols = [
            "ELO_H", "ELO_A", "ELO_diff",
            "H_gp_before", "H_gf_mean_before", "H_ga_mean_before",
            "A_gp_before", "A_gf_mean_before", "A_ga_mean_before",
            "gp_diff", "gf_mean_diff", "ga_mean_diff",
        ]

        for col in extra_cols:
            if col in df.columns:
                num_cols.append(col)

        self.numeric_cols_ = num_cols

        for col in self.numeric_cols_:
            col_values = df[col].astype(float)
            mean = float(col_values.mean())
            std = float(col_values.std(ddof=0))
            if std == 0.0 or not np.isfinite(std):
                std = 1.0
            self.means_[col] = mean
            self.stds_[col] = std
        
        # 2) STAT TARGETS (auxiliary labels, normalized)
        #    We include:
        #      - HS, AS (goals)
        #      - Special (OT/SO etc. as numeric target)
        #      - H_stat, A_stat, diff_stat for each stat_name
        stat_target_cols: List[str] = []

        # goals:
        for col in ["HS", "AS"]:
            if col not in df.columns:
                raise KeyError(f"Stat target column '{col}' not found in training df.")
            stat_target_cols.append(col)

        # Special as another stat target (NOT input)
        if "Special" in df.columns:
            stat_target_cols.append("Special")
        else:
            raise KeyError("Stat target column 'Special' not found in training df.")

        # stats and their diffs:
        for stat in self.stat_names:
            for prefix in ("H", "A"):
                col = f"{prefix}_{stat}"
                if col not in df.columns:
                    raise KeyError(f"Stat target column '{col}' not found in training df.")
                stat_target_cols.append(col)
            diff_col = f"diff_{stat}"
            if diff_col not in df.columns:
                raise KeyError(f"Stat target column '{diff_col}' not found in training df.")
            stat_target_cols.append(diff_col)

        self.stat_target_cols_ = stat_target_cols

        for col in self.stat_target_cols_:
            vals = df[col].astype(float).values
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            if std == 0.0 or not np.isfinite(std):
                std = 1.0
            self.stat_means_[col] = mean
            self.stat_stds_[col] = std

        # 3) TEAM ID RANGE (for embeddings)
        hid_max = int(df["HID"].astype(int).max())
        aid_max = int(df["AID"].astype(int).max())
        self.max_team_id_ = max(hid_max, aid_max)

        if self.use_unknown_team_:
            self.unk_team_id_ = self.max_team_id_ + 1

        return self

    def transform(
        self,
        df: pd.DataFrame,
        with_labels: bool = False,
        with_stat_targets: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        This now applies the fitted preprocessing to any future dataframe.

        Concretely it returns a dict with:
          - 'X_num': the standardized numeric matrix [N, num_features]
                -> this contains the leak-free numerics we need for the model
          - 'home_ids': [N]
          - 'away_ids': [N]
          - 'q_book': bookmaker probabilities [N, 3]
          - 'y': labels [N] (0/1/2), only if with_labels=True
                -> THIS CANNOT BE USED ON FUTURE GAMES.
          - 'y_stats': normalized auxiliary stat targets [N, S],
                       only if with_stat_targets=True and only for finished games.

        IMPORTANT:
          - For training on Games: with_labels=True, with_stat_targets=True
          - For opps/future games: with_labels=False, with_stat_targets=False
        """
        if not self.numeric_cols_:
            raise RuntimeError("preprocessor not fitted, please call fit() first")

        df = df.copy()
        df = self._ensure_diff_cols(df)

        X_num = self._standardize_numeric(df)
        home_ids, away_ids = self._map_team_ids(df)
        q_book = self.compute_bookmaker_probs(df)

        result: Dict[str, np.ndarray] = {
            "X_num": X_num,
            "home_ids": home_ids,
            "away_ids": away_ids,
            "q_book": q_book,
        }

        if with_labels:
            result["y"] = self.extract_labels(df)

        if with_stat_targets:
            result["y_stats"] = self._extract_normalized_stat_targets(df)

        return result

    # ---------- helpers for numeric inputs ----------

    def _standardize_numeric(self, df: pd.DataFrame) -> np.ndarray:
        # Here we take the numeric input data and turn it directly into np from
        # pd.DataFrame. This is Season only (for now).
        X_num_list = []
        for col in self.numeric_cols_:
            x = df[col].astype(float).values
            mean = self.means_[col]
            std = self.stds_[col]
            X_num_list.append((x - mean) / std)
        X_num = np.stack(X_num_list, axis=1).astype(np.float32)
        return X_num

    def _map_team_ids(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map HID/AID to int64 arrays in a safe range [0, max_team_id_].
        Any team id above max_team_id_ is clipped to max_team_id_ so we
        never index outside the embedding.
        """
        home_ids = df["HID"].astype(int).values
        away_ids = df["AID"].astype(int).values

        # clip to [0, max_team_id_]
        home_ids = np.clip(home_ids, 0, self.max_team_id_)
        away_ids = np.clip(away_ids, 0, self.max_team_id_)

        return home_ids.astype(np.int64), away_ids.astype(np.int64)
    # ---------- helpers for stat targets (auxiliary labels) ----------

    def _extract_normalized_stat_targets(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build a matrix of normalized auxiliary stat labels from post-match columns.

        For each col in stat_target_cols_:
          y_norm = (y - mean) / std

        These are used ONLY during training (finished games), never as inputs,
        and they are NOT required at prediction time.
        """
        if not self.stat_target_cols_:
            raise RuntimeError("Stat target columns not set. Call fit() first.")

        cols = []
        for col in self.stat_target_cols_:
            if col not in df.columns:
                raise KeyError(f"Stat target column '{col}' not found in df.")
            x = df[col].astype(float).values
            mean = self.stat_means_.get(col, 0.0)
            std = self.stat_stds_.get(col, 1.0)
            cols.append((x - mean) / std)
        Y_stats = np.stack(cols, axis=1).astype(np.float32)
        return Y_stats

    # ---------- labels & bookmaker probs ----------

    @staticmethod
    def extract_labels(df: pd.DataFrame) -> np.ndarray:
        # Here I turn the H/D/A to a single label so it is easier for the model

        H = df["H"].astype(int).values
        D = df["D"].astype(int).values
        A = df["A"].astype(int).values

        # y = 0 (home), 1 (draw), 2 (away)
        y = np.full_like(H, fill_value=-1, dtype=np.int64)
        y = np.where(H == 1, 0, y)
        y = np.where(D == 1, 1, y)
        y = np.where(A == 1, 2, y)

        if np.any(y == -1):
            raise ValueError("Some rows have invalid outcome encoding"
                             " (no or multiple H/D/A = 1).")
        return y

    @staticmethod
    def compute_bookmaker_probs(df: pd.DataFrame) -> np.ndarray:
        # This calculates the probabilities by simple normalisation of odds

        oddsH = df["OddsH"]
        oddsD = df["OddsD"]
        oddsA = df["OddsA"]

        # safe reciprocal: avoid divide-by-zero warnings
        rH = np.zeros_like(oddsH, dtype=float)
        rD = np.zeros_like(oddsD, dtype=float)
        rA = np.zeros_like(oddsA, dtype=float)

        maskH = oddsH > 0
        maskD = oddsD > 0
        maskA = oddsA > 0

        rH[maskH] = 1.0 / oddsH[maskH]
        rD[maskD] = 1.0 / oddsD[maskD]
        rA[maskA] = 1.0 / oddsA[maskA]

        m = rH + rD + rA
        # Here I avoid division by zero. If m == 0, then use uniform 1/3...
        m_safe = np.where(m > 0, m, 1.0)
        qH = np.where(m > 0, rH / m_safe, 1.0 / 3.0)
        qD = np.where(m > 0, rD / m_safe, 1.0 / 3.0)
        qA = np.where(m > 0, rA / m_safe, 1.0 / 3.0)

        Q = np.stack([qH, qD, qA], axis=1).astype(np.float32)
        return Q


# Example usage
#--------------------THESE LONG LINES LOOK NICE--------------------
if __name__ == "__main__":
    csv_path = "data/cleaned_games.csv"

    df_all = pd.read_csv(csv_path)

    # Lets say we train on all seasons <= 2010, and keep > 2010 for validation
    train_mask = df_all["Season"] <= 2010
    df_train = df_all[train_mask].reset_index(drop=True)
    df_val = df_all[~train_mask].reset_index(drop=True)

    prep = Preprocessor()
    prep.fit(df_train)

    # Training: we have finished games, so we can use labels + stat targets
    train_data = prep.transform(df_train, with_labels=True, with_stat_targets=True)

    # Validation “as opps”: no labels, no stat targets (mimic real betting)
    val_data = prep.transform(df_val, with_labels=False, with_stat_targets=False)

    print("Train numeric shape:", train_data["X_num"].shape)
    print("Train y_stats shape:", train_data["y_stats"].shape)
    print("Val numeric shape:", val_data["X_num"].shape)

