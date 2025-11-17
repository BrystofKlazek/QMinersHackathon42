import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


STAT_NAMES: List[str] = [
    "PEN", "MAJ", "PPG", "SHG", "SV", "PIM", "SOG",
    "BLK_S", "HIT", "BLK", "FO",
    "P1", "P2", "P3",
    "OT", "SO",
]


@dataclass
class Preprocessor:
    """
    This is my attempt at a preprocessor for the hockey games.
    it fits means / stds for numeric features on training data.
    I also thought it may be good to create diff features H_X - A_X
    (Maybe some edge will be discovered by number of shots H takes vs 
    A takes if it gets fed into the model explicitly rather than 
    implicitly)
    It standardizes numeric features.
    It keeps team IDs as integer indices.
    And it computes bookmaker probabilities from odds - just normalisation for now
    If typing i annoying, it might be removed - it is here so we all know what
    is happening even without having written the code ourselves.
    """
    
    stat_names: List[str] = STAT_NAMES
    numeric_cols_: List[str] = field(default_factory=list)
    means_: Dict[str, float] = field(default_factory=dict)
    stds_: Dict[str, float] = field(default_factory=dict)
    max_team_id_: int = 0

    def _ensure_diff_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        # This adds diff_X = H_X - A_X columns in-place and returns the df
        for stat in self.stat_names:
            h_col = f"H_{stat}"
            a_col = f"A_{stat}"
            diff_col = f"diff_{stat}"
            if diff_col not in df.columns:
                df[diff_col] = df[h_col] - df[a_col]
        return df

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        #This bit fits numeric scaling and team ID info on a training dataframe.
        #I normalise the data so that it works better with the NN or other ML algo

        df = df.copy()
        df = self._ensure_diff_cols(df)

        num_cols = ["Season", "Special"]
        for stat in self.stat_names:
            num_cols.append(f"H_{stat}")
            num_cols.append(f"A_{stat}")
            num_cols.append(f"diff_{stat}")

        self.numeric_cols_ = num_cols

        for col in self.numeric_cols_:
            col_values = df[col].astype(float)
            mean = float(col_values.mean())
            std = float(col_values.std(ddof=0))  
            if std == 0.0:
                std = 1.0  
            self.means_[col] = mean
            self.stds_[col] = std

        hid_max = int(df["HID"].max())
        aid_max = int(df["AID"].max())
        self.max_team_id_ = max(hid_max, aid_max)

        return self

    def _standardize_numeric(self, df: pd.DataFrame) -> np.ndarray:
        #Here we take the data and turn it directly into np from 
        #PD - heard it might make our time a bit easier...

        X_num_list = []
        for col in self.numeric_cols_:
            x = df[col].astype(float).values
            mean = self.means_[col]
            std = self.stds_[col]
            X_num_list.append((x - mean) / std)
        X_num = np.stack(X_num_list, axis=1).astype(np.float32)
        return X_num

    def _map_team_ids(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # This returns ids to simple "numpy" ids so they work in the model

        home_ids = df["HID"].astype(int).values
        away_ids = df["AID"].astype(int).values

        return home_ids.astype(np.int64), away_ids.astype(np.int64)

    @staticmethod
    def compute_bookmaker_probs(df: pd.DataFrame) -> np.ndarray:
        #This calculates the probabilities by simple normalisation

        oddsH = df["OddsH"].astype(float).values
        oddsD = df["OddsD"].astype(float).values
        oddsA = df["OddsA"].astype(float).values

        rH = np.where(oddsH > 0, 1.0 / oddsH, 0.0)
        rD = np.where(oddsD > 0, 1.0 / oddsD, 0.0)
        rA = np.where(oddsA > 0, 1.0 / oddsA, 0.0)

        m = rH + rD + rA
        # Here I avoid division by zero. If m == 0, then use uniform 1/3...
        m_safe = np.where(m > 0, m, 1.0)
        qH = np.where(m > 0, rH / m_safe, 1.0 / 3.0)
        qD = np.where(m > 0, rD / m_safe, 1.0 / 3.0)
        qA = np.where(m > 0, rA / m_safe, 1.0 / 3.0)

        Q = np.stack([qH, qD, qA], axis=1).astype(np.float32)
        return Q

    @staticmethod
    def extract_labels(df: pd.DataFrame) -> np.ndarray:
        #Here I turn the H/D/A to one hot encoding so it is again easier for the model

        H = df["H"].astype(int).values
        D = df["D"].astype(int).values
        A = df["A"].astype(int).values

        # y = 0 (home), 1 (draw), 2 (away)
        y = np.full_like(H, fill_value=-1, dtype=np.int64)
        y = np.where(H == 1, 0, y)
        y = np.where(D == 1, 1, y)
        y = np.where(A == 1, 2, y)

        if np.any(y == -1):
            raise ValueError("Some rows have invalid outcome encoding" + 
                             "(no or multiple H/D/A = 1).")
        return y

    def transform(
        self,
        df: pd.DataFrame,
        with_labels: bool = False,
    ) -> Dict[str, np.ndarray]:
        #This now applies thefitted preprocessing to any future dataframe
        #Concretly it returns dict with:
        #  - 'X_num', the standardized numeric matrix [N, num_features]
        #       -   this contains the numerics we need for the model
        #  - 'home_ids'
        #  - 'away_ids
        #  - 'q_book',bookmaker probabilities with size [N, 3]
        #  - 'y' returns labels (only if with_labels=True)
        #       - THIS CANT BE USED ON FUTURE GAMES.
        
        if not self.numeric_cols_:
            raise RuntimeError("ppreprocessor not fitted, please call fit() first")

        df = df.copy()
        df = self._ensure_diff_cols(df)

        X_num = self._standardize_numeric(df)
        home_ids, away_ids = self._map_team_ids(df)
        q_book = self.compute_bookmaker_probs(df)

        result = {
            "X_num": X_num,
            "home_ids": home_ids,
            "away_ids": away_ids,
            "q_book": q_book,
        }

        if with_labels:
            result["y"] = self.extract_labels(df)

        return result


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

    train_data = prep.transform(df_train, with_labels=True)
    val_data = prep.transform(df_val, with_labels=True)

    print("Train numeric shape:", train_data["X_num"].shape)
    print("Val numeric shape:", val_data["X_num"].shape)

