"""Feature engineering for Spaceship Titanic."""

import pandas as pd

SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Group structure: PassengerId is "gggg_pp"
    out["Group"] = out["PassengerId"].str.split("_").str[0]
    group_sizes = out.groupby("Group")["PassengerId"].transform("count")
    out["GroupSize"] = group_sizes
    out["IsAlone"] = (group_sizes == 1).astype(int)

    # Cabin is "deck/num/side"
    cabin = out["Cabin"].str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2]

    # Spending: passengers in cryosleep spend nothing — exploit both directions
    out["TotalSpend"] = out[SPENDING_COLS].sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)
    out.loc[out["CryoSleep"] == True, SPENDING_COLS] = 0  # noqa: E712
    out["CryoSleep"] = out["CryoSleep"].fillna(out["TotalSpend"] == 0)

    out["LuxurySpend"] = out["Spa"] + out["VRDeck"] + out["RoomService"]
    out["BasicSpend"] = out["FoodCourt"] + out["ShoppingMall"]

    return out.drop(columns=["PassengerId", "Cabin", "Name", "Group"])
