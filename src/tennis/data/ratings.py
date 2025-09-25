import pandas as pd
import numpy as np

def update_elo(df: pd.DataFrame, k_base=32.0, surface_offset=40.0, decay_half_life_days=365):
    # Simple, surface-aware Elo with time decay and inactivity handling.
    df = df.sort_values("date").copy()
    elos = {}
    last_date = {}

    def get_player_surface_key(pid, surface):
        return (pid, surface)

    def expected(r_a, r_b):
        return 1.0 / (1.0 + 10 ** (-(r_a - r_b) / 400))

    def decay(r, days):
        if days is None or days <= 0:
            return r
        decay_factor = 0.5 ** (days / decay_half_life_days)
        return 1500 + (r - 1500) * decay_factor

    a_elos, b_elos, exp_a_list = [], [], []
    for _, row in df.iterrows():
        a, b = row.player_a_id, row.player_b_id
        s = row.surface
        key_a = get_player_surface_key(a, s)
        key_b = get_player_surface_key(b, s)

        ra = elos.get(key_a, 1500.0)
        rb = elos.get(key_b, 1500.0)

        # inactivity decay
        for pid, key in [(a, key_a), (b, key_b)]:
            last = last_date.get(key, None)
            days = None if last is None else (row.date - last).days
            elos[key] = decay(elos.get(key, 1500.0), days)
            last_date[key] = row.date

        ra = elos.get(key_a, 1500.0)
        rb = elos.get(key_b, 1500.0)

        exp_a = expected(ra, rb)
        a_elos.append(ra)
        b_elos.append(rb)
        exp_a_list.append(exp_a)

        # outcome
        y = row["y"]  # 1 if player_a wins else 0
        k = k_base
        elos[key_a] = ra + k * (y - exp_a)
        elos[key_b] = rb + k * ((1 - y) - (1 - exp_a))

    df["elo_a_surface"] = a_elos
    df["elo_b_surface"] = b_elos
    df["elo_exp_a"] = exp_a_list
    return df
