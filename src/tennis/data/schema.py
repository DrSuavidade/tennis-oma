from typing import TypedDict, Optional
from datetime import date

class MatchRow(TypedDict, total=False):
    match_id: str
    date: date
    tour: str         # 'ATP' | 'WTA'
    tournament: str
    level: str
    round: str
    surface: str
    indoor: int
    best_of: int
    player_a_id: int
    player_b_id: int
    score: str
    retirement: int
    player_a_odds: float  # optional market
    player_b_odds: float
