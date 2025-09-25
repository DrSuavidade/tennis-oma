from pydantic_settings import BaseSettings
from pydantic import Field
import yaml

class Settings(BaseSettings):
    data_dir: str = Field(default="data")
    processed_dir: str = Field(default="data/processed")
    raw_dir: str = Field(default="data/raw")
    seed: int = 42
    tour: str = "both"
    include_market_features: bool = False
    exclude_retirements: bool = True
    calibrator: str = "auto"

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
