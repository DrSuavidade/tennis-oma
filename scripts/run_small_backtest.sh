#!/usr/bin/env bash
set -euo pipefail
python -m tennis.cli build-features --asof 2018-01-01
python -m tennis.cli train --folds 3 --models elo,gbdt --tour both
python -m tennis.cli stack --since 2018-01-01 --use-market no
python -m tennis.cli evaluate --since 2018-01-01
