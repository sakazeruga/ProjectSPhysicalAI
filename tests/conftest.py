import sys
from pathlib import Path

# プロジェクト直下のモジュール (interfaces, state_mapper, ...) を import できるようにする
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
