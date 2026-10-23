"""旧名。中身は tools/eval_ab.py（A/B 評価: 全体・音素の中の位置・帯域のテクスチャ）で、引数も同じ。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ab import main  # noqa: E402

if __name__ == "__main__":
    main()
