from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mscapital.research.rounds import research_rounds, run_research
from tests.helpers import temporary_config


class ResearchRoundsTests(unittest.TestCase):
    def test_plan_has_twenty_unique_seed_compatible_rounds(self) -> None:
        rounds = research_rounds()

        self.assertEqual(tuple(item.number for item in rounds), tuple(range(1, 21)))
        self.assertEqual(len({item.code for item in rounds}), 20)
        self.assertEqual(rounds[0].arguments["derived_feature_sets"], ("order_category_ratios",))
        self.assertIn("market__w*__mid_bps*", rounds[-1].arguments["exclude_patterns"])

    def test_runner_writes_resumable_ledger(self) -> None:
        with TemporaryDirectory() as directory:
            config = temporary_config(Path(directory))
            metrics = {
                "overall": {"global_score": 0.2},
                "folds": {"fold_1": {"global_score": 0.1}},
                "baseline_gate": {"passed": True},
            }
            with patch(
                "mscapital.research.rounds.train_oof",
                return_value=("lgbm-test", metrics),
            ) as train:
                ledger = run_research(config, rounds=(1,), threads=3)
                self.assertEqual(train.call_count, 1)
                self.assertEqual(train.call_args.kwargs["seeds"], (97,))
                self.assertEqual(train.call_args.args[0].runtime.threads, 3)
                self.assertEqual(ledger["rounds"]["1"]["status"], "completed")

                run_research(config, rounds=(1,), resume=True, threads=3)
                self.assertEqual(train.call_count, 1)


if __name__ == "__main__":
    unittest.main()
