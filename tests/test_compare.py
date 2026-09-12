from evaluation.compare import SUITES, render_table


def test_render_table_formats_floats_and_missing_values():
    rows = [
        {"config": "dense", "hit@1": 0.5455, "chunk_recall": 0.83, "doc_recall": 0.88,
         "mrr": 0.68, "ndcg": 0.70, "neg_gate": 0.3333},
        {"config": "hybrid", "hit@1": 0.64, "chunk_recall": 0.95, "doc_recall": 0.92,
         "mrr": 0.82, "ndcg": 0.79, "neg_gate": None},
    ]
    table = render_table(rows, k=5, extra_cols=[])
    lines = table.splitlines()
    assert lines[0].startswith("| config | hit@1 | chunk R@5 | doc R@5 | MRR | nDCG@5 | neg gate |")
    assert "| dense | 0.55 | 0.83 |" in lines[2]
    assert lines[3].endswith("| - |")


def test_every_suite_config_is_a_valid_settings_override():
    from core.config import Settings

    for suite in SUITES.values():
        for _, overrides in suite:
            Settings(**overrides)
