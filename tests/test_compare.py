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


def test_every_suite_override_survives_model_copy_without_validation():
    """model_copy(update=) skips validation; values must already be final types."""
    from core.config import Settings

    base = Settings()
    for suite in SUITES.values():
        for _, overrides in suite:
            copied = base.model_copy(update=overrides)
            copied.collection_name()  # would raise on an un-coerced string enum
            copied.fingerprint()


def test_deploy_source_zip_never_contains_secrets_or_corpus(tmp_path):
    """The build artifact goes to S3 and CodeBuild; .env and PDFs must never ride along."""
    import io, zipfile
    from scripts.deploy_fargate import zip_source

    (tmp_path / ".env").write_text("SECRET=x")
    (tmp_path / "data" / "raw").mkdir(parents=True); (tmp_path / "data" / "raw" / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "tests").mkdir(); (tmp_path / "tests" / "t.py").write_text("")
    (tmp_path / "api").mkdir(); (tmp_path / "api" / "main.py").write_text("")
    (tmp_path / "docker").mkdir(); (tmp_path / "docker" / "Dockerfile").write_text("FROM x")
    names = zipfile.ZipFile(io.BytesIO(zip_source(tmp_path))).namelist()
    assert names == ["api/main.py", "docker/Dockerfile"]
