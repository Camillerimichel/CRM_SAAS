import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["FASTAPI_SKIP_WARM"] = "1"

import src.api.dashboard as dashboard


class _NoopDB:
    def execute(self, *args, **kwargs):
        raise AssertionError("This test should not hit the database")


def test_populate_synthese_esg_context_preserves_exclusions_check(monkeypatch):
    monkeypatch.setattr(dashboard, "_get_esg_fields", lambda db, debug=False: ([], []))
    monkeypatch.setattr(dashboard, "_get_esg_averages_coverage", lambda db: {})
    monkeypatch.setattr(dashboard, "get_esg_top_metrics", lambda db, client_id, top_n=10: [])

    ctx = {
        "allocation_reference_name": "Allocation test",
        "esg_exclusions_check": {"client_exclusions": [{"code": "fossiles", "label": "Fossiles"}]},
    }

    out = dashboard._populate_synthese_esg_context(_NoopDB(), ctx, client_id=123)

    assert out["esg_exclusions_check"] == ctx["esg_exclusions_check"]
