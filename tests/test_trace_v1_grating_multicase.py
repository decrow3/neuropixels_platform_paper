import pandas as pd

from scripts.trace_v1_grating_multicase import select_cases


def test_multicase_selection_is_deterministic_and_balanced():
    rows = []
    for tf in (1.0, 2.0, 4.0, 8.0, 15.0):
        for index in range(8):
            rows.append(
                {
                    "site": "site2",
                    "session_id": 2,
                    "unit_id": int(tf * 1000 + index),
                    "preferred_tf_hz": tf,
                    "log10_raw_mod_idx": -0.2,
                    "log10_source_corrected_mod_idx": -0.2 + index / 100,
                    "source_log10_mod_idx_gain": index / 100,
                    "source_log10_psd_sd_gain": (index - 3) / 100,
                }
            )
    units = pd.DataFrame(rows)
    first = select_cases(units, audit_per_tf=3)
    second = select_cases(units, audit_per_tf=3)
    pd.testing.assert_frame_equal(first, second)
    audit = first.loc[first.selection_roles.str.contains("outcome_blind_audit_sample")]
    assert audit.groupby("preferred_tf_hz").size().eq(3).all()
    assert set(audit.preferred_tf_hz) == {1.0, 2.0, 4.0, 8.0, 15.0}
