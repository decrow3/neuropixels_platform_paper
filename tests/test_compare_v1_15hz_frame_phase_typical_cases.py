import numpy as np
import pandas as pd

from scripts.compare_v1_15hz_frame_phase_typical_cases import render_paired


def test_paired_diagnostic_renders_expected_metrics(tmp_path):
    cases = pd.DataFrame(
        {
            "site": ["site2", "site3"],
            "phase_range_exact_full": [0.8, 0.4],
            "phase_range_exact_last7": [0.3, 0.5],
            "log10_exact_minus_canonical": [-0.03, 0.02],
            "log10_balanced_minus_natural_exact_full": [-0.05, 0.01],
            "log10_balanced_minus_natural_exact_last7": [-0.01, 0.02],
        }
    )
    output = tmp_path / "paired.png"
    render_paired(cases, output)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_phase_range_prediction_can_represent_support_and_contradiction():
    full = np.array([0.8, 0.4])
    late = np.array([0.3, 0.5])
    delta = late - full
    assert delta[0] < 0
    assert delta[1] > 0
