from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_figure4_variant_review import (  # noqa: E402
    apply_anatomical_v1_filter,
    build_mousev2_probe_v1_audit,
    load_figure_cells,
    load_inputs,
)


def test_figure4_mousev2_units_are_visp_and_border_entries_use_deeper_units():
    groups, _, _ = load_inputs()
    _, filtered_cells, _ = apply_anatomical_v1_filter(groups, load_figure_cells())
    audit = build_mousev2_probe_v1_audit(filtered_cells)

    assert len(audit) == 32
    assert audit["subject_id"].nunique() == 8
    assert audit["n_visp_contacts"].gt(0).all()

    outside = audit.loc[~audit["surface_agrees_visp"]]
    assert set(zip(outside["subject_id"], outside["probe"])) == {
        (810532, "ProbeA"),
        (810532, "ProbeB"),
    }
    assert outside["all_used_units_deeper_than_entry"].all()
    assert outside["n_figure4_visp_units"].gt(0).all()
