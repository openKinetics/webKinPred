import json

import pandas as pd

from api.utils import get_experimental
from api.utils.extra_info import build_extra_info


def test_matched_row_with_blank_fields_is_json_safe(tmp_path, monkeypatch):
    path = tmp_path / "km.csv"
    inchi = get_experimental.smiles_to_inchi("CCO")
    pd.DataFrame(
        [
            {
                "protein_sequence": "MKT",
                "substrate_inchi": inchi,
                "km_value": 1.5,
                "km_SD": None,
                "ph": None,
                "temperature": 25.0,
                "protein_ID": "P1",
                "from_brenda": 1,
            }
        ]
    ).to_csv(path, index=False)
    monkeypatch.setattr(get_experimental, "KM_CSV", str(path))

    [found, missing] = get_experimental.lookup_experimental(["MKT", "MKT"], ["CCO", "CC"], "Km")

    json.dumps([found, missing], allow_nan=False)
    assert found["found"] and found["km_value"] == 1.5
    assert found["km_SD"] is None and found["ph"] is None
    assert missing == {"found": False, "idx": 1}
    text = build_extra_info(found, "Km")
    assert "Standard deviation is not reported" in text
    assert "BRENDA" in text
