from pathlib import Path

from scripts.openevo_webshop.carrier_contract_successor_202609300734 import validate_route_contract


def test_stage2_contract_cannot_masquerade_as_e3() -> None:
    repo = Path(__file__).resolve().parents[1]
    stage2 = repo / "configs/experiment/path-contracts/stage2-qwen3-path-contract-202609300734.json"
    try:
        validate_route_contract(repo, "E3_RANK_CAPACITY", stage2)
    except ValueError as exc:
        assert "route contract path/SHA drift" in str(exc)
    else:
        raise AssertionError("Stage2 contract was admitted as E3")


def test_e3_contract_cannot_masquerade_as_stage2() -> None:
    repo = Path(__file__).resolve().parents[1]
    e3 = repo / "configs/experiment/path-contracts/post-advisor-e3-rank-capacity-path-contract-202609300734.json"
    try:
        validate_route_contract(repo, "STAGE2_QWEN3", e3)
    except ValueError as exc:
        assert "route contract path/SHA drift" in str(exc)
    else:
        raise AssertionError("E3 contract was admitted as Stage2")
