import pytest

from vglab.errors import ContractError
from vglab.ids import ExperimentId, split_sha256


def test_experiment_id_is_stable_and_includes_config_prefix():
    experiment = ExperimentId(
        scene_id="dev-01",
        scene_split="dev",
        arm="S1",
        seed=17,
        config_sha256="abcdef0123456789" + "0" * 48,
    )
    assert experiment.text() == "p1:dev:dev-01:S1:seed17:abcdef012345"
    again = ExperimentId(
        scene_id="dev-01",
        scene_split="dev",
        arm="S1",
        seed=17,
        config_sha256="abcdef0123456789" + "0" * 48,
    )
    assert experiment.text() == again.text()


def test_experiment_id_rejects_unknown_arm_and_split():
    with pytest.raises(ContractError):
        ExperimentId("dev-01", "dev", "S9", 17, "a" * 64).text()
    with pytest.raises(ContractError):
        ExperimentId("dev-01", "train", "S0", 17, "a" * 64).text()
    with pytest.raises(ContractError):
        ExperimentId("dev-01", "dev", "S0", 17, "short").text()


def test_split_hash_excludes_its_own_field_and_tracks_membership():
    frames = {
        "a": "1" * 64,
        "b": "2" * 64,
        "c": "3" * 64,
    }
    first = split_sha256("scene", ["a"], ["b"], ["c"], frames)
    same = split_sha256("scene", ["a"], ["b"], ["c"], frames)
    assert first == same
    swapped = split_sha256("scene", ["b"], ["a"], ["c"], frames)
    assert first != swapped
    reordered = split_sha256("scene", ["a"], ["b"], ["c"], frames)
    assert first == reordered
    order_sensitive = split_sha256("scene", ["c", "a"], [], [], {"a": frames["a"], "c": frames["c"]})
    order_flipped = split_sha256("scene", ["a", "c"], [], [], {"a": frames["a"], "c": frames["c"]})
    assert order_sensitive != order_flipped


def test_split_hash_does_not_read_a_preexisting_hash_field():
    frames = {"a": "1" * 64}
    digest = split_sha256("scene", ["a"], [], [], frames)
    assert digest != split_sha256("scene", ["a"], [], [], {"a": "2" * 64})
    assert len(digest) == 64
