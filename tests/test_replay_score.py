"""The speaker scores used to compare tracker settings (tools/speaker_replay.py)."""
from tools.speaker_replay import score


def test_perfect_labels():
    per_person, wrong = score(["Speaker 1", "Speaker 2", "Speaker 1"], ["anna", "ben", "anna"])
    assert per_person == {"anna": 1, "ben": 1} and wrong == 0


def test_one_person_split_in_three():
    per_person, wrong = score(["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 3"], ["anna"] * 4)
    assert per_person == {"anna": 3} and wrong == 0.5


def test_two_people_merged_into_one_label():
    per_person, wrong = score(["Speaker 1"] * 4, ["anna", "anna", "anna", "ben"])
    assert per_person == {"anna": 1, "ben": 1} and wrong == 0.25
