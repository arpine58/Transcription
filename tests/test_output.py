from livetx.output import format_line


def test_line_has_time_speaker_hy_and_en():
    line = format_line(72.4, "Speaker 1", "Բարև, ինչպե՞ս ես։", "Hi, how are you?")
    assert line.startswith("[00:01:12] Speaker 1")
    assert "HY: Բարև, ինչպե՞ս ես։" in line and "EN: Hi, how are you?" in line
