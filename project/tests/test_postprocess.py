from src.postprocess import correct_ru_plate, normalize_plate


def test_normalize_cyrillic_to_latin():
    # Cyrillic plate letters -> Latin look-alikes, uppercased, alnum only
    assert normalize_plate("а123вс77") == "A123BC77"
    assert normalize_plate("Х 999 ХХ 199") == "X999XX199"


def test_normalize_strips_symbols():
    assert normalize_plate("a-001-bp!54") == "A001BP54"


def test_correct_ru_plate_fixes_confusions():
    # O on digit positions -> 0; Z->7, S->5 on digit positions
    assert correct_ru_plate("AOO9XX123") == "A009XX123"
    assert correct_ru_plate("AOOZAE799") == "A007AE799"


def test_correct_ru_plate_keeps_letter_positions():
    # digit look-alikes on letter positions -> letters (0->O)
    assert correct_ru_plate("A123BC77") == "A123BC77"


def test_correct_ru_plate_ignores_wrong_length():
    # only 8/9-char strings are corrected
    assert correct_ru_plate("54") == "54"
    assert correct_ru_plate("ABC") == "ABC"
