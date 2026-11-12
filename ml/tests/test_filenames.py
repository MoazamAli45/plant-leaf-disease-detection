import pytest

from leafdx.data.filenames import leaf_map_key, parse_npd_filename, resolve_leaf_id

U = "01a66316-0e98-4d3b-a56f-d78752cd043f___FREC_Scab 3003"


@pytest.mark.parametrize(
    ("name", "aug"),
    [
        (f"{U}.JPG", "original"),
        (f"{U}_270deg.JPG", "270deg"),
        (f"{U}_90deg.JPG", "90deg"),
        (f"{U}_180deg.jpg", "180deg"),
        (f"{U}_flipLR.JPG", "flipLR"),
        (f"{U}_flipTB.JPG", "flipTB"),
        (f"{U}_new30degFlipLR.JPG", "new30degFlipLR"),
        (f"{U}_new30degFlipTB.JPG", "new30degFlipTB"),
        (f"{U}_newGRR.JPG", "newGRR"),
        (f"{U}_flipLR_90deg.JPG", "flipLR+90deg"),
    ],
)
def test_parse_strips_augmentation_suffix(name, aug):
    p = parse_npd_filename(name)
    assert p.source_stem == U
    assert p.aug_type == aug


def test_identifier_underscores_are_not_suffixes():
    # "RS_HL" contains an underscore but is part of the photo id
    p = parse_npd_filename("abc___RS_HL 6251.JPG")
    assert p == parse_npd_filename("abc___RS_HL 6251.JPG")
    assert p.source_stem == "abc___RS_HL 6251"
    assert p.aug_type == "original"


def test_leaf_map_key_matches_official_loader():
    assert leaf_map_key(U) == "frec_scab 3003"
    assert leaf_map_key("x___RS_HL 1234 copy") == "rs_hl 1234"
    assert leaf_map_key("x___RS_HL 1234_final_masked") == "rs_hl 1234"


def test_resolve_leaf_id_requires_matching_class():
    leaf_map = {"rs_hl 6251": ["Soybean___healthy:::398.0", "Apple___healthy:::101.0"]}
    assert (
        resolve_leaf_id("a___RS_HL 6251", "Apple___healthy", leaf_map) == "Apple___healthy:::101.0"
    )
    assert resolve_leaf_id("a___RS_HL 6251", "Soybean___healthy", leaf_map) == (
        "Soybean___healthy:::398.0"
    )
    # single suggestion from another class is rejected (official loader would accept it)
    assert resolve_leaf_id("a___RS_HL 6251", "Tomato___healthy", leaf_map) is None
    assert resolve_leaf_id("a___unknown 1", "Apple___healthy", leaf_map) is None
