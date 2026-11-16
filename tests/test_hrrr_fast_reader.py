"""Tests for ``ptype.hrrr_fast.reader``.

Everything except the ``network``-marked test runs offline: idx parsing uses embedded idx text
(surface lines copied verbatim from ``hrrr.20231226/conus/hrrr.t00z.wrfnatf{00,12,24}.grib2.idx``),
and the decode bookkeeping is tested with a fake decoder so no GRIB library is required.

Run the network test with ``PTYPE_NETWORK_TESTS=1 pytest -m network tests/test_hrrr_fast_reader.py``.
"""

import os
from datetime import date

import numpy as np
import pytest
from ptype.hrrr_fast import reader

# Real surface/near-surface lines from the f12 wrfnat idx (message numbers and offsets unchanged).
# The two APCP lines are the duplicate-window case: run total 0-12 h and the hourly 11-12 h bucket.
SURFACE_F12 = """\
1032:678586186:d=2023122600:PRES:surface:12 hour fcst:
1033:680109258:d=2023122600:HGT:surface:12 hour fcst:
1034:682262953:d=2023122600:TMP:surface:12 hour fcst:
1035:683541636:d=2023122600:ASNOW:surface:0-12 hour acc fcst:
1038:685476474:d=2023122600:WEASD:surface:12 hour fcst:
1041:686556972:d=2023122600:TMP:2 m above ground:12 hour fcst:
1043:688971513:d=2023122600:SPFH:2 m above ground:12 hour fcst:
1044:690411514:d=2023122600:DPT:2 m above ground:12 hour fcst:
1047:694433668:d=2023122600:UGRD:10 m above ground:12 hour fcst:
1048:696815283:d=2023122600:VGRD:10 m above ground:12 hour fcst:
1049:699196898:d=2023122600:WIND:10 m above ground:11-12 hour max fcst:
1052:703360975:d=2023122600:CPOFP:surface:12 hour fcst:
1053:703611229:d=2023122600:PRATE:surface:12 hour fcst:
1054:703732743:d=2023122600:APCP:surface:0-12 hour acc fcst:
1055:704524762:d=2023122600:WEASD:surface:0-12 hour acc fcst:
1060:705109663:d=2023122600:APCP:surface:11-12 hour acc fcst:
1061:705812265:d=2023122600:WEASD:surface:11-12 hour acc fcst:
1063:706143827:d=2023122600:CSNOW:surface:12 hour fcst:
1064:706166079:d=2023122600:CICEP:surface:12 hour fcst:
1065:706171837:d=2023122600:CFRZR:surface:12 hour fcst:
1066:706189145:d=2023122600:CRAIN:surface:12 hour fcst:
1067:706287761:d=2023122600:SFCR:surface:12 hour fcst:
"""

# f00: analyses are labelled "anl" and the only APCP window is the empty "0-0 day acc fcst".
SURFACE_F00 = """\
1032:595836089:d=2023122600:PRES:surface:anl:
1033:597353995:d=2023122600:HGT:surface:anl:
1041:603408759:d=2023122600:TMP:2 m above ground:anl:
1044:607047755:d=2023122600:DPT:2 m above ground:anl:
1047:610449035:d=2023122600:UGRD:10 m above ground:anl:
1048:612830650:d=2023122600:VGRD:10 m above ground:anl:
1053:618789663:d=2023122600:PRATE:surface:anl:
1054:618789851:d=2023122600:APCP:surface:0-0 day acc fcst:
1063:618866686:d=2023122600:CRAIN:surface:anl:
1064:618866700:d=2023122600:CSNOW:surface:anl:
1065:618866800:d=2023122600:CICEP:surface:anl:
1066:618866900:d=2023122600:CFRZR:surface:anl:
1067:618867000:d=2023122600:SFCR:surface:anl:
"""

# f24: the run total switches to day units ("0-1 day acc fcst") while the bucket stays in hours.
APCP_F24 = """\
1054:705988115:d=2023122600:APCP:surface:0-1 day acc fcst:
1060:707874511:d=2023122600:APCP:surface:23-24 hour acc fcst:
"""

# Order of variables within one hybrid level in HRRR wrfnat (first 18 of the 20 per level).
HYBRID_ORDER = ["PRES", "CLMR", "CIMIXR", "RWMR", "SNMR", "GRLE", "NCONCD", "NCCICE", "SPNCR", "PMTF", "PMTC", "FRACCC", "HGT", "TMP", "SPFH", "UGRD", "VGRD", "VVEL"]


def hybrid_idx(n_levels: int, step: str = "12 hour fcst", msg_size: int = 1000) -> str:
    """Build hybrid-level idx lines in HRRR order (all variables for level 1, then level 2, ...)."""
    lines, offset, num = [], 0, 1
    for lev in range(1, n_levels + 1):
        for var in HYBRID_ORDER:
            lines.append(f"{num}:{offset}:d=2023122600:{var}:{lev} hybrid level:{step}:")
            offset += msg_size
            num += 1
    return "\n".join(lines) + "\n"


def make_idx(n_levels: int, surface: str, step: str = "12 hour fcst") -> str:
    """Hybrid lines followed by surface lines; surface offsets are far beyond the hybrid ones."""
    return hybrid_idx(n_levels, step) + surface


def test_parse_idx_offsets_and_ends():
    entries = reader.parse_idx(SURFACE_F12)
    assert len(entries) == 22
    first = entries[0]
    assert (first.number, first.offset, first.variable, first.level, first.step) == ("1032", 678586186, "PRES", "surface", "12 hour fcst")
    # end is the next message's start, so message sizes are exact
    assert first.end == 680109258
    assert first.size == 680109258 - 678586186
    # the last line's end is unknown until a HEAD request
    assert entries[-1].end is None and entries[-1].size is None


def test_parse_idx_sorts_and_handles_submessages():
    text = "2:200:d=1:B:surface:anl:\n1:0:d=1:A:surface:anl:\n2.1:200:d=1:C:surface:anl:\n3:500:d=1:D:surface:anl:\n"
    entries = reader.parse_idx(text)
    assert [e.offset for e in entries] == [0, 200, 200, 500]
    # sub-messages sharing an offset both end at the next distinct offset
    assert [e.end for e in entries] == [200, 500, 500, None]


def test_parse_idx_rejects_garbage():
    with pytest.raises(ValueError):
        reader.parse_idx("1:0:d=1:TMP\n")


@pytest.mark.parametrize(
    "step, expected",
    [
        ("anl", ("inst", 0.0, 0.0)),
        ("12 hour fcst", ("inst", 12.0, 12.0)),
        ("0-12 hour acc fcst", ("acc", 0.0, 12.0)),
        ("11-12 hour acc fcst", ("acc", 11.0, 12.0)),
        ("0-0 day acc fcst", ("acc", 0.0, 0.0)),
        ("0-1 day acc fcst", ("acc", 0.0, 24.0)),
        ("0-2 day acc fcst", ("acc", 0.0, 48.0)),
        ("11-12 hour max fcst", ("max", 11.0, 12.0)),
        ("15 min fcst", ("inst", 0.25, 0.25)),
    ],
)
def test_parse_step(step, expected):
    assert reader.parse_step(step) == expected


def test_parse_step_unknown():
    with pytest.raises(ValueError):
        reader.parse_step("bogus")


def _by_name(plan):
    return {(m.name, m.level_index): m.entry for m in plan.messages}


def test_select_messages_f12_duplicate_apcp():
    entries = reader.parse_idx(make_idx(4, SURFACE_F12))
    plan = reader.select_messages(entries, n_levels=3, fxx=12)
    names = _by_name(plan)
    # 6 variables x 3 levels + 11 surface fields + APCP
    assert len(plan.messages) == 6 * 3 + len(reader.SURFACE_FIELDS) + 1
    # the hourly bucket, not the run total, is selected
    assert names[("APCP_1h", None)].step == "11-12 hour acc fcst"
    assert names[("APCP_1h", None)].offset == 705109663
    # instantaneous WEASD would not be confused with accumulations; spot-check 2 m / 10 m names
    assert names[("TMP_2m", None)].level == "2 m above ground"
    assert names[("DPT_2m", None)].variable == "DPT"
    assert names[("UGRD_10m", None)].level == "10 m above ground"
    # level 4 is not requested, other hybrid variables (CLMR, VVEL) are not selected
    assert not any(m.entry.level == "4 hybrid level" for m in plan.messages)
    assert {m.entry.variable for m in plan.messages if m.level_index is not None} == set(reader.HYBRID_VARIABLES)
    # plan is in file order so get_ranges can coalesce neighbours
    assert plan.starts == sorted(plan.starts)


def test_select_messages_f00_and_day_units():
    plan = reader.select_messages(reader.parse_idx(make_idx(2, SURFACE_F00, step="anl")), n_levels=2, fxx=0)
    assert _by_name(plan)[("APCP_1h", None)].step == "0-0 day acc fcst"

    surface_f24 = SURFACE_F12.replace("12 hour fcst", "24 hour fcst")
    surface_f24 = "\n".join(line for line in surface_f24.splitlines() if ":APCP:" not in line) + "\n" + APCP_F24
    plan = reader.select_messages(reader.parse_idx(make_idx(1, surface_f24, step="24 hour fcst")), n_levels=1, fxx=24)
    assert _by_name(plan)[("APCP_1h", None)].step == "23-24 hour acc fcst"


def test_select_messages_f01_single_apcp():
    surface_f01 = SURFACE_F12.replace("12 hour fcst", "1 hour fcst")
    surface_f01 = "\n".join(line for line in surface_f01.splitlines() if ":APCP:" not in line)
    surface_f01 += "\n1054:703732743:d=2023122600:APCP:surface:0-1 hour acc fcst:\n"
    plan = reader.select_messages(reader.parse_idx(make_idx(1, surface_f01, step="1 hour fcst")), n_levels=1, fxx=1)
    assert _by_name(plan)[("APCP_1h", None)].step == "0-1 hour acc fcst"


def test_select_messages_missing_fields():
    entries = reader.parse_idx(make_idx(2, SURFACE_F12))
    with pytest.raises(KeyError, match="hybrid"):
        reader.select_messages(entries, n_levels=3, fxx=12)
    with pytest.raises(KeyError, match="APCP"):
        reader.select_messages(entries, n_levels=2, fxx=5)
    no_dpt = reader.parse_idx(make_idx(1, "\n".join(line for line in SURFACE_F12.splitlines() if ":DPT:" not in line) + "\n"))
    with pytest.raises(KeyError, match="DPT"):
        reader.select_messages(no_dpt, n_levels=1, fxx=12)
    with pytest.raises(ValueError):
        reader.select_messages(entries, n_levels=0, fxx=12)


def test_hybrid_levels_are_ordered_surface_upward():
    entries = reader.parse_idx(make_idx(5, SURFACE_F12))
    plan = reader.select_messages(entries, n_levels=5, fxx=12)
    tmp = sorted((m for m in plan.messages if m.name == "TMP"), key=lambda m: m.level_index)
    assert [m.level_index for m in tmp] == [0, 1, 2, 3, 4]
    assert [m.entry.level for m in tmp] == [f"{i} hybrid level" for i in range(1, 6)]


def test_decode_places_levels_with_fake_decoder(monkeypatch):
    """decode() must put hybrid level k at index k-1 regardless of worker scheduling."""
    entries = reader.parse_idx(make_idx(4, SURFACE_F12))
    plan = reader.select_messages(entries, n_levels=4, fxx=12)

    # Each fake "message" encodes the hybrid level number (0 for 2D fields) in its bytes.
    def level_of(m):
        lev = m.entry.level.split()[0]
        return int(lev) if m.level_index is not None else 0

    msgs = [level_of(m).to_bytes(2, "little") for m in plan.messages]
    monkeypatch.setitem(reader.DECODERS, "fake", lambda b: np.full((3, 5), int.from_bytes(bytes(b), "little"), dtype=np.float64))
    for workers in (1, 4):
        out = reader.decode(plan, msgs, decoder="fake", workers=workers, pool="thread")
        for v in reader.HYBRID_VARIABLES:
            assert out[v].shape == (4, 3, 5) and out[v].dtype == np.float32
            np.testing.assert_array_equal(out[v][:, 0, 0], [1, 2, 3, 4])
        assert out["PRES_sfc"].shape == (3, 5)
        assert set(out) == set(reader.HYBRID_VARIABLES) | set(reader.SURFACE_FIELDS) | {reader.APCP_NAME}


def test_decode_argument_checks():
    plan = reader.select_messages(reader.parse_idx(make_idx(1, SURFACE_F12)), n_levels=1, fxx=12)
    with pytest.raises(ValueError):
        reader.decode(plan, [b""], decoder="eccodes")
    with pytest.raises(ValueError):
        reader.decode(plan, [b""] * len(plan.messages), decoder="nope")


def test_merged_ranges():
    assert reader.merged_ranges([0, 10, 100], [5, 20, 110], coalesce=0) == [(0, 5), (10, 20), (100, 110)]
    assert reader.merged_ranges([0, 10, 100], [5, 20, 110], coalesce=5) == [(0, 20), (100, 110)]
    assert reader.merged_ranges([100, 0], [110, 5], coalesce=1000) == [(0, 110)]


def test_hrrr_key():
    assert reader.hrrr_key(date(2023, 12, 26), 0, 12) == "hrrr.20231226/conus/hrrr.t00z.wrfnatf12.grib2"


def test_lcc_forward_matches_hrrr_first_point():
    # Reference x/y from pyproj 3.8.0 with the HRRR grid parameters read from the GRIB header.
    x, y = reader.lcc_forward(-122.719528, 21.138123, -97.5, 38.5, 38.5, 38.5, 6371229.0)
    assert x == pytest.approx(-2697520.1425, abs=0.01)
    assert y == pytest.approx(-1587306.1526, abs=0.01)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("PTYPE_NETWORK_TESTS") != "1", reason="network test; set PTYPE_NETWORK_TESTS=1")
def test_read_real_file_from_s3():
    fields, grid = reader.read_hrrr_fields(date(2023, 12, 26), 0, 12, n_levels=2, decoder="eccodes", workers=4)
    assert (grid.nx, grid.ny, grid.row0_is_south) == (1799, 1059, True)
    assert fields["TMP"].shape == (2, 1059, 1799)
    # hybrid level 2 lies above level 1 everywhere
    assert np.all(fields["HGT"][1] > fields["HGT"][0])
    assert np.all(fields["HGT"][0] > fields["HGT_sfc"])
