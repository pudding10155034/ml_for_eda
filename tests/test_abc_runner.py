import pytest

from riskaware_eda.abc_runner import ABCExecutionError, parse_print_stats
from riskaware_eda.types import NetworkStats


def test_parse_official_abc_print_stats_format():
    output = """
    ABC command line: r i10.aig; ps
    i10 : i/o = 257/224 lat = 0 and = 2396 lev = 37
    i10 : i/o = 257/224 lat = 0 and = 1851 lev = 35
    """
    assert parse_print_stats(output) == NetworkStats(
        pis=257, pos=224, nodes=1851, depth=35, latches=0
    )


def test_parse_alternate_node_format():
    output = "network : pi = 10 po = 2 nd = 91 lev = 8"
    assert parse_print_stats(output) == NetworkStats(
        pis=10, pos=2, nodes=91, depth=8
    )


def test_parse_stats_rejects_unrelated_output():
    with pytest.raises(ABCExecutionError):
        parse_print_stats("ABC finished without any statistics")
