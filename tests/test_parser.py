import pytest
from arxiv_api import parseAtomFeed, ArxivApiError, ArxivError

FIXTURES = "tests/fixtures"


def load(name: str) -> str:
    with open(f"{FIXTURES}/{name}") as f:
        return f.read()


class TestParseAtomFeed:
    def test_single_entry(self):
        entries = parseAtomFeed(load("single_entry.xml"))
        assert len(entries) == 1
        entry = entries[0]
        assert entry["id"] == "http://arxiv.org/abs/2512.10504v2"
        assert entry["title"] == "Tianyan: Cloud services with quantum advantage"
        assert entry["authors"] == ["Tianyan Quantum Group"]
        assert entry["abstract"] == "Tianyan Quantum Cloud Platform offers cloud services demonstrating quantum advantage capabilities."
        assert entry["published"] == "2025-12-11T10:23:39Z"
        assert entry["pdf_url"] == "https://arxiv.org/pdf/2512.10504v2"

    def test_multi_entry(self):
        entries = parseAtomFeed(load("multi_entry.xml"))
        assert len(entries) == 2
        assert entries[0]["title"] == "Tianyan: Cloud services with quantum advantage"
        assert entries[1]["title"] == "Quantum error correction with surface codes"
        assert entries[1]["authors"] == ["Alice Smith", "Bob Jones"]

    def test_no_pdf_link(self):
        entries = parseAtomFeed(load("no_pdf.xml"))
        assert len(entries) == 1
        assert entries[0]["pdf_url"] is None

    def test_empty_results(self):
        entries = parseAtomFeed(load("empty_results.xml"))
        assert entries == []

    def test_error_entry_raises(self):
        with pytest.raises(ArxivApiError, match="incorrect id format for bad.12345"):
            parseAtomFeed(load("error_entry.xml"))

    def test_malformed_xml_raises(self):
        with pytest.raises(ArxivError, match="Failed to parse arXiv API response"):
            parseAtomFeed(load("malformed.xml"))
