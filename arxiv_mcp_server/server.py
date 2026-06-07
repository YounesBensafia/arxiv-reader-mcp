import re
import os
import io
import tarfile
import tempfile
import httpx
import fitz
import camelot
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from mcp.server.fastmcp import FastMCP
from .arxiv_api import search, get_by_id, get_recent, ArxivError

server = FastMCP(
    "arxiv-mcp-server",
    instructions="Search and retrieve papers from arXiv.",
)

_DATE_PATTERN = re.compile(r"^\d{10,12}$")


def _validate_max_results(v: int) -> str | None:
    if v < 1:
        return "max_results must be at least 1"
    if v > 30000:
        return "max_results cannot exceed 30000 (arXiv API limit)"
    return None


def _validate_date(v: str | None, name: str) -> str | None:
    if v is not None and not _DATE_PATTERN.match(v):
        return f"{name} must be in YYYYMMDDHHMM format (e.g. 202401010000)"
    return None


def _fmt(entries: list[dict]) -> str:
    lines = []
    for i, r in enumerate(entries, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   ID: {r['id'].removeprefix('http://arxiv.org/abs/')}")
        lines.append(f"   Authors: {', '.join(r['authors'])}")
        lines.append(f"   Published: {r['published']}")
        if r["pdf_url"]:
            lines.append(f"   PDF: {r['pdf_url']}")
        lines.append("")
    return "\n".join(lines)


@server.tool(
    description="Search arXiv papers by keyword, author, category, and/or date range. "
    "Use when the user wants to find papers matching specific terms, by a specific author, "
    "in a specific category, or within a date range. Supports Boolean-like searches via keyword. "
    "Returns a numbered list with title, arXiv ID, authors, publication date, and PDF link.",
)
def search_arxiv(
    keyword: str | None = None,
    author: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_results: int = 10,
    start: int = 0,
) -> str:
    if not any([keyword, author, category, date_from, date_to]):
        return "Error: Provide at least one of: keyword, author, category, date_from, date_to"

    err = _validate_max_results(max_results)
    if err:
        return f"Error: {err}"

    if start < 0:
        return "Error: start must be 0 or greater"

    err = _validate_date(date_from, "date_from")
    if err:
        return f"Error: {err}"
    err = _validate_date(date_to, "date_to")
    if err:
        return f"Error: {err}"

    try:
        results = search(
            keyword=keyword,
            author=author,
            category=category,
            date_from=date_from,
            date_to=date_to,
            max_results=max_results,
            start=start,
        )
    except ArxivError as e:
        return f"Error: {e}"
    if not results:
        return "No results found."
    return _fmt(results)


@server.tool(
    description="Fetch a single paper by its arXiv ID (e.g. 2301.07041). "
    "Use when the user provides a specific arXiv ID and wants full metadata including "
    "title, authors, publication date, abstract, and PDF link. "
    "IDs can be with or without version suffix (e.g. 2301.07041v2).",
)
def get_paper(arxiv_id: str) -> str:
    if not arxiv_id or not arxiv_id.strip():
        return "Error: arxiv_id must not be empty"
    try:
        result = get_by_id(arxiv_id.strip())
    except ArxivError as e:
        return f"Error: {e}"
    if not result:
        return f"Error: No paper found with ID '{arxiv_id}'."
    return (
        f"Title: {result['title']}\n"
        f"ID: {result['id'].removeprefix('http://arxiv.org/abs/')}\n"
        f"Authors: {', '.join(result['authors'])}\n"
        f"Published: {result['published']}\n"
        f"Abstract: {result['abstract']}\n"
        f"PDF: {result['pdf_url']}"
    )


@server.tool(
    description="Get the most recent papers in a given category (e.g. cs.AI, quant-ph). "
    "Use when the user wants to see the latest submissions in a specific arXiv category, "
    "sorted by submission date (newest first). Ideal for 'what is new' style queries.",
)
def get_recent(category: str, max_results: int = 10) -> str:
    if not category or not category.strip():
        return "Error: category must not be empty (e.g. cs.AI, quant-ph)"

    err = _validate_max_results(max_results)
    if err:
        return f"Error: {err}"

    try:
        results = get_recent(category.strip(), max_results=max_results)
    except ArxivError as e:
        return f"Error: {e}"
    if not results:
        return f"No recent papers found in category '{category}'."
    return _fmt(results)


@server.tool(
    description="Search papers by query text, with optional category filter and date range. "
    "Simpler than search_arxiv - just a search query plus optional category and date_from. "
    "Use when the user provides a natural language query like 'papers about transformers' "
    "and optionally a category or start date. The query searches across all fields (title, abstract, authors).",
)
def search_papers(
    query: str,
    category: str | None = None,
    max_results: int = 10,
    date_from: str | None = None,
) -> str:
    if not query or not query.strip():
        return "Error: query must not be empty"

    if category is not None and not category.strip():
        return "Error: if provided, category must not be empty"

    err = _validate_max_results(max_results)
    if err:
        return f"Error: {err}"

    err = _validate_date(date_from, "date_from")
    if err:
        return f"Error: {err}"

    try:
        results = search(
            keyword=query.strip(),
            category=category.strip() if category else None,
            max_results=max_results,
            date_from=date_from,
        )
    except ArxivError as e:
        return f"Error: {e}"
    if not results:
        return "No results found."
    return _fmt(results)


def _extract_tables_from_ar5iv(arxiv_id: str) -> list[str] | None:
    url = f"https://ar5iv.org/abs/{arxiv_id}"
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    html_tables = soup.find_all("table")
    if not html_tables:
        return None

    result = []
    for table in html_tables:
        for math in table.find_all("math"):
            alttext = math.get("alttext", "")
            math.replace_with(alttext)

        caption_el = table.find("caption")
        caption = caption_el.get_text(" ", strip=True) if caption_el else ""

        rows = table.find_all("tr")
        if not rows:
            continue

        md_lines = []
        header_done = False
        for row in rows:
            cells = row.find_all(["th", "td"])
            texts = [cell.get_text(" ", strip=True) for cell in cells]
            if not any(texts):
                continue
            md_lines.append("| " + " | ".join(texts) + " |")
            if not header_done and any(c.name == "th" for c in cells):
                md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
                header_done = True

        if md_lines:
            table_md = "\n".join(md_lines)
            if caption:
                table_md = f"*{caption}*\n\n{table_md}"
            result.append(table_md)

    return result if result else None


def _latex_tabular_to_markdown(tabular: str) -> str | None:
    tabular = re.sub(r"(?<!\\)%.*", "", tabular)
    tabular = re.sub(
        r"\\(?:hline|toprule|midrule|bottomrule|cline|hline)"
        r"(?:\[.*?\])?",
        "", tabular,
    )
    tabular = re.sub(
        r"\\(?:cmidrule)(?:\(.*?\))?(?:\[.*?\])?\{.*?\}",
        "", tabular,
    )
    tabular = re.sub(r"\\noalign\{.*?\}", "", tabular)

    def _replace_multicolumn(text):
        def _skip_braced(s, pos):
            """Find the next '{' at or after pos, then skip past the matching '}'."""
            while pos < len(s) and s[pos] != "{":
                if s[pos] == "[":
                    pos += 1
                    while pos < len(s) and s[pos] != "]":
                        pos += 1
                    pos += 1
                else:
                    pos += 1
            if pos >= len(s):
                return pos
            pos += 1
            depth = 1
            while pos < len(s) and depth > 0:
                if s[pos] == "{":
                    depth += 1
                elif s[pos] == "}":
                    depth -= 1
                pos += 1
            return pos

        i = 0
        while i < len(text):
            m = re.match(r"\\(?:multicolumn|multirow)", text[i:])
            if not m:
                i += 1
                continue
            start = i
            i += m.end()
            for _ in range(2):
                i = _skip_braced(text, i)
            content_start = i
            i = _skip_braced(text, i)
            inner = text[content_start + 1 : i - 1] 
            text = text[:start] + "{" + inner + "}" + text[i:]
            i = start
        return text

    tabular = _replace_multicolumn(tabular)

    tabular = re.sub(
        r"\\(?:text|textbf|textit|emph|texttt|textsc|textsf|textsl|mathrm|mathbf|mathit)"
        r"\{(.*?)\}",
        r"\1",
        tabular,
    )
    tabular = re.sub(r"\\(?:cite|ref|label|pageref)\{.*?\}", "", tabular)
    tabular = re.sub(r"\$\$(.*?)\$\$", r"\1", tabular, flags=re.DOTALL)
    tabular = re.sub(r"\$(.*?)\$", r"\1", tabular)
    tabular = (
        tabular.replace("\\#", "#")
        .replace("\\%", "%")
        .replace("\\$", "$")
        .replace("\\_", "_")
        .replace("\\&", "&")
        .replace("\\{", "{")
        .replace("\\}", "}")
    )
    tabular = re.sub(r"\\[a-zA-Z]+", "", tabular)
    tabular = tabular.replace("{", "").replace("}", "")

    def _clean_cell(text):
        text = re.sub(r"\\(?:vspace|hspace|smallskip|medskip|bigskip)\{[^}]*\}", "", text)
        text = re.sub(r"\\(\w+)\{([^}]*)\}", r"\2", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    tabular = _clean_cell(tabular)

    rows = [r.strip() for r in re.split(r"\\\\", tabular) if r.strip()]
    if not rows:
        return None

    rows = [r for r in rows if re.search(r"[^\s\-\—\|\&]", r)]

    md_lines = []
    for i, row in enumerate(rows):
        cells = [c.strip() for c in row.split("&")]
        cells = [c for c in cells if c]
        if not cells:
            continue
        md_lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")

    return "\n".join(md_lines) if md_lines else None


def _extract_tables_from_latex(arxiv_id: str) -> list[str] | None:
    url = f"https://arxiv.org/src/{arxiv_id}"
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except Exception:
        return None

    try:
        tar = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
    except Exception:
        return None

    tex_contents = []
    for member in tar.getmembers():
        if member.isfile() and member.name.endswith(".tex"):
            f = tar.extractfile(member)
            if f:
                try:
                    tex_contents.append(f.read().decode("utf-8", errors="replace"))
                except Exception:
                    continue

    if not tex_contents:
        return None

    def _extract_one_table(block: str) -> str | None:
        caption_m = re.search(
            r"\\caption\s*(\[.*?\])?\s*\{(.*?)\}", block, re.DOTALL
        )
        caption = caption_m.group(2).strip() if caption_m else ""
        caption = re.sub(r"\s*\\\\\s*", " ", caption)

        tab_m = re.search(
            r"\\begin\{(tabular\*?|tabularx)\}",
            block, re.DOTALL,
        )
        if not tab_m:
            return None
        env = tab_m.group(1)
        pos = tab_m.end()
        while pos < len(block) and block[pos] == "[":
            pos += 1
            while pos < len(block) and block[pos] != "]":
                pos += 1
            pos += 1
        while pos < len(block) and block[pos] == "{":
            depth = 1
            pos += 1
            while pos < len(block) and depth > 0:
                if block[pos] == "{":
                    depth += 1
                elif block[pos] == "}":
                    depth -= 1
                pos += 1
        body_start = pos
        end_pat = re.compile(r"\\end\{" + re.escape(env) + r"\}")
        end_m = end_pat.search(block, body_start)
        if not end_m:
            return None
        body = block[body_start:end_m.start()]

        md = _latex_tabular_to_markdown(body)
        if md:
            md = md.replace("~", " ")
            caption = caption.replace("~", " ")
            if caption:
                md = f"*{caption}*\n\n{md}"
        return md

    result = []

    table_re = re.compile(
        r"\\begin\{(table\*?)\}(.*?)\\end\{\1\}", re.DOTALL
    )
    for tex in tex_contents:
        for match in table_re.finditer(tex):
            md = _extract_one_table(match.group(2))
            if md:
                result.append(md)

    fig_re = re.compile(
        r"\\begin\{(figure\*?)\}(.*?)\\end\{\1\}", re.DOTALL
    )
    for tex in tex_contents:
        for match in fig_re.finditer(tex):
            if re.search(r"\\begin\{(tabular\*?|tabularx)\}", match.group(2)):
                md = _extract_one_table(match.group(2))
                if md:
                    result.append(md)

    return result if result else None


@server.tool(
    description="Download the PDF for a given arXiv ID and extract its full text content, "
    "including tables formatted as markdown. "
    "Tables are extracted from the best available source: ar5iv.org HTML → LaTeX source → PDF/Camelot. "
    "Use when the user needs to read the full paper text, not just the abstract. "
    "The arXiv ID is required (e.g. 2301.07041).",
)
def fetch_pdf(arxiv_id: str) -> str:
    if not arxiv_id or not arxiv_id.strip():
        return "Error: arxiv_id must not be empty"

    aid = arxiv_id.strip()
    try:
        paper = get_by_id(aid)
    except ArxivError as e:
        return f"Error: {e}"
    if not paper:
        return f"Error: No paper found with ID '{aid}'."
    if not paper["pdf_url"]:
        return f"Error: No PDF URL available for ID '{aid}'."
    try:
        response = httpx.get(paper["pdf_url"], timeout=60, follow_redirects=True)
        response.raise_for_status()
    except httpx.TimeoutException:
        return "Error: PDF download timed out"
    except httpx.ConnectError:
        return "Error: Could not connect to download PDF"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} downloading PDF"

    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(prefix="arxiv_", suffix=".pdf", delete=False)
        tmp.write(response.content)
        tmp.flush()
        tmp.close()
        tmp_path = tmp.name

        doc = fitz.open(tmp_path)
    except Exception as e:
        if tmp and tmp.name:
            os.unlink(tmp.name)
        return f"Error opening PDF: {e}"

    parts = []
    for page_num in range(1, doc.page_count + 1):
        page = doc[page_num - 1]
        page_label = f"--- Page {page_num} ---"
        page_parts = [page_label]
        text = page.get_text("text").strip()
        if text:
            page_parts.append(text)
        parts.append("\n\n".join(page_parts))

    doc.close()

    is_recent = False
    try:
        pub = datetime.fromisoformat(paper["published"].replace("Z", "+00:00"))
        is_recent = datetime.now(timezone.utc) - pub < timedelta(days=7)
    except Exception:
        pass

    ar5iv_tables = None if is_recent else _extract_tables_from_ar5iv(aid)
    latex_tables = _extract_tables_from_latex(aid)
    if is_recent and latex_tables is None:
        ar5iv_tables = _extract_tables_from_ar5iv(aid)

    ar5iv_reason: str | None = None
    latex_reason: str | None = None

    tables: list[str] | None = None
    source: str | None = None

    if ar5iv_tables and latex_tables:
        empty_cells = sum(t.count("|  |") for t in ar5iv_tables)
        if empty_cells < max(3, len(ar5iv_tables)):
            tables, source = ar5iv_tables, "ar5iv"
            latex_reason = "skipped (ar5iv chosen)"
        else:
            tables, source = latex_tables, "LaTeX"
            ar5iv_reason = f"skipped ({empty_cells} empty cells)"
    elif ar5iv_tables:
        tables, source = ar5iv_tables, "ar5iv"
        latex_reason = "source unavailable"
    elif latex_tables:
        tables, source = latex_tables, "LaTeX"
        ar5iv_reason = "source unavailable"
    else:
        source = "Camelot"
        ar5iv_reason = "unavailable"
        latex_reason = "unavailable"
        try:
            ctables = camelot.read_pdf(
                tmp_path, flavor="stream", pages="1-end",
                edge_tol=50, row_tol=10, column_tol=10,
            )
            page_tables: dict[int, list[str]] = {}
            for ct in ctables:
                r, c = ct.shape
                if r < 3 or c < 2 or ct.accuracy < 50:
                    continue
                md = []
                for ri in range(r):
                    cells = [str(ct.df.iloc[ri, ci]).strip() for ci in range(c)]
                    md.append("| " + " | ".join(cells) + " |")
                    if ri == 0:
                        md.append("| " + " | ".join(["---"] * c) + " |")
                page_tables.setdefault(ct.page, []).append("\n".join(md))
            if page_tables:
                tables = []
                for page_num in sorted(page_tables):
                    for ti, table_md in enumerate(page_tables[page_num], 1):
                        tables.append(f"(Page {page_num}) Table {ti}:\n{table_md}")
        except Exception:
            pass

    if tmp and tmp.name:
        os.unlink(tmp.name)
    result = "\n\n".join(parts)
    if tables:
        detail = f"from {source}, {len(tables)} table(s)"
        if ar5iv_reason:
            detail += f", ar5iv: {ar5iv_reason}"
        if latex_reason:
            detail += f", LaTeX: {latex_reason}"
        result += f"\n\n--- Tables ({detail}) ---\n\n" + "\n\n".join(tables)
    return result or "No text or tables could be extracted from the PDF."


def main():
    server.run()


if __name__ == "__main__":
    main()
