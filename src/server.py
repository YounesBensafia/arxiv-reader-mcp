from mcp.server.fastmcp import FastMCP
from arxiv_api import search, get_by_id, get_recent, ArxivError

server = FastMCP(
    "arxiv-mcp-server",
    instructions="Search and retrieve papers from arXiv.",
)


@server.tool(description="Search arXiv papers by keyword, author, category, and/or date range")
def search_arxiv(
    keyword: str | None = None,
    author: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_results: int = 10,
    start: int = 0,
) -> str:
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
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   ID: {r['id'].removeprefix('http://arxiv.org/abs/')}")
        lines.append(f"   Authors: {', '.join(r['authors'])}")
        lines.append(f"   Published: {r['published']}")
        if r["pdf_url"]:
            lines.append(f"   PDF: {r['pdf_url']}")
        lines.append("")
    return "\n".join(lines)


@server.tool(description="Fetch a single paper by its arXiv ID (e.g. 2301.07041)")
def get_paper(arxiv_id: str) -> str:
    try:
        result = get_by_id(arxiv_id)
    except ArxivError as e:
        return f"Error: {e}"
    if not result:
        return f"No paper found with ID {arxiv_id}."
    return (
        f"Title: {result['title']}\n"
        f"ID: {result['id'].removeprefix('http://arxiv.org/abs/')}\n"
        f"Authors: {', '.join(result['authors'])}\n"
        f"Published: {result['published']}\n"
        f"Abstract: {result['abstract']}\n"
        f"PDF: {result['pdf_url']}"
    )


@server.tool(description="Get the most recent papers in a given category (e.g. cs.AI, quant-ph)")
def get_recent_papers(category: str, max_results: int = 10) -> str:
    try:
        results = get_recent(category, max_results=max_results)
    except ArxivError as e:
        return f"Error: {e}"
    if not results:
        return f"No recent papers found in category {category}."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   ID: {r['id'].removeprefix('http://arxiv.org/abs/')}")
        lines.append(f"   Authors: {', '.join(r['authors'])}")
        lines.append(f"   Published: {r['published']}")
        if r["pdf_url"]:
            lines.append(f"   PDF: {r['pdf_url']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    server.run()
