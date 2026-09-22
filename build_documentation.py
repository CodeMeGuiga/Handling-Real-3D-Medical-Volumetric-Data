"""Build Documentation.pdf from Documentation.md, typeset to look like LaTeX article.cls.

There is no pandoc, LaTeX, wkhtmltopdf or weasyprint in this environment, so the pipeline
is deliberately small and self-contained:

    Documentation.md
      -> a hand-rolled Markdown to HTML converter (below)
      -> a self-contained HTML file with Latin Modern embedded as base64 @font-face
      -> headless Microsoft Edge, --print-to-pdf
      -> Documentation.pdf, verified by extracting its text with pypdf

The converter is intentionally *not* a general Markdown parser. It handles exactly the
constructs Documentation.md uses: ATX headings, paragraphs, bullet and numbered lists,
pipe tables, fenced code blocks, horizontal rules, and inline bold/italic/code. Anything
fancier should either be avoided in the source or added here deliberately.

Run:  python build_documentation.py

Two pitfalls this script exists to prevent, both hit before:

1. `file:///` + a raw Windows path containing a space renders Edge's "file not found"
   page instead of the document, and the resulting PDF looks plausible by file size.
   The URL is therefore always built with `pathlib.Path.as_uri()`, and the output is
   verified by extracting real text rather than by trusting the size.
2. Fonts are fetched from CTAN once into a local cache directory. The cache is
   gitignored, so a fresh clone re-downloads; builds after that are offline.
"""

from __future__ import annotations

import base64
import html
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Documentation.md"
OUTPUT = HERE / "Documentation.pdf"
FONT_CACHE = HERE / "assets" / "fonts"

EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

CTAN = "https://mirrors.ctan.org/fonts/lm/fonts/opentype/public/lm/"
FONTS = {
    # css family, weight, style -> CTAN file name
    ("LMRoman", "400", "normal"): "lmroman10-regular.otf",
    ("LMRoman", "700", "normal"): "lmroman10-bold.otf",
    ("LMRoman", "400", "italic"): "lmroman10-italic.otf",
    ("LMMono", "400", "normal"): "lmmono10-regular.otf",
}


# --------------------------------------------------------------------------- fonts
def font_bytes(filename: str) -> bytes:
    """Return the font file, downloading it into the local cache on first use."""
    cached = FONT_CACHE / filename
    if cached.exists():
        return cached.read_bytes()
    FONT_CACHE.mkdir(parents=True, exist_ok=True)
    url = CTAN + filename
    print(f"  fetching {filename} from CTAN")
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    cached.write_bytes(data)
    return data


def font_face_rules() -> str:
    rules = []
    for (family, weight, style), filename in FONTS.items():
        b64 = base64.b64encode(font_bytes(filename)).decode("ascii")
        rules.append(
            "@font-face {\n"
            f"  font-family: '{family}';\n"
            f"  font-weight: {weight};\n"
            f"  font-style: {style};\n"
            f"  src: url(data:font/otf;base64,{b64}) format('opentype');\n"
            "}"
        )
    return "\n".join(rules)


# ------------------------------------------------------------------- markdown -> html
def inline(text: str) -> str:
    """Inline spans: code, bold, italic. Code is extracted first so its contents are
    never re-processed as emphasis."""
    placeholders: list[str] = []

    def stash(match: re.Match) -> str:
        placeholders.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00{len(placeholders) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m.group(1))], text)
    return text


def render_table(rows: list[str]) -> str:
    """A pipe table, rendered booktabs-style: rules only, no vertical lines."""
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    header, body = cells[0], cells[2:]  # cells[1] is the |---|---| separator
    out = ["<table>", "<thead><tr>"]
    out += [f"<th>{inline(c)}</th>" for c in header]
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def slug(text: str) -> str:
    """Stable anchor id for a heading."""
    return "sec-" + re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", text).lower()).strip("-")


def build_toc(headings: list[tuple[int, str, str]]) -> str:
    """Table of contents from the collected (level, text, id) triples.

    Chapters and appendices are listed; part dividers act as group headers; subsections
    are indented one level. Anything deeper is omitted to keep the contents readable.
    """
    rows = ['<div class="toc">', "<h2 class=\"toc-title\">Contents</h2>"]
    for level, text, anchor in headings:
        plain = re.sub(r"<[^>]+>", "", text)
        if level == 2 and plain.startswith("Part "):
            css = "toc-part"
        elif level == 2:
            css = "toc-chapter"
        else:
            css = "toc-section"
        rows.append(f'<p class="{css}"><a href="#{anchor}">{text}</a></p>')
    rows.append("</div>")
    return "\n".join(rows)


def convert(markdown: str) -> str:
    lines = markdown.split("\n")
    out: list[str] = []
    headings: list[tuple[int, str, str]] = []
    i = 0
    # `first_in_section` drives article.cls's indentation rule: the paragraph directly
    # after a heading is flush left, every following paragraph is first-line indented.
    first_in_section = True

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            first_in_section = False
            continue

        if re.fullmatch(r"-{3,}", stripped):
            out.append("<hr>")
            first_in_section = True
            i += 1
            continue

        if stripped == "[[TOC]]":
            out.append("<!--TOC-->")
            first_in_section = True
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            body = inline(heading.group(2))
            anchor = slug(heading.group(2))
            # Part dividers get their own page and their own look; chapters start a page.
            css = ""
            if level == 2:
                css = ' class="part"' if heading.group(2).startswith("Part ") else ' class="chapter"'
            out.append(f'<h{level} id="{anchor}"{css}>{body}</h{level}>')
            if level in (2, 3):
                headings.append((level, body, anchor))
            first_in_section = True
            i += 1
            continue

        if stripped.startswith("|"):
            table: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table.append(lines[i])
                i += 1
            out.append(render_table(table))
            first_in_section = False
            continue

        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        number = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if bullet or number:
            tag = "ul" if bullet else "ol"
            pattern = r"^\s*[-*]\s+(.*)$" if bullet else r"^\s*\d+\.\s+(.*)$"
            items: list[str] = []
            while i < len(lines):
                match = re.match(pattern, lines[i])
                if match:
                    items.append(match.group(1))
                    i += 1
                elif lines[i].startswith(("  ", "\t")) and lines[i].strip() and items:
                    items[-1] += " " + lines[i].strip()   # continuation of the last item
                    i += 1
                else:
                    break
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            first_in_section = False
            continue

        paragraph = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^\s*(#{1,6}\s|[-*]\s|\d+\.\s|\||```|-{3,}$)", lines[i]
        ):
            paragraph.append(lines[i].strip())
            i += 1
        css = "noindent" if first_in_section else ""
        out.append(f'<p class="{css}">{inline(" ".join(paragraph))}</p>')
        first_in_section = False

    return "\n".join(out).replace("<!--TOC-->", build_toc(headings))


# ------------------------------------------------------------------------ the page
CSS = """
html { font-size: 10pt; }
body {
  font-family: 'LMRoman', 'Latin Modern Roman', Georgia, serif;
  line-height: 1.30;
  text-align: justify;
  hyphens: auto;
  margin: 0;
  color: #000;
}
@page { size: A4; margin: 1in; }

h1 { font-size: 1.85rem; text-align: center; font-weight: bold; margin: 3.5rem 0 0.4rem 0; }
h2 { font-size: 1.25rem; font-weight: bold; margin: 1.4rem 0 0.6rem 0; }
h3 { font-size: 1.05rem; font-weight: bold; margin: 1.0rem 0 0.4rem 0; }
h2, h3 { text-align: left; page-break-after: avoid; }

/* Each chapter opens a page, the way a book sets them. */
h2.chapter { page-break-before: always; padding-bottom: 0.25rem; border-bottom: 0.5pt solid #000; }

/* Part dividers get a page to themselves. */
h2.part {
  page-break-before: always; page-break-after: avoid;
  text-align: center; font-size: 1.5rem; margin-top: 38vh; border: none;
}
h2.part + hr { display: none; }

/* Table of contents. */
.toc { page-break-after: always; margin-top: 1.2rem; }
.toc-title { page-break-before: avoid; border: none; text-align: center; margin-bottom: 1rem; }
.toc a { color: #000; text-decoration: none; }
.toc p { text-indent: 0; margin: 0; text-align: left; }
.toc-part { font-weight: bold; margin-top: 0.8rem !important; }
.toc-chapter { margin-left: 0.8em !important; }
.toc-section { margin-left: 2.2em !important; font-size: 0.92rem; color: #333; }

p { margin: 0; text-indent: 1.5em; }
p.noindent { text-indent: 0; }

/* The italic paragraph straight after the title plays the role of \\begin{abstract}. */
h1 + p { text-indent: 0; margin: 0.6rem auto 1.4rem auto; width: 86%; font-size: 0.95rem; }

ul, ol { margin: 0.45rem 0 0.45rem 0; padding-left: 1.6em; }
li { margin: 0.12rem 0; text-align: justify; }

code, pre { font-family: 'LMMono', 'Latin Modern Mono', Consolas, monospace; font-size: 0.92em; }
pre {
  background: #f7f7f5; padding: 0.55rem 0.75rem; margin: 0.7rem 0;
  white-space: pre-wrap; page-break-inside: avoid;
  border-left: 1.8pt solid #999; font-size: 0.86rem; line-height: 1.25;
}
pre code { font-size: inherit; }

/* booktabs: horizontal rules only. */
table { border-collapse: collapse; margin: 0.7rem auto; width: 100%; font-size: 0.93rem;
        page-break-inside: avoid; }
thead th { border-top: 1.1pt solid #000; border-bottom: 0.5pt solid #000;
           padding: 0.28rem 0.45rem; text-align: left; font-weight: bold; }
tbody tr:last-child td { border-bottom: 1.1pt solid #000; }
td { padding: 0.24rem 0.45rem; vertical-align: top; text-align: left; }

hr { border: none; border-top: 0.4pt solid #bbb; margin: 1.1rem 0; }
"""


def build_html(markdown: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Handling Real 3D Medical Volumetric Data</title>
<style>
{font_face_rules()}
{CSS}
</style></head>
<body>
{convert(markdown)}
</body></html>
"""


# ----------------------------------------------------------------------- rendering
def find_edge() -> Path:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return candidate
    sys.exit("Microsoft Edge not found; edit EDGE_CANDIDATES")


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    # as_uri() percent-encodes the spaces in this project's folder name. A raw
    # "file:///" + str(path) silently renders Edge's error page instead.
    url = html_path.as_uri()
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [str(find_edge()), "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}", url],
            check=True, capture_output=True, timeout=180,
        )


def verify(pdf_path: Path, markdown: str) -> None:
    """Extract real text and check it against the source, rather than trusting size."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  [!] pypdf not installed; skipping text verification. "
              "Install with: pip install pypdf")
        return

    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # Latin Modern sets fi, fl and friends as single ligature glyphs, so extracted text
    # contains U+FB01 and company. Expand them before matching anything.
    for ligature, expansion in (("\ufb00", "ff"), ("\ufb01", "fi"), ("\ufb02", "fl"),
                                ("\ufb03", "ffi"), ("\ufb04", "ffl")):
        text = text.replace(ligature, expansion)

    if len(text) < 2000:
        sys.exit(f"extracted only {len(text)} characters: the PDF is probably an error page")

    title = markdown.splitlines()[0].lstrip("# ").strip()
    if title.split(":")[0] not in text:
        sys.exit(f"title {title!r} missing from the extracted text")

    for dash in ("\u2014", "\u2013"):
        if dash in text:
            sys.exit(f"found {dash!r} in the rendered PDF; this project uses neither")

    print(f"  verified: {len(reader.pages)} pages, {len(text):,} characters of text, no em/en dashes")


def main() -> None:
    print(f"building {OUTPUT.name} from {SOURCE.name}")
    if not SOURCE.exists():
        sys.exit(f"Source file not found: {SOURCE}")

    markdown = SOURCE.read_text(encoding="utf-8")
    page = build_html(markdown)

    html_path = HERE / "Documentation.html"
    html_path.write_text(page, encoding="utf-8")
    try:
        render_pdf(html_path, OUTPUT)
        verify(OUTPUT, markdown)
    finally:
        # The intermediate HTML is a build artifact, not a deliverable.
        html_path.unlink(missing_ok=True)
    print(f"  wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
