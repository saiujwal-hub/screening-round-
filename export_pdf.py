"""
CPRI & MIT Bengaluru - PowerNext-AI Screening Round Challenge
Methodology Note PDF Exporter & Page Count Verifier

Converts methodology_note.md to a publication-ready PDF using headless Chromium/Edge
and verifies that the resulting document adheres to the strict 2-page limit.
"""

import os
import re
import subprocess
import markdown


def generate_methodology_pdf(
    md_path="methodology_note.md",
    pdf_path="methodology_note.pdf",
    html_path="methodology_note.html"
):
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Methodology note not found at '{md_path}'")

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Convert markdown to html
    html_body = markdown.markdown(md_content, extensions=['extra', 'tables'])

    html_full = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CPRI Screening Round — Methodology Note</title>
<style>
    @page {{
        size: A4;
        margin: 1.4cm 1.5cm;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 9.4pt;
        line-height: 1.34;
        color: #1a1a1a;
    }}
    h1 {{
        font-size: 13pt;
        margin: 0 0 3px 0;
        color: #0f172a;
        font-weight: 700;
    }}
    h2 {{
        font-size: 10.4pt;
        margin: 7px 0 3px 0;
        color: #1e293b;
        font-weight: 600;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 2px;
    }}
    h3 {{
        font-size: 9.6pt;
        margin: 4px 0 2px 0;
        color: #1e293b;
        font-weight: 600;
    }}
    p {{
        margin: 3px 0 4px 0;
    }}
    ul, ol {{
        margin: 3px 0 4px 0;
        padding-left: 17px;
    }}
    li {{
        margin-bottom: 2px;
    }}
    strong {{
        color: #0f172a;
    }}
    hr {{
        border: none;
        border-top: 1px solid #cbd5e1;
        margin: 5px 0;
    }}
    code {{
        font-family: Consolas, Monaco, monospace;
        font-size: 8.8pt;
        background: #f1f5f9;
        padding: 1px 3px;
        border-radius: 3px;
    }}
    * {{
        hyphens: none !important;
        -webkit-hyphens: none !important;
    }}
    img {{
        max-width: 98%;
        max-height: 160px;
        height: auto;
        display: block;
        margin: 4px auto 1px auto;
    }}
    em {{
        font-size: 8.2pt;
        color: #334155;
        display: block;
        text-align: center;
        margin-top: 2px;
        margin-bottom: 4px;
    }}
    .equation-box {{
        text-align: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 5px 8px;
        margin: 5px auto;
        font-family: Consolas, "Segoe UI", sans-serif;
        font-weight: 600;
        color: #0f172a;
        font-size: 9.0pt;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_full)

    # Locate Chromium or Edge
    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    browser_exe = None
    for cand in edge_candidates:
        if os.path.exists(cand):
            browser_exe = cand
            break

    if browser_exe is None:
        raise RuntimeError("Neither Microsoft Edge nor Google Chrome found for headless PDF printing.")

    abs_html = os.path.abspath(html_path).replace(os.sep, '/')
    abs_pdf = os.path.abspath(pdf_path)

    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html}"
    ]

    try:
        subprocess.run(cmd, check=True)

        # Verify Page Count
        with open(abs_pdf, "rb") as f:
            pdf_bytes = f.read()

        pages = re.findall(rb"/Type\s*/Page\b", pdf_bytes)
        page_count = len(pages)

        print(f"=== Methodology Note PDF Export ===")
        print(f"Output PDF: {abs_pdf} ({len(pdf_bytes) / 1024.0:.1f} KB)")
        print(f"Word Count: {len(md_content.split())} words")
        print(f"Page Count: {page_count} page(s)")
        if page_count <= 2:
            print(f"VERIFICATION: PASSED (Document satisfies the <= 2-page challenge constraint)\n")
        else:
            raise ValueError(f"VERIFICATION FAILED: Document has {page_count} pages, exceeding the 2-page limit!")

        return page_count
    finally:
        if os.path.exists(html_path):
            try:
                os.remove(html_path)
            except Exception:
                pass


if __name__ == "__main__":
    generate_methodology_pdf()
