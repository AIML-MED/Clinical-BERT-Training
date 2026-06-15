import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


def inline_markup(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="MicrosoftYaHei">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#647482"))
    canvas.drawCentredString(A4[0] / 2, 9 * mm, str(document.page))
    canvas.restoreState()


def build_story(markdown: str, styles: dict):
    story = []
    paragraph_lines = []
    list_items = []
    code_lines = []
    in_code = False

    def flush_paragraph():
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines)
            story.append(Paragraph(inline_markup(text), styles["body"]))
            paragraph_lines.clear()

    def flush_list():
        if list_items:
            items = [ListItem(Paragraph(inline_markup(item), styles["list"])) for item in list_items]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=15, bulletFontName="MicrosoftYaHei"))
            story.append(Spacer(1, 2 * mm))
            list_items.clear()

    def flush_code():
        if code_lines:
            story.append(Preformatted("\n".join(code_lines), styles["code"]))
            story.append(Spacer(1, 2 * mm))
            code_lines.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            flush_list()
            if in_code:
                flush_code()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            flush_list()
            continue

        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1))
            story.append(Paragraph(inline_markup(heading.group(2)), styles[f"h{level}"]))
            continue

        bullet = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.*)$", line)
        if bullet:
            flush_paragraph()
            list_items.append(bullet.group(1))
            continue

        if line.strip() == "---":
            flush_paragraph()
            flush_list()
            story.append(Spacer(1, 3 * mm))
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    flush_list()
    flush_code()
    return story


def main():
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("MicrosoftYaHeiBold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))

    styles = getSampleStyleSheet()
    custom = {
        "h1": ParagraphStyle(
            "H1", fontName="MicrosoftYaHeiBold", fontSize=20, leading=27,
            textColor=colors.HexColor("#17365d"), spaceAfter=12, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2", fontName="MicrosoftYaHeiBold", fontSize=14, leading=20,
            textColor=colors.HexColor("#17365d"), spaceBefore=15, spaceAfter=7, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3", fontName="MicrosoftYaHeiBold", fontSize=11.5, leading=17,
            textColor=colors.HexColor("#285b78"), spaceBefore=10, spaceAfter=5, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body", fontName="MicrosoftYaHei", fontSize=9.5, leading=15,
            textColor=colors.HexColor("#18212b"), spaceAfter=7,
        ),
        "list": ParagraphStyle(
            "List", fontName="MicrosoftYaHei", fontSize=9.3, leading=14,
            textColor=colors.HexColor("#18212b"), spaceAfter=2,
        ),
        "code": ParagraphStyle(
            "Code", fontName="MicrosoftYaHei", fontSize=8.2, leading=12,
            leftIndent=8, rightIndent=8, borderColor=colors.HexColor("#4f86a6"),
            borderWidth=0.7, borderPadding=7, backColor=colors.HexColor("#f4f7f9"),
            textColor=colors.HexColor("#263746"), splitLongWords=True,
        ),
    }

    document = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="OSCAR Continued Pretraining 数据处理与运行记录",
        author="Clinical-BERT-Training",
    )
    markdown = input_path.read_text(encoding="utf-8")
    document.build(build_story(markdown, custom), onFirstPage=footer, onLaterPages=footer)
    print(output_path)


if __name__ == "__main__":
    main()
