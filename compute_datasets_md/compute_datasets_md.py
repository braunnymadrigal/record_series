from pathlib import Path
import pymupdf4llm

# Constants
INPUT_DIR = Path("datasets_pdf")
OUTPUT_DIR = Path("datasets_md")
MAX_CHARS = 5_000


def convert_pdf_to_md(pdf_path: Path) -> None:
    """
    Convert one PDF into Markdown, preserving the relative folder structure.
    Output is limited to MAX_CHARS characters.
    """

    relative_path = pdf_path.relative_to(INPUT_DIR)
    output_path = OUTPUT_DIR / relative_path.with_suffix(".md")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {pdf_path}")

    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))

        md_text = md_text[:MAX_CHARS]

        output_path.write_text(md_text, encoding="utf-8")

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"ERROR processing {pdf_path}: {e}")


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = [
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_path in pdf_files:
        convert_pdf_to_md(pdf_path)

    print("Done.")


if __name__ == "__main__":
    main()