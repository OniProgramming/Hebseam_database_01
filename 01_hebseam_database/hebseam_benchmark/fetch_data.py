from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
from pathlib import Path

BOOKS = ("Gen", "Exod", "Ps")
TRACTATES = (
    "Seder Zeraim/Mishnah Berakhot",
    "Seder Nezikin/Mishnah Bava Kamma",
    "Seder Nezikin/Mishnah Bava Metzia",
    "Seder Nezikin/Mishnah Bava Batra",
    "Seder Nezikin/Pirkei Avot",
    "Seder Moed/Mishnah Shabbat",
    "Seder Nashim/Mishnah Ketubot",
    "Seder Moed/Mishnah Pesachim",
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"exists: {destination}")
        return
    urllib.request.urlretrieve(url, destination)
    print(f"downloaded: {destination} ({destination.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    for book in BOOKS:
        download(
            f"https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/{book}.xml",
            data_dir / f"{book}.xml",
        )

    base = "https://raw.githubusercontent.com/cltk/hebrew_text_sefaria/master/json/Mishnah"
    for i, tractate in enumerate(TRACTATES):
        encoded = urllib.parse.quote(tractate)
        download(f"{base}/{encoded}/Hebrew/merged.json", data_dir / f"m_{i}.json")


if __name__ == "__main__":
    main()
