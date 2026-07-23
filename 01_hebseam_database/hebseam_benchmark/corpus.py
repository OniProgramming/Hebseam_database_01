from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VOCALIZATION = "consonantal"
SPLIT_PREFIXES = True
DIVINE_NAME_STRONGS = {"3068", "136", "430", "410", "433", "7706", "5945"}

OSIS_NS = "{http://www.bibletechnologies.net/2003/OSIS/namespace}"
CANTILLATION = set(range(0x0591, 0x05B0)) | {0x05BD, 0x05BF, 0x05C0, 0x05C3, 0x05C4, 0x05C5, 0x05C6}
NIQQUD = set(range(0x05B0, 0x05BD)) | {0x05C1, 0x05C2, 0x05C7}
CONSONANTS = set(range(0x05D0, 0x05EB))


def devocalize(text: str) -> str:
    if VOCALIZATION == "pointed":
        keep = CONSONANTS | NIQQUD | CANTILLATION
    elif VOCALIZATION == "vocalized":
        keep = CONSONANTS | NIQQUD
    else:
        keep = CONSONANTS
    return "".join(ch for ch in text if ord(ch) in keep)


def strongs_number(lemma: str) -> str:
    match = re.search(r"\d+", lemma)
    return match.group(0) if match else ""


@dataclass(frozen=True)
class Token:
    surface: str
    lemma: str = ""
    strongs: str = ""
    morph: str = ""
    is_prefix: bool = False
    is_proper: bool = False
    is_divine: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Token":
        return Token(**data)


@dataclass
class Verse:
    osis: str
    chapter: int
    verse: int
    tokens: list[Token] = field(default_factory=list)
    source: str = ""

    @property
    def text(self) -> str:
        return " ".join(t.surface for t in self.tokens if t.surface)

    @property
    def annotation_complete(self) -> bool:
        return bool(self.tokens) and all(bool(t.morph) for t in self.tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "osis": self.osis,
            "chapter": self.chapter,
            "verse": self.verse,
            "source": self.source,
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Verse":
        return Verse(
            osis=data.get("osis", ""),
            chapter=int(data.get("chapter", 0)),
            verse=int(data.get("verse", 0)),
            source=data.get("source", ""),
            tokens=[Token.from_dict(t) for t in data.get("tokens", [])],
        )


def _split_word(element: ET.Element) -> list[Token]:
    raw = "".join(element.itertext())
    morph = (element.get("morph") or "").lstrip("HA")
    lemma = element.get("lemma") or ""
    surfaces = raw.split("/")
    morphs = morph.split("/") if morph else [""]
    lemmas = lemma.split("/") if lemma else [""]
    n = max(len(surfaces), len(morphs), len(lemmas))

    def pad(values: list[str]) -> list[str]:
        return values + [values[-1]] * (n - len(values)) if values else [""] * n

    surfaces, morphs, lemmas = pad(surfaces), pad(morphs), pad(lemmas)
    tokens: list[Token] = []
    for i, (surface, morph_code, lemma_code) in enumerate(zip(surfaces, morphs, lemmas)):
        s = devocalize(surface)
        if not s:
            continue
        strongs = strongs_number(lemma_code)
        tokens.append(Token(
            surface=s,
            lemma=lemma_code.strip(),
            strongs=strongs,
            morph=morph_code,
            is_prefix=i < n - 1,
            is_proper=morph_code.startswith("Np"),
            is_divine=strongs in DIVINE_NAME_STRONGS,
        ))
    if SPLIT_PREFIXES:
        return tokens
    if not tokens:
        return []
    return [Token(
        surface=devocalize(raw.replace("/", "")),
        lemma=lemma,
        strongs=tokens[-1].strongs,
        morph=tokens[-1].morph,
        is_proper=tokens[-1].is_proper,
        is_divine=any(t.is_divine for t in tokens),
    )]


def load_osis(path: str | Path, source: str) -> list[Verse]:
    root = ET.parse(path)
    verses: list[Verse] = []
    for element in root.iter(f"{OSIS_NS}verse"):
        osis = element.get("osisID")
        if not osis:
            continue
        parts = osis.split(".")
        if len(parts) < 3:
            continue
        verse = Verse(osis=osis, chapter=int(parts[-2]), verse=int(parts[-1]), source=source)
        for word in element.iter(f"{OSIS_NS}w"):
            verse.tokens.extend(_split_word(word))
        if verse.tokens:
            verses.append(verse)
    return verses


def pseudo_verses(words: list[str], *, source: str, target_tokens: int = 21) -> list[Verse]:
    result: list[Verse] = []
    for i in range(0, len(words) - target_tokens + 1, target_tokens):
        chunk = words[i:i + target_tokens]
        tokens = [Token(surface=devocalize(w)) for w in chunk if devocalize(w)]
        if tokens:
            result.append(Verse(f"{source}.{len(result)+1}", 0, len(result)+1, tokens, source))
    return result
