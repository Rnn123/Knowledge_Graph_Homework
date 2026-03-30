from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_TEXT_PATH = BASE_DIR / "data" / "raw" / "turing_corpus.txt"
OUTPUT_DIR = BASE_DIR / "output"


ENTITY_DEFINITIONS = [
    {
        "name": "Alan Turing",
        "type": "PERSON",
        "aliases": ["Alan Turing"],
    },
    {
        "name": "Alonzo Church",
        "type": "PERSON",
        "aliases": ["Alonzo Church"],
    },
    {
        "name": "King's College, Cambridge",
        "type": "ORGANIZATION",
        "aliases": ["King's College, Cambridge"],
    },
    {
        "name": "Princeton University",
        "type": "ORGANIZATION",
        "aliases": ["Princeton University"],
    },
    {
        "name": "Bletchley Park",
        "type": "ORGANIZATION",
        "aliases": ["Bletchley Park"],
    },
    {
        "name": "Government Code and Cypher School",
        "type": "ORGANIZATION",
        "aliases": ["Government Code and Cypher School"],
    },
    {
        "name": "National Physical Laboratory",
        "type": "ORGANIZATION",
        "aliases": ["National Physical Laboratory"],
    },
    {
        "name": "University of Manchester",
        "type": "ORGANIZATION",
        "aliases": ["University of Manchester"],
    },
    {
        "name": "London",
        "type": "LOCATION",
        "aliases": ["London"],
    },
    {
        "name": "Cambridge",
        "type": "LOCATION",
        "aliases": ["Cambridge"],
    },
    {
        "name": "Wilmslow",
        "type": "LOCATION",
        "aliases": ["Wilmslow"],
    },
    {
        "name": "England",
        "type": "LOCATION",
        "aliases": ["England"],
    },
    {
        "name": "On Computable Numbers",
        "type": "WORK",
        "aliases": ["On Computable Numbers"],
    },
    {
        "name": "Computing Machinery and Intelligence",
        "type": "WORK",
        "aliases": ["Computing Machinery and Intelligence"],
    },
    {
        "name": "Turing machine",
        "type": "CONCEPT",
        "aliases": ["Turing machine"],
    },
    {
        "name": "Turing Test",
        "type": "CONCEPT",
        "aliases": ["Turing Test"],
    },
    {
        "name": "Bombe",
        "type": "DEVICE",
        "aliases": ["Bombe"],
    },
    {
        "name": "Automatic Computing Engine",
        "type": "DEVICE",
        "aliases": ["Automatic Computing Engine"],
    },
    {
        "name": "Manchester Mark I",
        "type": "DEVICE",
        "aliases": ["Manchester Mark I"],
    },
    {
        "name": "Enigma",
        "type": "DEVICE",
        "aliases": ["Enigma"],
    },
    {
        "name": "World War II",
        "type": "EVENT",
        "aliases": ["World War II"],
    },
]


def read_corpus(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()]


def build_entity_pattern(aliases: list[str]) -> re.Pattern[str]:
    ordered_aliases = sorted(aliases, key=len, reverse=True)
    escaped = [re.escape(alias) for alias in ordered_aliases]
    return re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(escaped), re.IGNORECASE)


def extract_entities(sentences: list[str]) -> list[dict]:
    entity_store: dict[str, dict] = {}
    for entity in ENTITY_DEFINITIONS:
        entity_store[entity["name"]] = {
            "name": entity["name"],
            "type": entity["type"],
            "mention_count": 0,
            "sentence_ids": [],
            "evidence": [],
        }

    for sentence_id, sentence in enumerate(sentences, start=1):
        for entity in ENTITY_DEFINITIONS:
            pattern = build_entity_pattern(entity["aliases"])
            matches = pattern.findall(sentence)
            if matches:
                record = entity_store[entity["name"]]
                record["mention_count"] += len(matches)
                if sentence_id not in record["sentence_ids"]:
                    record["sentence_ids"].append(sentence_id)
                if sentence not in record["evidence"]:
                    record["evidence"].append(sentence)

        for year in re.findall(r"\b(?:18|19|20)\d{2}\b", sentence):
            if year not in entity_store:
                entity_store[year] = {
                    "name": year,
                    "type": "TIME",
                    "mention_count": 0,
                    "sentence_ids": [],
                    "evidence": [],
                }
            record = entity_store[year]
            record["mention_count"] += 1
            if sentence_id not in record["sentence_ids"]:
                record["sentence_ids"].append(sentence_id)
            if sentence not in record["evidence"]:
                record["evidence"].append(sentence)

    extracted = [record for record in entity_store.values() if record["mention_count"] > 0]
    return sorted(extracted, key=lambda item: (item["type"], item["name"]))


def add_relation(
    relations: list[dict],
    seen: set[tuple[str, str, str]],
    subject: str,
    predicate: str,
    obj: str,
    sentence: str,
) -> None:
    key = (subject, predicate, obj)
    if key in seen:
        return
    relations.append(
        {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "evidence": sentence,
        }
    )
    seen.add(key)


def extract_relations(sentences: list[str]) -> list[dict]:
    relations: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for sentence in sentences:
        birth_match = re.search(
            r"Alan Turing was born in (?P<location>[A-Za-z]+) in (?P<year>\d{4})\.",
            sentence,
        )
        if birth_match:
            add_relation(relations, seen, "Alan Turing", "BORN_IN", birth_match.group("location"), sentence)
            add_relation(relations, seen, "Alan Turing", "BORN_IN_YEAR", birth_match.group("year"), sentence)
            continue

        study_match = re.search(
            r"Alan Turing studied at (?P<organization>King's College, Cambridge)\.",
            sentence,
        )
        if study_match:
            add_relation(relations, seen, "Alan Turing", "STUDIED_AT", study_match.group("organization"), sentence)
            continue

        advanced_study_match = re.search(
            r"Alan Turing later studied at (?P<organization>Princeton University) under (?P<mentor>Alonzo Church)\.",
            sentence,
        )
        if advanced_study_match:
            add_relation(relations, seen, "Alan Turing", "STUDIED_AT", advanced_study_match.group("organization"), sentence)
            add_relation(relations, seen, "Alan Turing", "STUDIED_UNDER", advanced_study_match.group("mentor"), sentence)
            continue

        publication_match = re.search(
            r"In (?P<year>\d{4}), Alan Turing published \"(?P<work>On Computable Numbers)\", which introduced the idea of the (?P<concept>Turing machine)\.",
            sentence,
        )
        if publication_match:
            work = publication_match.group("work")
            add_relation(relations, seen, "Alan Turing", "PUBLISHED", work, sentence)
            add_relation(relations, seen, work, "PUBLISHED_IN_YEAR", publication_match.group("year"), sentence)
            add_relation(relations, seen, work, "INTRODUCED", publication_match.group("concept"), sentence)
            continue

        wartime_match = re.search(
            r"During (?P<event>World War II), Alan Turing worked at (?P<organization>Bletchley Park) for the (?P<employer>Government Code and Cypher School)\.",
            sentence,
        )
        if wartime_match:
            add_relation(relations, seen, "Alan Turing", "WORKED_AT", wartime_match.group("organization"), sentence)
            add_relation(relations, seen, "Alan Turing", "WORKED_FOR", wartime_match.group("employer"), sentence)
            continue

        design_match = re.search(
            r"Alan Turing helped design the (?P<device>Bombe) to break (?P<target>Enigma)-encrypted messages\.",
            sentence,
        )
        if design_match:
            add_relation(relations, seen, "Alan Turing", "DESIGNED", design_match.group("device"), sentence)
            add_relation(relations, seen, design_match.group("device"), "USED_AGAINST", design_match.group("target"), sentence)
            continue

        ace_match = re.search(
            r"After the war, Alan Turing worked at the (?P<organization>National Physical Laboratory) and proposed the (?P<device>Automatic Computing Engine)\.",
            sentence,
        )
        if ace_match:
            add_relation(relations, seen, "Alan Turing", "WORKED_AT", ace_match.group("organization"), sentence)
            add_relation(relations, seen, "Alan Turing", "PROPOSED", ace_match.group("device"), sentence)
            continue

        manchester_match = re.search(
            r"Alan Turing later joined the (?P<organization>University of Manchester) and contributed to the (?P<device>Manchester Mark I)\.",
            sentence,
        )
        if manchester_match:
            add_relation(relations, seen, "Alan Turing", "JOINED", manchester_match.group("organization"), sentence)
            add_relation(relations, seen, "Alan Turing", "CONTRIBUTED_TO", manchester_match.group("device"), sentence)
            continue

        test_match = re.search(
            r"In (?P<year>\d{4}), Alan Turing proposed the (?P<concept>Turing Test) in the paper \"(?P<work>Computing Machinery and Intelligence)\"\.",
            sentence,
        )
        if test_match:
            add_relation(relations, seen, "Alan Turing", "PROPOSED", test_match.group("concept"), sentence)
            add_relation(relations, seen, test_match.group("concept"), "DESCRIBED_IN", test_match.group("work"), sentence)
            add_relation(relations, seen, test_match.group("concept"), "PUBLISHED_IN_YEAR", test_match.group("year"), sentence)
            continue

        death_match = re.search(
            r"Alan Turing died in (?P<city>Wilmslow), (?P<country>England), in (?P<year>\d{4})\.",
            sentence,
        )
        if death_match:
            add_relation(relations, seen, "Alan Turing", "DIED_IN", death_match.group("city"), sentence)
            add_relation(relations, seen, "Alan Turing", "DIED_IN_YEAR", death_match.group("year"), sentence)
            add_relation(relations, seen, death_match.group("city"), "LOCATED_IN", death_match.group("country"), sentence)

    return relations


def write_json(path: Path, data: list[dict]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8-sig")


def write_csv(path: Path, relations: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["subject", "predicate", "object"])
        writer.writeheader()
        for relation in relations:
            writer.writerow(
                {
                    "subject": relation["subject"],
                    "predicate": relation["predicate"],
                    "object": relation["object"],
                }
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    text = read_corpus(RAW_TEXT_PATH)
    sentences = split_sentences(text)

    entities = extract_entities(sentences)
    relations = extract_relations(sentences)

    write_json(OUTPUT_DIR / "entities.json", entities)
    write_json(OUTPUT_DIR / "relations.json", relations)
    write_csv(OUTPUT_DIR / "triples.csv", relations)

    summary = defaultdict(int)
    for entity in entities:
        summary[entity["type"]] += 1

    print("Knowledge extraction completed.")
    print(f"Sentences: {len(sentences)}")
    print(f"Entities: {len(entities)}")
    print(f"Relations: {len(relations)}")
    print("Entity type summary:")
    for entity_type in sorted(summary):
        print(f"  {entity_type}: {summary[entity_type]}")


if __name__ == "__main__":
    main()
