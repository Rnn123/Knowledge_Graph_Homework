from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
RULE_CORPUS_PATH = BASE_DIR / "data" / "raw" / "turing_corpus.txt"
WIKI_CORPUS_PATH = BASE_DIR / "data" / "raw" / "alan_turing_wiki.txt"
MODEL_DIR = BASE_DIR / "models" / "bert-base-ner"
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


LABEL_MAPPING = {
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
    "MISC": "MISC",
}


NOISY_MENTIONS = {
    "alan",
    "turing",
    "english",
    "british",
    "june",
    "file",
    "category",
    "redirect",
    "good article",
}


def read_corpus(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def normalize_text_noise(text: str) -> str:
    replacements = {
        "&nbsp;": " ",
        "鈥?": "-",
        "鈥": "-",
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "’": "'",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def strip_templates(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = re.sub(r"\{\{[^{}]*\}\}", " ", current, flags=re.DOTALL)
    return current


def clean_wiki_markup(text: str) -> str:
    text = normalize_text_noise(text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>/]*?>.*?</ref>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ref[^>]*/>", " ", text, flags=re.IGNORECASE)
    text = strip_templates(text)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\]", " ", text)
    text = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\n\1\n", text)
    text = text.replace("'''", "")
    text = text.replace("''", "")
    text = re.sub(r"^\s*[*!|].*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*;.*$", " ", text, flags=re.MULTILINE)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [sentence.strip() for sentence in parts if sentence.strip()]


def build_entity_pattern(aliases: list[str]) -> re.Pattern[str]:
    ordered_aliases = sorted(aliases, key=len, reverse=True)
    escaped = [re.escape(alias) for alias in ordered_aliases]
    return re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(escaped), re.IGNORECASE)


def extract_entities_rule_based(sentences: list[str]) -> list[dict[str, Any]]:
    entity_store: dict[str, dict[str, Any]] = {}
    for entity in ENTITY_DEFINITIONS:
        entity_store[entity["name"]] = {
            "name": entity["name"],
            "type": entity["type"],
            "mention_count": 0,
            "sentence_ids": [],
            "evidence": [],
            "source": "rule_based_dictionary",
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
                    "source": "rule_based_regex",
                }
            record = entity_store[year]
            record["mention_count"] += 1
            if sentence_id not in record["sentence_ids"]:
                record["sentence_ids"].append(sentence_id)
            if sentence not in record["evidence"]:
                record["evidence"].append(sentence)

    extracted = [record for record in entity_store.values() if record["mention_count"] > 0]
    return sorted(extracted, key=lambda item: (item["type"], item["name"]))


def load_local_ner_pipeline(model_dir: Path):
    if not model_dir.exists():
        return None

    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(str(model_dir), local_files_only=True)
        return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    except Exception as exc:
        print(f"Failed to load local NER model from {model_dir}: {exc}")
        return None


def normalize_model_mention(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = text.strip(" \"'`[]{}()")
    return text


def should_keep_model_entity(mention: str, entity_type: str) -> bool:
    if not mention:
        return False
    if len(mention) < 2 or len(mention) > 80:
        return False
    if re.fullmatch(r"[\W_]+", mention):
        return False
    if mention.lower() in NOISY_MENTIONS:
        return False
    if mention.startswith(("Category:", "File:", "Image:")):
        return False
    if "http" in mention.lower():
        return False
    if re.search(r"\d", mention):
        return False
    if entity_type == "MISC" and not any(char.isupper() for char in mention):
        return False
    return True


def add_entity_record(
    entity_store: dict[tuple[str, str], dict[str, Any]],
    mention: str,
    entity_type: str,
    sentence_id: int,
    sentence: str,
    source: str,
) -> None:
    key = (entity_type, mention.casefold())
    if key not in entity_store:
        entity_store[key] = {
            "name": mention,
            "type": entity_type,
            "mention_count": 0,
            "sentence_ids": [],
            "evidence": [],
            "source": source,
        }

    record = entity_store[key]
    record["mention_count"] += 1
    if sentence_id not in record["sentence_ids"]:
        record["sentence_ids"].append(sentence_id)
    if sentence not in record["evidence"] and len(record["evidence"]) < 5:
        record["evidence"].append(sentence)


def extract_entities_with_model(sentences: list[str], ner_pipeline) -> list[dict[str, Any]]:
    entity_store: dict[tuple[str, str], dict[str, Any]] = {}
    candidate_sentences = [
        (sentence_id, sentence)
        for sentence_id, sentence in enumerate(sentences, start=1)
        if 10 <= len(sentence) <= 400
    ]

    batch_size = 8
    for start in range(0, len(candidate_sentences), batch_size):
        batch = candidate_sentences[start:start + batch_size]
        batch_sentences = [sentence for _, sentence in batch]
        batch_results = ner_pipeline(batch_sentences, batch_size=batch_size)

        for (sentence_id, sentence), predictions in zip(batch, batch_results):
            for prediction in predictions:
                raw_label = prediction.get("entity_group") or prediction.get("entity")
                entity_type = LABEL_MAPPING.get(raw_label)
                if entity_type is None:
                    continue

                mention = normalize_model_mention(prediction.get("word", ""))
                if not should_keep_model_entity(mention, entity_type):
                    continue

                add_entity_record(
                    entity_store,
                    mention=mention,
                    entity_type=entity_type,
                    sentence_id=sentence_id,
                    sentence=sentence,
                    source="transformer_ner",
                )

            for year in re.findall(r"\b(?:18|19|20)\d{2}\b", sentence):
                add_entity_record(
                    entity_store,
                    mention=year,
                    entity_type="TIME",
                    sentence_id=sentence_id,
                    sentence=sentence,
                    source="rule_based_regex",
                )

    extracted = list(entity_store.values())
    return sorted(extracted, key=lambda item: (item["type"], item["name"].casefold()))


def add_relation(
    relations: list[dict[str, str]],
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


def extract_relations(sentences: list[str]) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
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


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, relations: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
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


def choose_entity_corpus_path() -> Path:
    if WIKI_CORPUS_PATH.exists() and WIKI_CORPUS_PATH.stat().st_size > 0:
        return WIKI_CORPUS_PATH
    return RULE_CORPUS_PATH


def load_entity_sentences(corpus_path: Path) -> list[str]:
    text = read_corpus(corpus_path)
    if corpus_path == WIKI_CORPUS_PATH:
        text = clean_wiki_markup(text)
    return split_sentences(text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entity_corpus_path = choose_entity_corpus_path()
    entity_sentences = load_entity_sentences(entity_corpus_path)
    relation_sentences = split_sentences(read_corpus(RULE_CORPUS_PATH))

    ner_pipeline = load_local_ner_pipeline(MODEL_DIR)
    if ner_pipeline is None:
        entity_method = "rule_based_dictionary_fallback"
        entity_corpus_path = RULE_CORPUS_PATH
        entity_sentences = relation_sentences
        entities = extract_entities_rule_based(relation_sentences)
        print(f"Local NER model not available at {MODEL_DIR}.")
        print("Falling back to the original rule-based entity extractor.")
    else:
        entity_method = "transformer_ner"
        entities = extract_entities_with_model(entity_sentences, ner_pipeline)

    relations = extract_relations(relation_sentences)

    write_json(OUTPUT_DIR / "entities.json", entities)
    write_json(OUTPUT_DIR / "relations.json", relations)
    write_csv(OUTPUT_DIR / "triples.csv", relations)

    summary = defaultdict(int)
    for entity in entities:
        summary[entity["type"]] += 1

    print("Knowledge extraction completed.")
    print(f"Entity extraction method: {entity_method}")
    print(f"Entity corpus: {entity_corpus_path}")
    print(f"Entity sentences: {len(entity_sentences)}")
    print(f"Relation sentences: {len(relation_sentences)}")
    print(f"Entities: {len(entities)}")
    print(f"Relations: {len(relations)}")
    print("Entity type summary:")
    for entity_type in sorted(summary):
        print(f"  {entity_type}: {summary[entity_type]}")


if __name__ == "__main__":
    main()
