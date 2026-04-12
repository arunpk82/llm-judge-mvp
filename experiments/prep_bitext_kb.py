"""
Step 1: Build RAG Knowledge Base from Bitext Customer Support Dataset
=====================================================================

Downloads the Bitext dataset from HuggingFace and builds an intent-indexed
knowledge base (JSON) for the Science Gate experiment.

Usage:
    pip install datasets
    python experiments/prep_bitext_kb.py

Output:
    experiments/bitext_knowledge_base.json
"""

import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset  # type: ignore[attr-defined]  # type: ignore[attr-defined]


def main():
    print("Downloading Bitext Customer Support dataset...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    data = ds["train"]

    print(f"Loaded {len(data)} records")
    print(f"Columns: {data.column_names}")

    # Show a sample to understand structure
    sample = data[0]
    print("\nSample record:")
    for k, v in sample.items():
        print(f"  {k}: {str(v)[:120]}")

    # Group by intent — collect all responses as "documentation"
    intent_docs = defaultdict(list)
    intent_queries = defaultdict(list)

    for row in data:
        # Bitext columns: instruction, intent, response, category, etc.
        intent = row.get("intent", "unknown")
        response = row.get("response", "")
        instruction = row.get("instruction", "")

        if response.strip():
            intent_docs[intent].append(response.strip())
        if instruction.strip():
            intent_queries[intent].append(instruction.strip())

    print(f"\nFound {len(intent_docs)} unique intents")
    print(f"Intents: {sorted(intent_docs.keys())}")

    # Build knowledge base: for each intent, concatenate unique responses
    # as "support documentation" (deduplicated, limited to keep reasonable size)
    knowledge_base = {}
    for intent, responses in intent_docs.items():
        # Deduplicate responses
        unique_responses = list(dict.fromkeys(responses))
        # Take up to 10 representative responses as the "knowledge base"
        # for this intent — these represent what the support docs would contain
        selected = unique_responses[:10]
        doc = "\n\n".join(selected)

        knowledge_base[intent] = {
            "intent": intent,
            "category": "",  # filled below
            "total_examples": len(responses),
            "unique_examples": len(unique_responses),
            "selected_count": len(selected),
            "documentation": doc,
            "sample_queries": list(dict.fromkeys(intent_queries.get(intent, [])))[:5],
        }

    # Fill in categories from first matching record
    for row in data:
        intent = row.get("intent", "unknown")
        cat = row.get("category", "")
        if intent in knowledge_base and not knowledge_base[intent]["category"]:
            knowledge_base[intent]["category"] = cat

    # Save
    out_path = Path(__file__).parent / "bitext_knowledge_base.json"
    with open(out_path, "w") as f:
        json.dump(knowledge_base, f, indent=2)

    print(f"\nKnowledge base saved to {out_path}")
    print(f"  {len(knowledge_base)} intents indexed")
    total_docs = sum(v["selected_count"] for v in knowledge_base.values())
    print(f"  {total_docs} total document entries")

    # Show intent list for mapping
    print("\nAvailable intents for mapping:")
    for intent in sorted(knowledge_base.keys()):
        info = knowledge_base[intent]
        print(
            f"  {intent:<40} ({info['total_examples']:>4} examples, cat: {info['category']})"
        )


if __name__ == "__main__":
    main()
