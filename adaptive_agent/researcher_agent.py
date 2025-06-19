# researcher_agent.py (adaptive with arXiv integration)

import arxiv
import openai
import os
import json
from datetime import datetime, timedelta
import time

openai.api_key = ""

FEEDBACK_LOG = "./feedback_log.json"
CACHE_PATH = "./research_cache.json"
USED_IDEA_PATH = "used_idea_hashes.json"
USED_ARXIV_PATH = "used_arxiv_ids.json"
MAX_RESULTS = 10
SEARCH_QUERY = "domain adaptation OR representation learning OR generalisation"
CATEGORIES = ["cs.LG", "cs.CV", "eess.IV", "q-bio.QM"]


def load_feedback():
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, "r") as f:
            return json.load(f)
    return {}

def load_used_ids(path=USED_ARXIV_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f))
    return set()

def save_used_ids(used_ids, path=USED_ARXIV_PATH):
    with open(path, "w") as f:
        json.dump(list(used_ids), f, indent=2)

def load_used_idea_hashes(path=USED_IDEA_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f))
    return set()

def save_used_idea_hashes(hashes, path=USED_IDEA_PATH):
    with open(path, "w") as f:
        json.dump(list(hashes), f, indent=2)

def fetch_recent_arxiv_papers(max_results=MAX_RESULTS, start_year=2023, used_ids=None, retries=3, delay=5):
    for attempt in range(retries):
        try:
            search = arxiv.Search(
                query=SEARCH_QUERY,
                max_results=max_results*2,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            papers = []
            used_ids = set(used_ids or [])
            for result in search.results():
                if result.get_short_id() in used_ids:
                    continue
                if result.published.year >= start_year and any(cat in result.categories for cat in CATEGORIES):
                    papers.append({
                        "title": result.title,
                        "abstract": result.summary,
                        "link": result.entry_id,
                        "arxiv_id": result.get_short_id(),
                        "published": result.published.isoformat()
                    })
            return papers
        except Exception as e:
            print(f"arXiv fetch failed (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return []

def summarize_with_feedback(papers, feedback):
    # Extract reasons from failed attempts
    fail_reasons = [
        f"{m['title']}: {m['reason']}" for m in feedback.values() if m.get("status") == "failed"
    ]
    feedback_summary = "\n".join(fail_reasons) or "None."

    text_input = "Here is a summary of methods that have failed previously:\n" + feedback_summary
    text_input += "\n\nHere are the abstracts of recent papers:\n"
    for i, p in enumerate(papers):
        text_input += f"\nPaper {i+1}:\nTitle: {p['title']}\nAbstract: {p['abstract']}\n"

    text_input += (
        "\nBased on past failures and these papers, suggest 3 implementable methods that improve generalisation "
        "in weakly supervised learning or WSI models. Avoid methods similar to failed ones. For each method, return it wrapped "
        "in [BEGIN_IDEA] and [/END_IDEA] tags, and include:\n"
        "- A title\n- A justification\n- Which paper inspired it\n"
    )

    response = openai.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": "You are a research analyst specializing in ML generalisability."},
            {"role": "user", "content": text_input}
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content

def save_cache(papers):
    with open(CACHE_PATH, "w") as f:
        json.dump(papers, f, indent=2)

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return []

if __name__ == "__main__":
    print("Loading feedback log...")
    feedback = load_feedback()

    print("Fetching recent arXiv papers...")
    papers = fetch_recent_arxiv_papers()

    print("Summarizing papers with respect to past failures...")
    summary = summarize_with_feedback(papers, feedback)

    print("Summary generated:\n")
    print(summary)

    print("Caching papers...")
    save_cache(papers)
