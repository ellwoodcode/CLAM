# orchestrator.py

import json
import os
from datetime import datetime
from researcher_agent import fetch_recent_arxiv_papers, summarize_with_feedback, load_feedback, load_used_ids
from strategist_agent import adapt_method_to_wsi, score_idea
from coder_agent import try_implement_method
import re

FEEDBACK_LOG = "feedback_log.json"
REPORT_PATH = "experiment_log.md"
VETTED_IDEAS_PATH = "vetted_ideas.json"
USED_ARXIV_PATH = "used_arxiv_ids.json"

def append_feedback(method_id, title, source, status, reason=None, auc=None):
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, "r") as f:
            feedback = json.load(f)
    else:
        feedback = {}

    feedback[method_id] = {
        "title": title,
        "source": source,
        "status": status,
        "reason": reason,
        "auc": auc,
        "timestamp": datetime.now().isoformat()
    }

    with open(FEEDBACK_LOG, "w") as f:
        json.dump(feedback, f, indent=2)

def append_report(section):
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{section}\n")


def load_vetted_ideas():
    """Return a list of previously vetted ideas."""
    if os.path.exists(VETTED_IDEAS_PATH):
        with open(VETTED_IDEAS_PATH, "r") as f:
            return json.load(f)
    return []

def save_vetted_ideas(ideas):
    """Persist the list of vetted ideas to disk."""
    with open(VETTED_IDEAS_PATH, "w") as f:
        json.dump(ideas, f, indent=2)

def extract_wrapped_ideas(summary):
    return re.findall(r"\[BEGIN_IDEA\](.*?)\[/END_IDEA\]", summary, re.DOTALL)

if __name__ == "__main__":
    print("Loading feedback...")
    feedback = load_feedback()
    vetted = load_vetted_ideas()
    used_ids = load_used_ids()

    while len(vetted) < 5:
        print("Fetching recent papers...")
        papers = fetch_recent_arxiv_papers(used_ids=used_ids)
        summary = summarize_with_feedback(papers, feedback)

        print("Researcher Suggestions:\n")
        print(summary)

        for paper in papers:
            used_ids.add(paper["arxiv_id"])

        ideas = extract_wrapped_ideas(summary)

        for idea in ideas:
            print("Scoring idea...")
            scores = score_idea(idea)
            if scores.get("total", 0) < 21:
                print(f"   Discarded (score {scores['total']})")
                continue
            vetted.append({"idea": idea, "scores": scores})
            save_vetted_ideas(vetted)
            print(f"   Accepted (score {scores['total']})")

            if len(vetted) >= 5:
                break

    for i, item in enumerate(vetted):
        method_id = f"method_{datetime.now().strftime('%Y%m%d')}_{i+1}"
        idea_text = item["idea"]

        print(f"\nAdapting Idea {i+1} to WSI context...")
        adaptation_spec = adapt_method_to_wsi(idea_text)

        print("Passing to coder agent...")
        result = try_implement_method({
            "id": method_id,
            "method": idea_text,
            "justification": adaptation_spec
        })

        report_section = (
            f"### {method_id}: {idea_text.splitlines()[0]}\n"
            f"**Adaptation Summary**:\n```\n{adaptation_spec}\n```\n"
            f"**Coder Result**:\n- Success: {result['success']}\n"
            f"- AUC: {result['auc']:.4f}\n"
            f"- Cohort AUCs: {result['cohort_aucs']}\n"
            f"- Script: {result['script_path']}\n"
            f"- Log: {result['log_path']}\n"
            f"- Notes:\n```\n{result['notes']}\n```"
        )

        append_report(report_section)

        if result["success"] and result["auc"] > 0.5:  
            append_feedback(method_id, idea_text.splitlines()[0], "arXiv", "success", auc=result["auc"])
            print("New method improved generalisability! Stopping here.")
            break
        else:
            append_feedback(method_id, idea_text.splitlines()[0], "arXiv", "failed", reason="Did not outperform baseline", auc=result["auc"])
            print("No improvement, trying next...")
