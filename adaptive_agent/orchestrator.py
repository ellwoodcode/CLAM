# orchestrator.py

import json
import os
from datetime import datetime
from researcher_agent import fetch_recent_arxiv_papers, summarize_with_feedback, load_feedback, save_cache
from strategist_agent import adapt_method_to_wsi
from coder_agent import try_implement_method

FEEDBACK_LOG = "feedback_log.json"
REPORT_PATH = "experiment_log.md"

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
    with open(REPORT_PATH, "a") as f:
        f.write(f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{section}\n")

if __name__ == "__main__":
    print("🔁 Loading feedback...")
    feedback = load_feedback()

    print("🔍 Fetching recent papers...")
    papers = fetch_recent_arxiv_papers()
    summary = summarize_with_feedback(papers, feedback)

    print("📚 Researcher Suggestions:\n")
    print(summary)

    ideas = summary.split("\n\n")
    for i, idea in enumerate(ideas):
        method_id = f"method_{datetime.now().strftime('%Y%m%d')}_{i+1}"
        if not idea.strip():
            continue

        print(f"\n🧠 Adapting Idea {i+1} to WSI context...")
        adaptation_spec = adapt_method_to_wsi(idea)

        print("👨‍💻 Passing to coder agent...")
        result = try_implement_method({
            "id": method_id,
            "method": idea,
            "justification": adaptation_spec
        })

        report_section = (
            f"### {method_id}: {idea.splitlines()[0]}\n"
            f"**Adaptation Summary**:\n```\n{adaptation_spec}\n```\n"
            f"**Coder Result**:\n- Success: {result['success']}\n"
            f"- AUC: {result['auc']:.4f}\n"
            f"- Cohort AUCs: {result['cohort_aucs']}\n"
            f"- Notes:\n```\n{result['notes']}\n```"
        )

        append_report(report_section)

        if result["success"] and result["auc"] > 0.79:  # Example threshold
            append_feedback(method_id, idea.splitlines()[0], "arXiv", "success", auc=result["auc"])
            print("🎉 New method improved generalisability! Stopping here.")
            break
        else:
            append_feedback(method_id, idea.splitlines()[0], "arXiv", "failed", reason="Did not outperform baseline", auc=result["auc"])
            print("❌ No improvement, trying next...")
