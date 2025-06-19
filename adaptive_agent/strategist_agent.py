# strategist_agent.py

import openai
import os
import json

openai.api_key = ""

def adapt_method_to_wsi(method_text, slide_level=True, feature_based=True):
    """
    Sends the raw method suggestion from the researcher agent to GPT
    and receives an adapted, coder-ready implementation spec that targets main_template.py only.
    """
    template_summary = (
        "The current script, `main_template.py`, implements a full training pipeline for weakly supervised WSI classification "
        "using Multiple Instance Learning (MIL) with CLAM variants (`clam_sb`, `clam_mb`) or simpler MIL MLP architectures. "
        "The training loop, validation, model construction, and data handling are all defined in this single script. "
        "It processes slides as bags of patch-level features and uses slide-level binary or multiclass labels. "
        "It imports utility classes like `EarlyStopping`, `Accuracy_Logger`, and `print_network` from `core_utils`, "
        "but does not rely on external training logic. All learning logic is local and self-contained."
    )
    context = (
        "You are a strategist responsible for adapting machine learning methods to weakly supervised classification "
        "of Whole Slide Images (WSIs) in digital pathology. Each slide is represented as a bag of features "
        "(e.g., extracted by ResNet50), with only slide-level labels. CLAM is currently used as the baseline architecture.\n\n"

        "Your goal is to describe **how to integrate the proposed method into a single Python script, `main_template.py`.**\n\n"
        "⚠️ DO NOT suggest creating additional scripts. The code-writing agent will overwrite `main_template.py`, so your spec "
        "must describe changes entirely within that file.\n\n"

        f"{template_summary}\n\n"

        f"Here is the proposed method:\n\n{method_text}\n\n"

        "Adapt this method to this WSI context. Your output must include:\n"
        "- A precise implementation strategy for patch-level MIL with slide-level labels\n"
        "- What specific parts of `main_template.py` should be modified, added, or removed\n"
        "- Any architectural changes (e.g., encoder, losses, attention, auxiliary heads)\n"
        "- Any required import or dependency changes\n"
        "- All assumptions (e.g., feature format, patch size, label types)\n"
        "- Any risks or likely failure modes when using this method in this context\n\n"
        "Make sure your spec is clear, complete, and ready to be passed to a coding agent that will write the final script."
    )

    response = openai.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": "You are an ML strategist adapting research to WSI classification using MIL."},
            {"role": "user", "content": context}
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content


def score_idea(idea_text):
    """Score an idea for novelty, potential impact and ease of implementation.

    The function asks GPT to rate the idea on three criteria from 0–10 and
    returns a dictionary with the individual scores and their sum.
    """

    prompt = (
    "You are an expert research strategist evaluating proposed machine learning methods for weakly supervised WSI classification.\n"
    "For each idea, assess it in the following categories:\n"
    "- 'novelty': Has this idea (or close variations) been applied to WSI or MIL problems before? Give higher scores for approaches new to this domain. (0-10)\n"
    "- 'potential': If applied to WSI classification, how likely is this method to meaningfully improve generalisability across institutions, scanners, or cohorts? Consider robustness and empirical plausibility. (0-10)\n"
    "- 'feasibility': Could this method be realistically implemented in a PyTorch-based CLAM pipeline with modest effort? Consider compatibility with attention mechanisms, available inputs (features, labels), and code modularity. (0-10)\n"
    "You may internally reason, but do not include any explanations in your output.\n"
    "Return a JSON object with integer fields: 'novelty', 'potential', 'feasibility', and a text field 'commentary' (with any comments about your scores, keep it succinct).\n"
    "Respond ONLY with the JSON.\n\n"
    f"Idea:\n{idea_text}"
    )

    response = openai.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": "You evaluate research ideas."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    try:
        scores = json.loads(response.choices[0].message.content)
        scores = {
            "novelty": int(scores.get("novelty", 0)),
            "potential": int(scores.get("potential", 0)),
            "feasibility": int(scores.get("feasibility", 0)),
            "commentary": scores.get("commentary", "no comment")
        }
        print(f"Novelty: {scores['novelty']}\n")
        print(f"Potential: {scores['potential']}\n")
        print(f"Feasibility: {scores['feasibility']}\n")
        print(f"Comments: {scores['commentary']}\n")
    except Exception:
        scores = {"novelty": 0, "potential": 0, "feasibility": 0, "commentary": "error"}

    scores["total"] = scores["novelty"] + scores["potential"] + scores["feasibility"]
    return scores


if __name__ == "__main__":
    test_idea = (
        "Method: Entropy-Regularized Consistency Training\n"
        "Justification: This method encourages predictions to be both consistent under augmentations and low in entropy, "
        "thus improving domain robustness. Originally used in semi-supervised learning.\n"
        "Source: CVPR 2023, arXiv:2303.12345"
    )

    spec = adapt_method_to_wsi(test_idea)
    print("\nAdapted WSI Method Spec:\n")
    print(spec)
