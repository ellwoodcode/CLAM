# strategist_agent.py

import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def adapt_method_to_wsi(method_text, slide_level=True, feature_based=True):
    """
    Sends the raw method suggestion from the researcher agent to GPT
    and receives an adapted, coder-ready implementation spec.
    """
    context = (
        "You are a strategist responsible for adapting machine learning methods to weakly supervised classification "
        "of Whole Slide Images (WSIs) in digital pathology. Each slide is represented as a bag of features (e.g., ResNet50), "
        "with only slide-level binary labels (tumor vs. normal). CLAM is currently used as the baseline architecture.\n\n"
        f"Here is a proposed method:\n\n{method_text}\n\n"
        "Adapt this method to this WSI context. Your output should include:\n"
        "- A precise implementation strategy suitable for patch-level MIL with slide-level labels\n"
        "- Any architectural modifications (e.g., auxiliary heads, new losses)\n"
        "- Assumptions or requirements (e.g., patch features, attention maps)\n"
        "- What files would be needed or modified (e.g., training script, dataset class, loss function)\n"
        "- Any specific risks or expected failure points in this context\n"
        "Make sure your spec is useful for a coding agent to implement in PyTorch."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an ML strategist adapting research to WSI classification using MIL."},
            {"role": "user", "content": context}
        ],
        temperature=0.5,
    )

    return response.choices[0].message['content']

if __name__ == "__main__":
    test_idea = (
        "Method: Entropy-Regularized Consistency Training\n"
        "Justification: This method encourages predictions to be both consistent under augmentations and low in entropy, "
        "thus improving domain robustness. Originally used in semi-supervised learning.\n"
        "Source: CVPR 2023, arXiv:2303.12345"
    )

    spec = adapt_method_to_wsi(test_idea)
    print("\n🧠 Adapted WSI Method Spec:\n")
    print(spec)
