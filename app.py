import gradio as gr
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.moderation_layer import ContextualModerator

print("Starting Moderation System...")
moderator = ContextualModerator(toxicity_model_path="./fine_tuned_llm")
print("UI ready.")

def process_comment(comment):
    if not comment.strip():
        return {}, "-", "-", "Please enter text.", ""

    result = moderator.analyze_and_moderate(comment)

    # Format scores for UI
    label_scores = {k: round(v, 4) for k, v in result["scores"].items()}
    intent_str = f"{result['intent']['intent']} ({result['intent']['method']})"

    return (
        label_scores,
        intent_str,
        result["verdict"],
        result["explanation"],
        result["paraphrase"]
    )

demo = gr.Interface(
    fn=process_comment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Type a comment to test...",
        label="Input Comment"
    ),
    outputs=[
        gr.Label(label="Toxicity Probabilities"),
        gr.Textbox(label="Detected Intent"),
        gr.Textbox(label="Decision"),
        gr.Textbox(label="Explanation"),
        gr.Textbox(label="Cleaned Paraphrase"),
    ],
    title="Toxic Comment Moderation System",
    description=(
        "Stage 1: DistilBERT scores toxicity across 6 categories.\n"
        "Stage 2: Intent Detection checks for constructive vs hostile intent.\n"
        "Stage 3: Cleans profanity if the intent is not a personal attack."
    ),
    examples=[
        ["This login button is completely broken and slow!"],
        ["You are a total idiot and a complete loser, go away!"],
        ["This food was so good!"],
        ["I am so annoyed by this weather today."],
        ["This project is helpful!"],
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)
