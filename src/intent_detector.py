import os
import re
from transformers import pipeline

class IntentDetector:
    """
    Detects the communicative intent behind a user's comment.
    Helps the moderation layer decide whether to block or paraphrase.
    """
    def __init__(self):
        self.classifier = None
        self.candidate_labels = [
            "Critique_Feedback",
            "Emotional_Venting",
            "Hostile_Attack",
            "General_Discussion"
        ]
        
        # Simple regex for fast intent detection
        self.hostile_patterns = [
            r'\byou (are|r)\b.*?\b(idiot|stupid|dumb|fool|bitch|bastard|loser|jerk|ugly|trash)\b',
            r'\b(fuck|screw) (you|u)\b',
            r'\bgo (die|kill yourself)\b',
            r'\bshut (up|the fuck up)\b'
        ]
        
        self.critique_patterns = [
            r'\b(app|site|page|website|system|button|feature|bug|issue|login|error|crash|broken|fix|slow)\b',
            r'\b(not working|doesn\'t work|cannot|can\'t|fails to|impossible to)\b'
        ]

    def detect_intent_heuristic(self, text):
        """Regex-based fallback classification."""
        text_lower = text.lower()
        
        for pattern in self.hostile_patterns:
            if re.search(pattern, text_lower):
                return "Hostile_Attack", 0.90
                
        for pattern in self.critique_patterns:
            if re.search(pattern, text_lower):
                return "Critique_Feedback", 0.85
                
        return None, 0.0

    def detect_intent(self, text):
        """
        Uses a 3-stage cascade:
        1. Fast regex
        2. Fine-tuned DistilBERT (if available)
        3. Zero-shot MNLI fallback
        """
        FINE_TUNED_PATH = "./fine_tuned_intent_model"

        # 1: Regex
        heuristic_intent, confidence = self.detect_intent_heuristic(text)
        if heuristic_intent:
            return {
                "intent": heuristic_intent,
                "confidence": confidence,
                "method": "Heuristic Rules"
            }

        # 2: Fine-tuned model
        if os.path.isdir(FINE_TUNED_PATH):
            if self.classifier is None:
                print(f"Loading intent classifier from {FINE_TUNED_PATH}")
                try:
                    self.classifier = pipeline(
                        "text-classification",
                        model=FINE_TUNED_PATH,
                        tokenizer=FINE_TUNED_PATH,
                        device=-1
                    )
                except Exception as e:
                    print(f"Failed to load fine-tuned model: {e}")
                    self.classifier = "FAILED"

            if self.classifier and self.classifier != "FAILED":
                try:
                    result = self.classifier(text, truncation=True, max_length=128)[0]
                    return {
                        "intent": result["label"],
                        "confidence": float(result["score"]),
                        "method": "Fine-Tuned DistilBERT"
                    }
                except Exception as e:
                    print(f"Inference error: {e}")

        # 3: Zero-shot fallback
        if self.classifier is None or self.classifier == "FAILED":
            print("Loading zero-shot intent classifier...")
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="valhalla/distilbart-mnli-12-3"
                )
            except Exception as e:
                print(f"Failed to load zero-shot model: {e}")
                return {
                    "intent": "General_Discussion",
                    "confidence": 0.5,
                    "method": "Fallback Default"
                }

        hypothesis_template = "The intent of this comment is {}."
        label_mapping = {
            "constructive critique or bug report": "Critique_Feedback",
            "emotional frustration or venting": "Emotional_Venting",
            "personal hostile attack or harassment": "Hostile_Attack",
            "general discussion or opinion": "General_Discussion"
        }

        res = self.classifier(
            text,
            list(label_mapping.keys()),
            hypothesis_template=hypothesis_template
        )

        top_label = res["labels"][0]
        top_score = res["scores"][0]
        mapped_intent = label_mapping.get(top_label, "General_Discussion")

        return {
            "intent": mapped_intent,
            "confidence": float(top_score),
            "method": "Zero-Shot MNLI"
        }
