import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from src.intent_detector import IntentDetector

class ContextualModerator:
    def __init__(self, toxicity_model_path="./fine_tuned_llm"):
        self.device = "cpu"
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Base Toxicity Model
        print("Loading toxicity model...")
        try:
            self.tox_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_path)
            self.tox_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_path)
            self.tox_model.eval()
            self.has_custom_llm = True
            print("Loaded fine-tuned model.")
        except Exception as e:
            print(f"Model not found: {e}. Using fallback mode.")
            self.has_custom_llm = False

        # Intent Detector
        self.intent_detector = IntentDetector()

        # Detoxification Model (loaded when needed)
        self.detox_tokenizer = None
        self.detox_model = None
        print("Detox model will load on first use.")

        # Simple regex replacements
        self.regex_map = {
            r'\bfuc?k(?:ing|in)?\b': 'extremely',
            r'\bshit(?:ty)?\b': 'bad',
            r'\bdamn\b': 'really',
            r'\bass\b': 'very',
            r'\bbitch(?:in|ing)?\b': 'complaining',
            r'\bcrap(?:py)?\b': 'poor',
            r'\bhell\b': 'heck',
        }
        print("Moderation system ready.")

    def predict_toxicity(self, text):
        """Run text through DistilBERT to get toxicity scores."""
        if not self.has_custom_llm:
            return {lbl: 0.05 for lbl in self.labels}

        inputs = self.tox_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.tox_model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        return {lbl: float(probs[i]) for i, lbl in enumerate(self.labels)}

    def paraphrase_text(self, text, intent_info=None):
        """Detoxify using regex, then T5 model if needed."""
        paraphrased = text
        for pattern, replacement in self.regex_map.items():
            paraphrased = re.sub(pattern, replacement, paraphrased, flags=re.IGNORECASE)

        if paraphrased.lower() != text.lower():
            return paraphrased, "Regex Replace"

        if self.detox_model is None:
            print("Loading T5 detox model...")
            self.detox_tokenizer = AutoTokenizer.from_pretrained("s-nlp/t5-paranmt-detox")
            self.detox_model = AutoModelForSeq2SeqLM.from_pretrained("s-nlp/t5-paranmt-detox")
            self.detox_model.eval()

        inputs = self.detox_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            output_ids = self.detox_model.generate(**inputs, max_new_tokens=64)
        result = self.detox_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return result, "T5 Detox"

    def analyze_and_moderate(self, text):
        """Combines scoring, intent detection, and detox."""
        scores = self.predict_toxicity(text)
        intent_data = self.intent_detector.detect_intent(text)
        intent = intent_data["intent"]

        has_threat_hate = scores.get('threat', 0) > 0.15 or scores.get('identity_hate', 0) > 0.18
        insult_severe = max(scores.get('insult', 0), scores.get('severe_toxic', 0))
        general_toxicity = max(scores.get('toxic', 0), scores.get('obscene', 0))

        # Personal attack or hate -> Block
        if intent == "Hostile_Attack" or has_threat_hate or insult_severe > 0.30:
            verdict = "Blocked"
            explanation = f"Detected Intent: {intent}. Contains harassment or hate speech."
            paraphrase = "Blocked due to personal attack."

        # Critique/Venting with profanity -> Detoxify
        elif (intent in ["Critique_Feedback", "Emotional_Venting"]) and general_toxicity > 0.25:
            verdict = "Detoxified (Maintained Intent)"
            paraphrase, engine = self.paraphrase_text(text, intent_info=intent_data)
            explanation = f"Detected Intent: {intent}. Cleaned profanity while preserving feedback."

        # General toxicity -> Detoxify
        elif general_toxicity > 0.35:
            verdict = "Detoxified"
            paraphrase, engine = self.paraphrase_text(text, intent_info=intent_data)
            explanation = f"Detected Intent: {intent}. Cleaned profanity."

        # Clean
        else:
            verdict = "Clean"
            explanation = f"No major toxicity found. Intent: {intent}."
            paraphrase = text

        return {
            "scores": scores,
            "intent": intent_data,
            "verdict": verdict,
            "explanation": explanation,
            "paraphrase": paraphrase
        }
