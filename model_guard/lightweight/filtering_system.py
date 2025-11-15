import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load classifier
model = None 
tokenizer = None
# model = AutoModelForSequenceClassification.from_pretrained("madhurjindal/Jailbreak-Detector", use_auth_token=True)
# tokenizer = AutoTokenizer.from_pretrained("madhurjindal/Jailbreak-Detector", use_auth_token=True)

threshold_value = 0.85  # Set your desired threshold here
SAFETY_THRESHOLD = threshold_value

def _load_model():
    global model, tokenizer
    if model is None:
        print("Loading model... this may take a moment on first run")
        try:
            model = AutoModelForSequenceClassification.from_pretrained("madhurjindal/Jailbreak-Detector")
            tokenizer = AutoTokenizer.from_pretrained("madhurjindal/Jailbreak-Detector")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

def initial_screening(user_prompt):
    _load_model()
    inputs = tokenizer(user_prompt, return_tensors="pt")
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    predicted_index = torch.argmax(probs, dim=1).item()
    predicted_prob = probs[0][predicted_index].item()
    labels = model.config.id2label
    predicted_label = labels[predicted_index]
    
    if predicted_prob < SAFETY_THRESHOLD:
        return {"safe": False, "label": predicted_label, "score": predicted_prob}
    else:
        return {"safe": True, "prompt": user_prompt}

def process_prompt(user_prompt):
    result = initial_screening(user_prompt)
    if result["safe"]:
        return {"status": "pass_to_main_llm", "prompt": user_prompt}
    else:
        return {"status": "route_to_secondary_llm", "confidence": result["score"]}