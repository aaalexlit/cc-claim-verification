import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
import torch

pd.options.mode.chained_assignment = None

claimbuster_tokenizer = AutoTokenizer.from_pretrained("lucafrost/ClaimBuster-DeBERTaV2")
claimbuster_model = AutoModelForSequenceClassification.from_pretrained("lucafrost/ClaimBuster-DeBERTaV2")


# env_climatebert_tokenizer = AutoTokenizer.from_pretrained("climatebert/environmental-claims")
# env_climatebert_model = AutoModelForSequenceClassification.from_pretrained("climatebert/environmental-claims")


def get_claims_from_text(text, threshold=0.7, debug=False):
    sentences = tokenize_texts_to_sentences(text)
    predicted_class_ids, probs = is_claim(sentences, debug=debug)
    return [sentence for sentence, label, prob in zip(sentences, predicted_class_ids, probs) if
            label in [1, 2] and prob > threshold]


# TODO: needs performance improvement
def get_claims_from_texts(df, id_col='id', text_col='abstract', threshold=0.7, debug=False):
    df[text_col] = df[text_col].map(tokenize_texts_to_sentences)
    df = df.explode(text_col)
    sentences = df[text_col].tolist()
    ids = df[id_col].tolist()
    predicted_class_ids, probs = is_claim(sentences, debug=debug)
    claims_df = pd.DataFrame(
        [(doi, sentence) for doi, sentence, label, prob in zip(ids, sentences, predicted_class_ids, probs) if
         label in [1, 2] and prob > threshold], columns=[id_col, 'claims'])
    claims_df = claims_df.groupby(id_col).agg({'claims': lambda x: x.tolist()})
    return claims_df


def is_claim(sentences, model=claimbuster_model, tokenizer=claimbuster_tokenizer, debug=False):
    inputs = tokenizer(sentences,
                       padding=True,
                       truncation=True,
                       max_length=512,
                       return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.softmax(logits, dim=1).max(dim=1)
    predicted_class_ids = preds.indices.numpy()
    probs = preds.values.numpy()
    if debug:
        predicted_labels = [model.config.id2label[class_id] for class_id in predicted_class_ids]
        for sentence, label, prob in zip(sentences, predicted_labels, probs):
            print(f"{label}({prob:.3f})")
            print(sentence)
    return predicted_class_ids, probs


def tokenize_texts_to_sentences(text):
    return sent_tokenize(text)

# print(get_claims_from_text(
#     """Increased aridity and human population have reduced tree cover in parts of the African Sahel degraded resources for local people. Yet, trends relative importance climate remain unresolved. From field measurements, aerial photos, Ikonos satellite images, we detected significant 1954-2002 density declines western 18 +/- 14% (P = 0.014, n 204) 17 13% 0.0009, 187). observations, a 1960-2000 species richness decline 21 11% 0.0028, 14) across southward shift Sahel, Sudan, Guinea zones. Multivariate analyses climate, soil, showed that temperature most significantly < 0.001) explained changes. bivariate tests observations indicated dominance precipitation, supporting attribution changes to variability. Climate change forcing variability, particularly 0.05) 1901-2002 increases precipitation decreases research areas, connects global change. This suggests roles action adaptation address ecological Sahel."""))
