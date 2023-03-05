import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize import sent_tokenize
import torch

pd.options.mode.chained_assignment = None

claimbuster_tokenizer = AutoTokenizer.from_pretrained("lucafrost/ClaimBuster-DeBERTaV2")
claimbuster_model = AutoModelForSequenceClassification.from_pretrained("lucafrost/ClaimBuster-DeBERTaV2")


# env_climatebert_tokenizer = AutoTokenizer.from_pretrained("climatebert/environmental-claims")
# env_climatebert_model = AutoModelForSequenceClassification.from_pretrained("climatebert/environmental-claims")


def get_claims_from_text(text, threshold=0.7, debug=False):
    sentences = sent_tokenize(text)
    predicted_class_ids, probs = is_claim(sentences, debug=debug)
    return [sentence for sentence, label, prob in zip(sentences, predicted_class_ids, probs) if
            label in [1, 2] and prob > threshold]


# TODO: potentially needs performance improvement
def get_claims_from_texts(df, id_col='id', text_col='abstract', threshold=0.7, debug=False):
    df[text_col] = df[text_col].map(sent_tokenize)
    df = df.explode(text_col)
    sentences = df[text_col].tolist()
    ids = df[id_col].tolist()
    predicted_class_ids, probs = is_claim(sentences, debug=debug)
    claims_df = pd.DataFrame(
        [(doi, sentence) for doi, sentence, label, prob in zip(ids, sentences, predicted_class_ids, probs) if
         label in [1, 2] and prob > threshold], columns=[id_col, 'claims'])
    claims_df = claims_df.groupby(id_col).agg({'claims': lambda x: x.tolist()})
    return claims_df


def is_claim(sentences,
             model=claimbuster_model,
             tokenizer=claimbuster_tokenizer,
             debug=False):
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model,
                    tokenizer=tokenizer, device=device)
    labels, probs = [], []
    for out in pipe(sentences, batch_size=1):
        labels.append(out['label'])
        probs.append(out['score'])
    if debug:
        for sentence, label, prob in zip(sentences, labels, probs):
            print(f"{label}({prob:.3f})")
            print(sentence)
    return list(map(lambda l: model.config.label2id[l], labels)), probs


if __name__ == "__main__":
    print(get_claims_from_text(
        """Increased aridity and human population have reduced tree cover in parts of the African Sahel degraded resources for local people. Yet, trends relative importance climate remain unresolved. From field measurements, aerial photos, Ikonos satellite images, we detected significant 1954-2002 density declines western 18 +/- 14% (P = 0.014, n 204) 17 13% 0.0009, 187). observations, a 1960-2000 species richness decline 21 11% 0.0028, 14) across southward shift Sahel, Sudan, Guinea zones. Multivariate analyses climate, soil, showed that temperature most significantly < 0.001) explained changes. bivariate tests observations indicated dominance precipitation, supporting attribution changes to variability. Climate change forcing variability, particularly 0.05) 1901-2002 increases precipitation decreases research areas, connects global change. This suggests roles action adaptation address ecological Sahel."""))
