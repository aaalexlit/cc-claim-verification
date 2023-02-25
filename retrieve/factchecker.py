import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from check_article import get_evidences_from_text

climate_factcheck_tokenizer = AutoTokenizer.from_pretrained("amandakonet/climatebert-fact-checking")
climate_factcheck_model = AutoModelForSequenceClassification.from_pretrained("amandakonet/climatebert-fact-checking")

climate_factcheck_model.config.id2label = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT_ENOUGH_INFO"
}


def predict_supports_or_refutes(claim_evidence_array):
    def claim_evidence_pair_data():
        for claim, evidences in claim_evidence_array:
            for evidence in evidences:
                yield {"text": claim, "text_pair": evidence}

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=climate_factcheck_model,
                    tokenizer=climate_factcheck_tokenizer, device=device)
    labels = []
    probs = []
    for out in pipe(claim_evidence_pair_data(), batch_size=64):
        labels.append(out['label'])
        probs.append(out['score'])
    return labels, probs


def convert_claim_evidence_to_text(claim_evidence_array):
    res = []
    for claim, evidences in claim_evidence_array:
        evidences_text = [evidence.content for evidence in evidences]
        res.append([claim, evidences_text])
    return res


def is_supported_by_science(text, print_neutral=False, threshold=0.5):
    claim_evidence_array = get_evidences_from_text(text)
    claim_evidence_array = convert_claim_evidence_to_text(claim_evidence_array)
    claim_evidence_pairs = list(
        zip(*[(claim, evidence) for claim, evidences in claim_evidence_array for evidence in evidences]))
    labels, probs = predict_supports_or_refutes(claim_evidence_array)
    for claim, evidence, label, prob in zip(*claim_evidence_pairs, labels, probs):
        if prob > threshold:
            if (label == 'NOT_ENOUGH_INFO' and print_neutral) or label != 'NOT_ENOUGH_INFO':
                print(f'Label:\n{label}\nClaim:\n{claim}\nEvidence:\n{evidence}\nProb:\n{prob:.2f}\n')
