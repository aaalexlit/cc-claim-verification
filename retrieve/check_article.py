from collections import defaultdict

import jsonlines
from nltk import sent_tokenize

import index_in_faiss_by_abstract
import index_in_milvus_by_abstract
from claim.claim_identifier import get_claims_from_text
from index.utils import get_abstracts_matching_claims
from utils import get_text_from_url
from index.utils import filter_climate_related

faiss_indexer = index_in_faiss_by_abstract.get_indexer()


def get_evidences_from_text(text, debug=False):
    if text.startswith('http'):
        text = get_text_from_url(text)
    if not filter_climate_related([text]):
        print("The text is not climate related")
        return []
    claims_from_text = get_claims_from_text(text)
    res = []
    for claim in claims_from_text:
        evidences = get_abstracts_matching_claims(claim)
        res.append((claim, evidences))
        if debug:
            print('Claim\n', claim)
            for i, evidence_doc in enumerate(evidences):
                print(f'Evidence {i}:\n',
                      f'Similarity: {evidence_doc.score:.3f}\n'
                      f'Quote: {evidence_doc.content}\n',
                      f'Article Title: {evidence_doc.meta.get("title", "")}\n',
                      f'DOI: {evidence_doc.meta.get("doi", "")}\n')
    return res


# https://github.com/dwadden/multivers/blob/main/doc/data.md
def convert_evidences_from_abstracts_to_multivers_format(text, indexer):
    if text.startswith('http'):
        text = get_text_from_url(text)
    claims = get_claims_from_text(text, threshold=0.5)
    evidence_abstracts = get_abstracts_matching_claims(claims, indexer, top_k=30)
    with jsonlines.open('claims_1.jsonl', 'w') as claims_writer, \
            jsonlines.open('corpus_1.jsonl', 'w') as corpus_writer:
        doc_id = 0

        for claim_id, matches in enumerate(evidence_abstracts):
            doc_ids = []
            for i, match in enumerate(matches):
                evidence_abstract = {
                    'doc_id': doc_id,
                    'title': match.meta.get("title", ""),
                    'abstract': sent_tokenize(match.content)
                }
                corpus_writer.write(evidence_abstract)
                doc_ids.append(doc_id)
                doc_id += 1
            claim_doc = {
                'id': claim_id,
                'claim': claims[claim_id],
                'doc_ids': doc_ids
            }
            claims_writer.write(claim_doc)


# https://github.com/dwadden/multivers/blob/main/doc/data.md
def convert_evidences_from_phrases_to_multivers_format(text):
    with jsonlines.open('claims.jsonl', 'w') as claims_writer, \
            jsonlines.open('corpus.jsonl', 'w') as corpus_writer:
        doc_id = 0
        for claim_id, (claim, evidences) in enumerate(get_evidences_from_text(text)):
            doc_ids = []
            title_dic = defaultdict(list)
            # group evidences by title to add them to the same doc in corpus
            for evidence_doc in evidences:
                title = evidence_doc.meta.get("title", "")
                quote = evidence_doc.content
                title_dic[title].append(quote)

            for title in title_dic:
                quotes_from_the_same_article = {
                    'doc_id': doc_id,
                    'title': title,
                    'abstract': title_dic.get(title)
                }
                corpus_writer.write(quotes_from_the_same_article)
                doc_ids.append(doc_id)
                doc_id += 1
            claim_doc = {
                'id': claim_id,
                'claim': claim,
                'doc_ids': doc_ids
            }
            claims_writer.write(claim_doc)


if __name__ == "__main__":
    milvus_indexer = index_in_milvus_by_abstract.get_indexer(remote=True)
    get_abstracts_matching_claims(
        ["Rain now falls in massive bursts more frequently than it once did, triggering record floods.",
         "CO2 is not the cause of our current warming trend."],
        milvus_indexer,
        debug=True)
    # convert_evidences_from_abstracts_to_multivers_format("https://www.nationalgeographic.com/environment/article/amazon-rainforest-now-appears-to-be-contributing-to-climate-change",
    #                                                      indexer=milvus_indexer)
    # convert_evidences_from_abstracts_to_multivers_format(
    #     """CO2 is not the cause of our current warming trend.
    #     Arctic sea ice has expanded in recent years.
    #     Polar bears’ population is growing and is not threatened by climate change.
    #     CO2 is good for plant life""")
