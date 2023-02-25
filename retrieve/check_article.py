from collections import defaultdict

import jsonlines

import claim_identifier
import index.index_in_faiss_by_abstract
from claim_identifier import get_claims_from_text
from index.utils import retrieve_matches_for_a_phrase
from utils import get_text_from_url


def get_evidences_from_text(text, debug=False):
    if text.startswith('http'):
        text = get_text_from_url(text)
    claims_from_text = get_claims_from_text(text)
    res = []
    for claim in claims_from_text:
        evidences = retrieve_matches_for_a_phrase(claim)
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
def convert_evidences_to_multivers_format(text):
    if text.startswith('http'):
        text = get_text_from_url(text)
    claims = get_claims_from_text(text, threshold=0.5)
    evidence_abstracts = index.index_in_faiss_by_abstract\
        .get_abstracts_matching_claims(claims, top_k=30)
    with jsonlines.open('claims_1.jsonl', 'w') as claims_writer, \
            jsonlines.open('corpus_1.jsonl', 'w') as corpus_writer:
        doc_id = 0

        for claim_id, matches in enumerate(evidence_abstracts):
            doc_ids = []
            for i, match in enumerate(matches):
                evidence_abstract = {
                    'doc_id': doc_id,
                    'title': match.meta.get("title", ""),
                    'abstract': claim_identifier.tokenize_texts_to_sentences(match.content)
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
# def convert_evidences_to_multivers_format(text):
#     with jsonlines.open('claims.jsonl', 'w') as claims_writer, \
#             jsonlines.open('corpus.jsonl', 'w') as corpus_writer:
#         doc_id = 0
#         for claim_id, (claim, evidences) in enumerate(get_evidences_from_text(text)):
#             doc_ids = []
#             title_dic = defaultdict(list)
#             # group evidences by title to add them to the same doc in corpus
#             for evidence_doc in evidences:
#                 title = evidence_doc.meta.get("title", "")
#                 quote = evidence_doc.content
#                 title_dic[title].append(quote)
#
#             for title in title_dic:
#                 quotes_from_the_same_article = {
#                     'doc_id': doc_id,
#                     'title': title,
#                     'abstract': title_dic.get(title)
#                 }
#                 corpus_writer.write(quotes_from_the_same_article)
#                 doc_ids.append(doc_id)
#                 doc_id += 1
#             claim_doc = {
#                 'id': claim_id,
#                 'claim': claim,
#                 'doc_ids': doc_ids
#             }
#             claims_writer.write(claim_doc)


convert_evidences_to_multivers_format("https://www.nationalgeographic.com/environment/article/amazon-rainforest-now-appears-to-be-contributing-to-climate-change")
# convert_evidences_to_multivers_format(
#     """CO2 is not the cause of our current warming trend.
#     Arctic sea ice has expanded in recent years.
#     Polar bears’ population is growing and is not threatened by climate change.
#     CO2 is good for plant life""")

# get_evidences("""“The question is, ‘how much of an impact do we have on [climate]?’ That has not totally been quantified”.
# climate change is happening anyway, “The ice age happened without us…It’s probably this constant cycle”""")

# convert_evidences_to_multivers_format("""Greenland’s Melting Ice Is No Cause for Climate-Change Panic
# The annual loss has been decreasing in the past decade even as the globe continues to warm.
# One of the most sacred tenets of climate alarmism is that Greenland’s vast ice sheet is shrinking ever more rapidly because of human-induced climate change. The media and politicians warn constantly of rising sea levels that would swamp coastlines from Florida to Bangladesh. A typical headline: “Greenland ice sheet on course to lose ice at fastest rate in 12,000 years.”
# With an area of 660,000 square miles and a thickness up to 1.9 miles, Greenland’s ice sheet certainly deserves attention. Its shrinking has been a major cause of recent sea-level rise, but as is often the case in climate science, the data tell quite a different story from the media coverage and the political laments.
# The chart nearby paints a bigger picture that is well known to experts but largely absent from the media and even from the most recent United Nations climate report. It shows the amount of ice that Greenland has lost every year since 1900, averaged over 10-year intervals; the annual loss averages about 110 gigatons. (A gigaton is one billion metric tons, or slightly over 2.2 trillion pounds.) That is a lot, but that water has caused the planet’s oceans to rise each year by only 0.01 inch, about one-fifth the thickness of a dime.
# In contrast, the United Nations’ Intergovernmental Panel on Climate Change projects that for the most likely course of greenhouse-gas emissions in the 21st century, the average annual ice loss would be somewhat larger than the peak values shown in the graph. That would cause sea level to rise by 3 inches by the end of this century, and if losses were to continue at that rate, it would take about 10,000 years for all the ice to disappear, causing sea level to rise more than 20 feet.
# To assess the importance of human influences, we can look at how the rate of ice loss has changed over time.
# In that regard, the graph belies the simplistic notion that humans are melting Greenland. Since human warming influences on the climate have grown steadily—they are now 10 times what they were in 1900— you might expect Greenland to lose more ice each year. Instead there are large swings in the annual ice loss and it is no larger today than it was in the 1930s, when human influences were much smaller. Moreover, the annual loss of ice has been decreasing in the past decade even as the globe continues to warm.
# While a warming globe might eventually be the dominant cause of Greenland’s shrinking ice, natural cycles in temperatures and currents in the North Atlantic that extend for decades have been a much more important influence since 1900. Those cycles, together with the recent slowdown, make it plausible that the next few decades will see a further, perhaps dramatic slowing of ice loss. That would be inconsistent with the IPCC’s projection and wouldn’t at all support the media’s exaggerations.
# Much climate reporting today highlights short-term changes when they fit the narrative of a broken climate but then ignores or plays down changes when they don’t, often dismissing them as “just weather.”
# Climate unfolds over decades. Although short-term changes might be deemed news, they need to be considered in a many-decade context. Media coverage omitting that context misleadingly raises alarm. Greenland’s shrinking ice is a prime example of that practice.
# If Greenland’s ice loss continues to slow, headline writers will have to find some other aspect of Greenland’s changes to grab our attention, and politicians will surely find some other reason to justify their favorite climate policies.""")
#
