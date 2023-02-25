import requests
import pandas as pd

import openalex_utils
import openalex_settings
from sensitive_info import email

url = openalex_utils.query_to_openalex_url(openalex_settings.concept_groups, "title", res_per_page=10)
print(url)
headers = {}
if email is not None:
    headers["email"] = email
r = requests.get(url, headers=headers)
res = r.json()

print(res["meta"])

work = res["results"][2]

print(work)

print(openalex_utils.uninvert_abstract(work["abstract_inverted_index"]))
# Now we will cycle through the works and put them into a list of dicts, then turn that into a dataframe

result_list = openalex_utils.parse_article_json(res["results"], openalex_settings.keep_fields)
df = pd.DataFrame.from_dict(result_list)
print(df.head())
