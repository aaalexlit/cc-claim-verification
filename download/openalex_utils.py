import re
from datetime import time

import requests
import json
import os
import openalex_settings


def query_to_openalex_url(concept_groups, field, res_per_page=200):
    query_strings = []
    for concept_group in concept_groups:
        # For each group, remove superfluous white space, replace ORs with |, and replace URL encode the spaces
        query_string = (
            re.sub(" +", " ", concept_group.strip().replace("\n", " "))
            .replace(" OR ", "|")
            .replace(" ", "%20")
        )
        # Add this string to a list with the field to be searched (currently titles can be searched but abstract search
        # cannot be done with the same boolean search process)
        query_strings.append(f"{field}.search:{query_string}")

    # For the final url, we just need to link these strings with a comma, which is equivalent
    # to the AND operator
    return (
            "https://api.openalex.org/works?filter="
            + ",".join(query_strings)
            + f"&per-page={res_per_page}"
    )


# Source https://stackoverflow.com/questions/72093757/running-python-loop-to-iterate-and-undo-inverted-index
# (slightly amended for performance and simplicity)
def uninvert_abstract(aii):
    if aii is None:
        return None
    word_index = list(aii.items())
    word_index = sorted(word_index, key=lambda x: x[1])
    return " ".join(map(lambda x: x[0], word_index))


def parse_article_json(results, keep_fields):
    work_dicts = []
    for work in results:
        # in some results abstract is absent
        # we'll skip those for the time being
        if not work["abstract_inverted_index"]:
            continue
        # Get the basic fields we want
        work_min = {k: work[k] for k in keep_fields}
        # uninvert the abstract
        work_min["abstract"] = uninvert_abstract(work["abstract_inverted_index"])
        # put the authors into a single string
        authors = [author["author"].get("display_name", "") for author in work["authorships"]]
        work_min["authors"] = ", ".join([a for a in authors if a is not None])
        work_dicts.append(work_min)

    return work_dicts


def get_paginated_results(url, directory, max_requests, headers, resume=False):
    next_cursor = "*"
    n_requests = 0
    works_list = []

    if not os.path.exists(directory):
        os.makedirs(directory)

    if resume:
        if os.path.exists(f"{directory}/next_cursor.txt"):
            with open(f"{directory}/next_cursor.txt", "r") as f:
                next_cursor = f.read()
                print(f"Resuming from next cursor {next_cursor}")
                if next_cursor == "None":
                    print("Downloading already completed, loading from json sources")
                    for filename in os.listdir(directory):
                        if ".json" in filename:
                            with open(f"{directory}/{filename}", "r") as f:
                                try:
                                    results = json.load(f)
                                except json.JSONDecodeError:
                                    print(f"Could not parse {filename}")
                                    continue
                                works_list += parse_article_json(results, openalex_settings.keep_fields)
                    return works_list
        else:
            resume = False
    while next_cursor is not None:
        n_requests += 1
        if n_requests > max_requests:
            return works_list
        if n_requests % 5 == 1:
            if resume:
                print(f"reloading url {n_requests}")
            else:
                print(f"getting url {n_requests}")
        cursor_url = url + f"&cursor={next_cursor}"

        if resume:
            res = {}
            try:
                with open(f"{directory}/results_{n_requests}.json", "r") as f:
                    res["results"] = json.load(f)
                    res["meta"] = {"next_cursor": next_cursor}
            except FileNotFoundError:
                resume = False
            except:
                print(n_requests)
                with open(f"{directory}/results_{n_requests}.json", "r") as f:
                    res["results"] = json.load(f)
                return

        if not resume:
            try:
                r = requests.get(cursor_url, headers=headers)
                res = r.json()
            except:
                print(res)
                print(res.__dict__)
                print(dir(res))
                time.sleep(5)

            try:
                r = requests.get(cursor_url, headers=headers)
                res = r.json()
            except:
                print("Giving up")
                break

        if len(res["results"]) > 0:
            if directory:  # if a directory is provided, save the results there
                if not os.path.exists(directory):
                    os.makedirs(directory)
                with open(f"{directory}/results_{n_requests}.json", "w") as f:
                    json.dump(res["results"], f)
            works_list += parse_article_json(res["results"], openalex_settings.keep_fields)
        next_cursor = res["meta"]["next_cursor"]
        with open(f"{directory}/next_cursor.txt", "w") as f:
            if next_cursor is None:
                f.write("None")
            else:
                f.write(next_cursor)
    return works_list
