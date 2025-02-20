import re
import pandas as pd
import re
from bs4 import BeautifulSoup

# does not raise error for invalid tags
def parse_text_with_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    parsed_data = [(tag.name, tag.text.strip()) for tag in soup.find_all()]
    return parsed_data

def check_parsing(text):
    """Check if the text contains valid, invalid, and properly closed tags."""
    {
        "PARSING_ERROR": False,
        "NO_REQUIRED_TAGS": False,
        "INVALID_TAGS_FOUND": False
    }

    # Find all opening and closing tags
    open_tags = re.findall(r'<(\w+\s*\d*)>', text)
    close_tags = re.findall(r'</(\w+\s*\d*)>', text)

    # Count occurrences of each tag
    open_tag_counts = {tag: open_tags.count(tag) for tag in set(open_tags)}
    close_tag_counts = {tag: close_tags.count(tag) for tag in set(close_tags)}

    # Ensure every tag has an equal number of opening and closing counterparts
    for tag in set(open_tag_counts.keys()).union(set(close_tag_counts.keys())):
        if open_tag_counts.get(tag, 0) != close_tag_counts.get(tag, 0):
            return False, "PARSING_ERROR" # Parsing error due to unbalanced tags

    if set(open_tags) == set():
        return False, "NO_TAGS_FOUND"

    if set(open_tags).issubset({'insert 1', 'delete 1', 'noedit 1'}):
        return True, None
    else:
        return False, "ERROR_WITH_TAGS" # Error with tags
    # Check if at least one of the required tags is present
    # if not re.search(r'<insert 1>|<delete 1>|<noedit 1>', text):
    #     return None, True, False 

    # # Check if there are any invalid insert or delete tags (e.g., insert 2, delete 3, etc.)
    # if re.search(r'<insert [2-9]\d*>|<delete [2-9]\d*>', text):
    #     return None, False, True

    # return True


def process_text(text):
    text = text.replace("</noedit>", "</noedit 1>")

    is_correct, reason = check_parsing(text)

    if is_correct:
        # Remove <insert 1> tags and their content
        text = re.sub(r'<insert 1>.*?</insert 1>', '', text, flags=re.DOTALL)
        # Remove <noedit 1> tags and their content
        text = re.sub(r'<noedit 1>.*?</noedit 1>', '', text, flags=re.DOTALL)
        # Replace <delete 1> tags but keep their content
        text = re.sub(r'<delete 1>(.*?)</delete 1>', r'\1', text, flags=re.DOTALL)

        return text.strip(), True, None
    else:
        return None, False, reason


def match_without_whitespace(pattern, text):
    if not text:
        return None
    # Remove all whitespace from both pattern and text
    pattern_no_space = re.sub(r'\s+', '', pattern)
    text_no_space = re.sub(r'\s+', '', text)

    # Use re.fullmatch to check if the entire processed text matches the pattern
    return bool(re.fullmatch(pattern_no_space, text_no_space))

if __name__ == "__main__":

    dataset = pd.read_csv('/home/piyush.sar/Projects/LegalSifter/llm-edit/src/result/COEDITDATASET_googlegemma29bit_result.csv')

    results = dataset["google/gemma-2-9b-it"].to_list()

    output = [list(process_text(result)) for result in results]

    dataset[["Extracted Original Text", "Correct ?", "Error Reason"]] = output

    dataset["grounded ?"] = dataset[["content","Extracted Original Text"]].apply(
        lambda x: match_without_whitespace(x["content"], x["Extracted Original Text"]), axis=1)

    dataset.to_csv('/home/piyush.sar/Projects/LegalSifter/llm-edit/src/result/COEDITDATASET_googlegemma29bit_result_20_02.csv', index=False)
    
    total_datapoints = len(dataset)
    dataset_malformed_tags_error_percentage = round(len(dataset[dataset["Correct ?"] == False]) / total_datapoints * 100,2)
    total_datapoints_after_removing_parsing_errors = total_datapoints - len(dataset[dataset["Correct ?"]== False])
    dataset_grounding_error_percentage = round(len(dataset[dataset["grounded ?"] == False]) / total_datapoints_after_removing_parsing_errors * 100,2)
    
    print(
        f"""
        Total Datapoints: {total_datapoints}
        Dataset Malformed Tags Error Percentage: {dataset_malformed_tags_error_percentage}%
        Total Datapoints After Removing Parsing Errors: {total_datapoints_after_removing_parsing_errors}
        Dataset Grounding Error Percentage: {dataset_grounding_error_percentage}%
        """
        )
    