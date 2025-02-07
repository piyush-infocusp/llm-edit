"""module to test models"""
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import re
import html
from huggingface_hub import InferenceClient

def load_dataset(path):
    dataset_container = []
    df = pd.read_csv(path)
    for row in df.to_dict(orient='records'):
        dataset_container.append(row)
    return dataset_container


client = InferenceClient(
	api_key=''
)


def get_generation_result(model, prompt):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=None
    )
    return completion.choices[0].message.content


def load_template(template_dir, template_name):
    # Create a Jinja environment
    env = Environment(loader=FileSystemLoader(template_dir))

    # Load a specific template
    template = env.get_template(template_name)

    # Print the output
    return template


def sanitize_string(input_string):
    # 1. Trim leading and trailing whitespace
    sanitized = input_string.strip()
    
    # 2. Escape HTML special characters to prevent XSS (Cross-Site Scripting)
    sanitized = html.escape(sanitized)
    
    # 3. Remove unwanted special characters (optional)
    #    Remove characters that aren't letters, digits, or spaces
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', sanitized)
    
    # 4. Optionally, normalize string (convert to lowercase or other formatting)
    sanitized = sanitized.lower()
    
    return sanitized

def run_generation(models):
    dataset_path = "/home/piyush.sar/Projects/LegalSifter/llm-edit/src/datasets/dataset.csv"
    prefix = ", ".join([sanitize_string(model) for model in models])
    store_path = f"/home/piyush.sar/Projects/LegalSifter/llm-edit/src/result/{prefix}_result.csv"
    dataset = load_dataset(dataset_path)

    template_dir = "/home/piyush.sar/Projects/LegalSifter/llm-edit/src/templates"

    template_name = "prompt.jinja"
    template = load_template(template_dir, template_name)
    master_results = {}
    for model in models:
        results = []
        for datapoint in dataset:
            data = {
                'text': datapoint['content'],
                'instructions': [
                    {'id': 1, 'task': datapoint["guidance_1"]},
                    {'id': 2, 'task': datapoint["guidance_2"]},
                    {'id': 3, 'task': datapoint["guidance_3"]},
                ]
            }
            prompt = template.render(data)
            result = get_generation_result(model, prompt)
            print("----------------------START---------------------")
            print(models)
            print()
            print(prompt)
            print()
            print(result)
            results.append(result)
            print("----------------------END---------------------")
            print()
        master_results[model] = results

    df = pd.read_csv(dataset_path)
    for model, results in master_results.items():
        df[model] = results
    df.to_csv(store_path, index=False)

if __name__ == "__main__":
    run_generation(["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"])