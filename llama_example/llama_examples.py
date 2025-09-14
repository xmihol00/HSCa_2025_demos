#!/usr/bin/env python3
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# helper to initialize the model and pipeline
def load_model(model_path):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # load model in half precision for 24gb gpu
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto"
    )
    # build pipeline
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )

# helper to run prompt through model
def ask(llama, prompt):
    output = llama(prompt)[0]["generated_text"]
    # remove prompt echo if present
    return output[len(prompt):].strip()

# all demo examples
def run_example(llama, example):
    if example == "instruction":
        prompt = "Explain photosynthesis to a 10-year-old in simple terms."
        print(ask(llama, prompt))

    elif example == "reasoning":
        prompt = """You are a tutor helping a student with math.
Question: If a train leaves at 3 PM traveling at 60 km/h, 
and another leaves the same station at 4 PM traveling at 80 km/h in the same direction, 
when will the second train catch up? 
Explain step by step."""
        print(ask(llama, prompt))

    elif example == "summarization":
        text = """
Artificial intelligence refers to the simulation of human intelligence in machines
that are programmed to think like humans and mimic their actions...
"""  # you can replace with longer text
        prompt = f"Summarize the following text in 3 bullet points:\n\n{text}"
        print(ask(llama, prompt))

    elif example == "style":
        prompt = """Rewrite the following paragraph in the style of a Shakespearean play:

The city is noisy and full of energy. People rush to work, cars honk, 
and every corner has someone selling something."""
        print(ask(llama, prompt))

    elif example == "codegen":
        prompt = """Write a Python function that checks whether a number is prime.
Add comments to explain each step."""
        print(ask(llama, prompt))

    elif example == "codeexp":
        code_snippet = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
        prompt = f"Explain what the following Python code does in simple terms:\n{code_snippet}"
        print(ask(llama, prompt))

    elif example == "json":
        prompt = """Extract structured data from the following sentence.
Output as JSON with keys: 'name', 'age', 'city'.

Sentence: Maria is 29 years old and lives in Barcelona."""
        print(ask(llama, prompt))

    elif example == "chat":
        conversation = """
You are a friendly chatbot. Keep answers short.

User: Hi, my name is David.
Assistant: Hello David! How are you today?
User: I’m doing great. Can you remind me what my name is?
Assistant:
"""
        print(ask(llama, conversation))

    else:
        print(f"Unknown example: {example}")


def main():
    parser = argparse.ArgumentParser(
        description="""Run demo examples with Meta-LLaMA-3.1-8B-Instruct
examples:
# run instruction following example
python llama_examples.py -e instruction

# run reasoning example
python llama_examples.py -e reasoning

# run summarization example
python llama_examples.py -e summarization

# short demo for structured json output
python llama_examples.py -e json"""
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="./Meta-Llama-3.1-8B-Instruct",
        help="path to local LLaMA-3.1 8B Instruct model folder"
    )
    parser.add_argument(
        "-e", "--example",
        type=str,
        required=True,
        choices=[
            "instruction", "reasoning", "summarization", "style",
            "codegen", "codeexp", "json", "chat"
        ],
        help="which example to run"
    )
    args = parser.parse_args()

    # load model
    llama = load_model(args.model)

    # run chosen example
    run_example(llama, args.example)


if __name__ == "__main__":
    main()
