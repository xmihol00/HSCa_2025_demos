#!/usr/bin/env python3
import time
import logging
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("llama-cli")

# helper: measure elapsed time
class Timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.start = time.perf_counter()
        logger.info(f"{self.label} started")
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        end = time.perf_counter()
        logger.info(f"{self.label} finished in {end - self.start:.2f}s")

# load model only once
def load_model(model_path="./Meta-Llama-3.1-8B-Instruct"):
    with Timer("model load"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto"
        )
        llama = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
    return llama

# helper to query model with logging
def ask(llama, prompt):
    with Timer("preprocessing"):
        prepared_prompt = prompt.strip()

    with Timer("inference"):
        output = llama(prepared_prompt)[0]["generated_text"]

    with Timer("postprocessing"):
        reply = output[len(prepared_prompt):].strip()
    return reply

# predefined examples
def run_example(llama, example):
    if example == "1":
        prompt = "Explain photosynthesis to a 10-year-old in simple terms."
    elif example == "2":
        prompt = """You are a tutor helping a student with math.
Question: If a train leaves at 3 PM traveling at 60 km/h, 
and another leaves the same station at 4 PM traveling at 80 km/h in the same direction, 
when will the second train catch up? 
Explain step by step."""
    elif example == "3":
        text = """
Artificial intelligence refers to the simulation of human intelligence in machines
that are programmed to think like humans and mimic their actions...
"""
        prompt = f"Summarize the following text in 3 bullet points:\n\n{text}"
    elif example == "4":
        prompt = """Rewrite the following paragraph in the style of a Shakespearean play:

The city is noisy and full of energy. People rush to work, cars honk, 
and every corner has someone selling something."""
    elif example == "5":
        prompt = """Write a Python function that checks whether a number is prime.
Add comments to explain each step."""
    elif example == "6":
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
    elif example == "7":
        prompt = """Extract structured data from the following sentence.
Output as JSON with keys: 'name', 'age', 'city'.

Sentence: Maria is 29 years old and lives in Barcelona."""
    else:
        logger.warning("invalid example choice")
        return
    reply = ask(llama, prompt)
    print("\n=== MODEL OUTPUT ===\n")
    print(reply)
    print("\n====================\n")

# chat mode
def chat_mode(llama):
    print("\n=== Chat mode (type 'exit' to quit) ===\n")
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            print("Exiting chat mode...")
            break

        # build conversation prompt
        history.append(f"User: {user_input}")
        conversation = "\n".join(history) + "\nAssistant:"

        reply = ask(llama, conversation)
        history.append(f"Assistant: {reply}")

        print(f"\nAssistant: {reply}\n")

# interactive menu
def main():
    llama = load_model()

    while True:
        print("""
Choose an option:
  1. Instruction following
  2. Reasoning
  3. Summarization
  4. Style transfer
  5. Code generation
  6. Code explanation
  7. Structured JSON output
  8. Chat mode
  9. Custom prompt
  0. Exit
""")
        choice = input("Enter your choice: ").strip()

        if choice == "0":
            print("Goodbye!")
            break
        elif choice in [str(i) for i in range(1, 8)]:
            run_example(llama, choice)
        elif choice == "8":
            chat_mode(llama)
        elif choice == "9":
            prompt = input("Enter your custom prompt: ")
            reply = ask(llama, prompt)
            print("\n=== MODEL OUTPUT ===\n")
            print(reply)
            print("\n====================\n")
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")
        sys.exit(0)
