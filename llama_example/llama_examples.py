#!/usr/bin/env python3
import time
import logging
import sys
import argparse
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
def load_model(model_path="./Meta-Llama-3.1-8B-Instruct", device_map="cuda:0", torch_dtype="auto", max_new_tokens=1024, temperature=0.7, top_p=0.9):
    with Timer("model load"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,    # specifies which GPU/device to load the model on, affects memory usage and inference speed
            torch_dtype=torch_dtype   # automatically selects the optimal data type (float16/bfloat16) to reduce memory usage while maintaining quality
        )
        llama = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,  # maximum number of new tokens to generate, controls output length and inference time
            temperature=temperature,        # controls randomness in token selection (0.0=deterministic, 1.0=very random), affects creativity vs consistency
            top_p=top_p                     # nucleus sampling parameter that keeps only tokens with cumulative probability <= 0.9, affects output diversity
        )
    return llama

# helper to query model with logging
def ask(llama, prompt):
    with Timer("preprocessing"):
        print("\n=== PROMPT ===\n")
        print(prompt)
        print("\n====================\n")
        prepared_prompt = prompt.strip()

    with Timer("inference"):
        output = llama(prepared_prompt)[0]["generated_text"]

    with Timer("postprocessing"):
        reply = output[len(prepared_prompt):].strip() # the model echoes the prompt, remove it
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
    parser = argparse.ArgumentParser(description="LLaMA model inference with configurable parameters")
    parser = argparse.ArgumentParser(
        description="LLaMA model inference with configurable parameters",
        epilog="""
Examples:
  %(prog)s                                          # use default settings
  %(prog)s --temperature 0.5 --max-new-tokens 512   # lower creativity, shorter responses
  %(prog)s --device-map cpu --torch-dtype float32   # run on CPU with float32 precision
  %(prog)s --top-p 0.95 --temperature 1.0           # more diverse and creative outputs
  %(prog)s --device-map auto --max-new-tokens 4096  # auto device selection, longer responses
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device-map", default="cuda:0", 
                       choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7", "cuda:8", "cpu", "auto"],
                       help="device to load the model on (default: cuda:0)")
    parser.add_argument("--torch-dtype", default="auto",
                       choices=["auto", "float16", "bfloat16", "float32"],
                       help="data type for model weights (default: auto)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                       help="maximum number of new tokens to generate (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="controls randomness in generation (0.0-2.0, default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="nucleus sampling parameter (0.0-1.0, default: 0.9)")
    
    args = parser.parse_args()
    
    llama = load_model(
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    while True:
        print("""
Choose an option:
  1.  Instruction following
  2.  Reasoning
  3.  Summarization
  4.  Style transfer
  5.  Code generation
  6.  Code explanation
  7.  Structured JSON output
  8.  Chat mode
  9.  Custom prompt
  10. Performance test
  0.  Exit
""")
        choice = input("Enter your choice: ").strip()

        if choice in ("0", "q", "quit", "exit"):
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
        elif choice == "10":
            test_prompt = "Explain the theory of relativity in simple terms."
            iterations = 10
            print(f"Running performance test with {iterations} iterations...")
            start_time = time.perf_counter()
            for i in range(iterations):
                print(f"\n--- Iteration {i+1} ---")
                reply = ask(llama, test_prompt)
                print("\n=== MODEL OUTPUT ===\n")
                print(reply)
                print("\n====================\n")
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f"Total time for {iterations} iterations: {total_time:.2f}s")
            print(f"Average time per iteration: {total_time/iterations:.2f}s")
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")
        sys.exit(0)
