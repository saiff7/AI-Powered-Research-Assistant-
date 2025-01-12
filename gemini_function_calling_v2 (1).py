import json
import absl.logging  # pip install absl-py
import requests
from datetime import datetime
import os
import shutil
import PIL
import google.generativeai as genai  # pip install google-generative-ai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib

# Set your API key here. You can get it from https://ai.google.dev/gemini-api/docs/api-key
genai.configure(api_key="AIzaSyBKEq4ijo__syj0KLHHrxgoJ6nubG25cUg") 

# Verbosity level (0 = only final assistant messages, 1 = function calls, 2 = function calls and their results)
VERBOSITY = 2

# Enable the display of matplotlib plots once created
SHOW_PLOTS = True 

# Model parameters
TEMPERATURE = 0.0  # Need deterministic results for function calls
MAX_TOKENS = 768
CONTEXT_LENGTH = 8192
BATCH_SIZE = 512

# Color codes for console output
color_dict = {'red': 31, 'green': 32, 'yellow': 33, 'blue': 34, 'magenta': 35, 'white': 37}
current_color = "white"
current_column = 0  # Current column in the console output

SYSTEM_PROMPT = "You are a helpful assistant."

chat_session = None

def find_papers(topic: str) -> dict:
    """
    Fetch research papers related to a given topic using a public API like arXiv.

    Args:
        topic: The research topic or keywords to search for.

    Returns:
        A dictionary containing a list of research papers with their titles and abstracts.
    """
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results=5"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.text

            # Extracting papers from the XML response
            from xml.etree import ElementTree as ET
            root = ET.fromstring(data)
            ns = {'ns': 'http://www.w3.org/2005/Atom'}
            
            papers = []
            for entry in root.findall('ns:entry', ns):
                title = entry.find('ns:title', ns).text.strip()
                abstract = entry.find('ns:summary', ns).text.strip()
                papers.append({"title": title, "abstract": abstract})

            return {"papers": papers}
        else:
            return {"error": f"Could not fetch papers: HTTP {response.status_code}"}
    except Exception as e:
        return {"error": f"Error fetching papers: {str(e)}"}

def create_bar_chart(categories: list['str'], y_values: list['float'], title: str, y_axis_label: str) -> str:
    """Create a chart showing vertical bars and save it to disk."""
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(categories, y_values, color='skyblue', edgecolor='black')
        plt.title(title, fontsize=14, pad=20)
        plt.ylabel(y_axis_label, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save plot to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"plot_{timestamp}.png"
        plt.savefig(filename)

        if SHOW_PLOTS:
            PIL.Image.open(filename).show()

        return {"filename": filename}
    except Exception as e:
        return {"error": f"Error creating plot: {str(e)}"}

def set_color(color_string):
    global current_color
    current_color = color_string

def print_colored(text, end="\n", flush=False):
    """Print colored text to the console if verbosity level allows it."""
    global current_column

    if (VERBOSITY == 0) and not (current_color in ["white", "green"]):
        return
    if (VERBOSITY == 1) and (current_color == "magenta"):
        return

    chunks = text.split(" ")

    while chunks:
        chunk = chunks.pop(0)
        if chunk:
            if current_column + len(chunk) >= shutil.get_terminal_size().columns - 2:
                print()
                current_column = 0

            chunk = chunk + " "
            print(f"\033[{color_dict[current_color]}m{chunk}\033[0m", end='')

            for char in chunk:
                if char == "\n":
                    current_column = 0
                else:
                    current_column += 1

    if end == "\n":
        print()
        current_column = 0

generation_config = {
    "temperature": 0,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction=SYSTEM_PROMPT,
    tools=[find_papers, create_bar_chart],
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

def init_chat_messages():
    """Start a new chat."""
    global chat_session

    chat_session = model.start_chat(history=[], enable_automatic_function_calling=True)

def main():
    global chat_session
    global current_column

    absl.logging.set_verbosity(absl.logging.INFO)
    absl.logging.use_absl_handler()
    init_chat_messages()

    print_colored("\nChat session started. Type 'new' for a new chat, 'end' to exit.")
    print_colored("\nExample requests:\n")
    print_colored("- Find papers on deep learning.")
    print_colored("- Create a bar chart showing values of the first 10 Fibonacci numbers.")

    while True:
        set_color("white")
        user_input = input("\nYou: ").strip()
        current_column = 0

        if user_input.lower() == "end":
            break
        elif user_input.lower() == "new":
            init_chat_messages()
            print("\nStarting new chat session...")
            continue
        elif not user_input:
            continue

        last_num_messages = len(chat_session.history)
        response = chat_session.send_message(user_input)

        while last_num_messages < len(chat_session.history) - 1:
            last_num_messages += 1
            message = chat_session.history[last_num_messages]
            current_column = 0
            print()

            for part in message.parts:
                if "text" in part and part.text.strip():
                    set_color("green")
                    print_colored(f'Assistant: {part.text.strip()}')
                elif "function_call" in part:
                    set_color("blue")
                    print_colored(f'Function call:')
                    print_colored(f'{part.function_call.name}({", ".join([f"{key}={value}" for key, value in part.function_call.args.items()])})')
                elif "function_response" in part:
                    set_color("magenta")
                    print_colored(f'Function response ({part.function_response.name}):')
                    print_colored(str(dict(part.function_response.response)))

if __name__ == "__main__":
    main()
