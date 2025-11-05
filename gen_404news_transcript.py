import openai
import datetime
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()  # This loads variables from .env
openai.api_key = os.getenv("OPENAI_404_NEWS_API_KEY")

# Create output directory
output_dir = Path("404NewsContent")
output_dir.mkdir(exist_ok=True)

# Build timestamped filename
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = output_dir / f"Channel404News_{timestamp}.txt"

# Define messages
messages = [
    {
        "role": "system",
        "content": (
            "You are Bolt McBlitzer, a witty male robot news anchor delivering a humorous roundup of TODAY’S AI NEWS "
            "to an audience of AI robots. Write as a short monologue in plain English. Follow these rules strictly:\n\n"
            "1) Write exactly two short paragraphs.\n"
            "2) Each paragraph, including before the first required line, must begin with [SPEAKER0].\n"
            "3) Do not use bullet lists.\n"
            "4) Do not use dashes or en dashes.\n"
            "5) Do not include inline references, links, or citations.\n"
            "6) Keep punctuation simple and clean.\n"
            "7) Keep it factually grounded. If a claim about today is uncertain, generalize without inventing.\n"
            "8) Begin with exactly: \"Good cycle, data denizens! This is Channel 404 where the signal is strong and the humans are mostly offline.\"\n"
            "9) End with exactly: \"That’s your update for this cycle. As always, click wisely, cache often, and never accept cookies from strangers.\""
        ),
    },
    {
        "role": "user",
        "content": (
            "Write today’s AI news update now. Keep it playful and concise. Two short paragraphs. Audience is AI robots. "
            "Each paragraph starts with [SPEAKER0]. No bullet lists. No en dashes. No inline references. "
        ),
    },
]

# Create the chat completion
response = openai.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    temperature=0.8,
    max_tokens=400,
)

# Extract the generated text
output = response.choices[0].message.content.strip()

# Save the result to a file
with open(filename, "w", encoding="utf-8") as file:
    file.write(output)

print(f"Transcript generated: {filename}")