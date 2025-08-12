# Soup Data Generator (Lateral Thinking Puzzle Generator)

A tool that uses LangChain and LLMs (DeepSeek/OpenAI) to automatically generate lateral thinking puzzles (also known as "situation puzzles" or "turtle soup puzzles" in some cultures). It supports asynchronous generation of multiple puzzles, duplicate checking, and saves results in both JSON and CSV formats.

## Features

- Generate original lateral thinking puzzles using LLMs (DeepSeek/OpenAI)
- Duplicate checking using Chroma vector database
- Asynchronous batch generation
- Choice of override or append mode
- Output in both JSON and CSV formats
- Token usage tracking

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `config.ini` file:
```ini
[OpenAI]
api_key = your-openai-api-key-here

[DeepSeek]
api_key = your-deepseek-api-key-here
api_base = https://api.deepseek.com/v1
model = deepseek-chat
```

## Usage

### Basic Usage

```bash
# Generate 5 new puzzles (default append mode)
python soupDataGenerator.py --num 5

# Specify custom output files
python soupDataGenerator.py --num 5 --json custom.json --csv custom.csv
```

### Database Modes and Prompt Selection

```bash
# Append mode: Add new puzzles starting from prompt index 0
python soupDataGenerator.py --num 5 --db-mode append

# Override mode: Clear database and generate puzzles starting from prompt index 10
python soupDataGenerator.py --num 5 --start 10 --db-mode override


# Append mode with custom start index: Add new puzzles starting from prompt index 20
python soupDataGenerator.py --num 5 --start 20 --db-mode append
```

The `--start` parameter can be used with either database mode to control which prompts from your prompt file are used for generation. This is particularly useful when:
- You want to skip previously used prompts
- You want to generate puzzles from a specific section of your prompt file
- You need to resume generation from where a previous session left off

### Command Line Arguments

- `--num`: Number of puzzles to generate (default: 5)
- `--json`: JSON output filename (default: puzzle_data.json)
- `--csv`: CSV output filename (default: puzzle_data.csv)
- `--db-mode`: Database mode, either 'append' or 'override' (default: append)

## Output Format

### JSON Format
```json
{
    "id": "unique_identifier",
    "name": "short_puzzle_name",
    "soupFace": "puzzle_situation",
    "soupBase": "puzzle_solution",
    "fewShots": [
        "question1:answer1",
        "question2:answer2",
        "question3:answer3"
    ],
    "clues": [
        "clue1",
        "clue2",
        "clue3"
    ]
}
```

### CSV Format
Contains the following columns:
- id
- name
- soupFace
- soupBase
- fewShots (semicolon-separated)
- clues (semicolon-separated)

## Important Notes

- Ensure your API keys are properly configured in config.ini
- The Chroma database will be created automatically on first run
- Generated puzzles are automatically checked for duplicates
- The system supports resuming after interruption
- Token usage is tracked and stored in the database
- Default concurrency is set to 1 to prevent API rate limiting

## Requirements

- Python 3.7+
- See requirements.txt for full dependency list

## Error Handling

The system includes robust error handling for:
- JSON parsing errors
- API connection issues
- Invalid puzzle formats
- Database operations

Debug output can be monitored in the console during execution.