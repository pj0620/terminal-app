def load_prompt(prompt: str) -> str:
  with open('prompts/' + prompt) as f:
    return f.read()
