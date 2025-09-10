from llama_cpp import Llama
import os, json

MODEL_PATH = os.path.join(os.getcwd(), "models", "medgemma-4b-it_Q4_K_M.gguf")
llm = Llama(model_path=MODEL_PATH, n_threads=8)

prompt = "I have a fever. What should I do?"
resp = llm(prompt, max_tokens=120, temperature=0.0)
print("RAW RESPONSE:")
print(json.dumps(resp, indent=2) if isinstance(resp, dict) else repr(resp))
