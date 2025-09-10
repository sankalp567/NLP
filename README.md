# Healthcare Assistant (Task 1 - NLP Project)

This is a domain-specific healthcare information assistant built with **Google MedGemma-4B-IT (quantized GGUF)**.

## Features
- Answers general healthcare FAQs (flu, hydration, exercise).
- Provides hospital/clinic support info (mock dataset).
- Instruction-tuned prompts for safe, concise answers.
- Gradio UI for interactive chat.
- 
## The app will launch on http://localhost:7860
## Setup
```bash
git clone <repo>
cd Healthcare-Assistant-MedGemma
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python app.py 
.
