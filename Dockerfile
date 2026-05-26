FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers scipy numpy scikit-learn ruptures tqdm

RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M'); \
    AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM-135M')"

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

COPY solution_2031_paradigm.py .
COPY main.py .

ENTRYPOINT ["python", "main.py"]
