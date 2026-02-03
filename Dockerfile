FROM nvidia/cuda:12.4.1-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.4/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# Install vLLM
RUN python3 -m pip install vllm==0.11.0

# Patch 1: DisabledTqdm to handle huggingface_hub>=0.25 passing disable= in kwargs
RUN python3 -c "p='/usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader/weight_utils.py'; c=open(p).read(); o='    def __init__(self, *args, **kwargs):\n        super().__init__(*args, **kwargs, disable=True)'; n='    def __init__(self, *args, **kwargs):\n        kwargs.pop(\"disable\", None)\n        super().__init__(*args, **kwargs, disable=True)'; open(p,'w').write(c.replace(o,n)) if o in c else None; print('Patched DisabledTqdm' if o in c else 'DisabledTqdm already patched')"

# Patch 2: get_cached_tokenizer to skip caching for MistralTokenizer (missing all_special_tokens_extended)
RUN python3 -c "p='/usr/local/lib/python3.10/dist-packages/vllm/transformers_utils/tokenizer.py'; c=open(p).read(); o='    cached_tokenizer = copy.copy(tokenizer)'; n='    # Skip caching for tokenizers without standard HF attributes (e.g., MistralTokenizer)\n    if not hasattr(tokenizer, \"all_special_tokens_extended\"):\n        return tokenizer\n    cached_tokenizer = copy.copy(tokenizer)'; open(p,'w').write(c.replace(o,n,1)) if o in c and 'all_special_tokens_extended' not in c else None; print('Patched get_cached_tokenizer' if o in c else 'get_cached_tokenizer already patched')"

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 

ENV PYTHONPATH="/:/vllm-workspace"

COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
