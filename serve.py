"""Standalone FastAPI service for the fine-tuned Donut receipt extractor.

Run:
    python -m venv .venv && .venv/Scripts/activate
    pip install -r requirements.txt
    uvicorn serve:app --host 0.0.0.0 --port 8000

Defaults to the published model; set MODEL_PATH to use a local folder instead.

Then:
    curl -F "file=@receipt.jpg" http://localhost:8000/extract
    or open http://localhost:8000/docs for a browser upload form
"""
import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Override with MODEL_PATH to point at a local folder or a different revision.
MODEL_PATH = os.environ.get('MODEL_PATH', 'aashishkumar-tech/donut-cord-summary')
FIELDS = ['total_price', 'cashprice', 'changeprice',
          'subtotal_price', 'tax_price', 'discount_price']
TASK_START, TASK_END = '<s_extraction>', '</s_extraction>'
MAX_NEW_TOKENS = 24

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = DonutProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device).eval()
tok = processor.tokenizer

DSI = model.config.decoder_start_token_id
TAG_IDS = {tok.convert_tokens_to_ids(t)
           for t in [TASK_START, TASK_END]
           + [x for f in FIELDS for x in (f'<s_{f}>', f'</s_{f}>')]}


@torch.no_grad()
def guided_predict(image: Image.Image) -> dict:
    """Decode one field at a time between its own tags -- tags can never be emitted
    inside a value, so the output is always well formed."""
    pv = processor(image.convert('RGB'), return_tensors='pt').pixel_values.to(device)
    seq = [DSI, tok.convert_tokens_to_ids(TASK_START)]
    pred = {}
    for f in FIELDS:
        open_id = tok.convert_tokens_to_ids(f'<s_{f}>')
        close_id = tok.convert_tokens_to_ids(f'</s_{f}>')
        prefix = seq + [open_id]
        out = model.generate(
            pv,
            decoder_input_ids=torch.tensor([prefix], device=device),
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=close_id,
            pad_token_id=tok.pad_token_id,
            bad_words_ids=[[t] for t in TAG_IDS if t != close_id],
            num_beams=1,
        )
        ids = [t for t in out[0].tolist()[len(prefix):]
               if t not in TAG_IDS and t != tok.pad_token_id]
        pred[f] = tok.decode(ids, skip_special_tokens=True).strip()
        seq = prefix + ids + [close_id]
    return pred


app = FastAPI(title='Receipt field extractor', version='1.0')


@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL_PATH, 'device': device}


@app.post('/extract')
async def extract(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'not a readable image: {exc}')
    return {'filename': file.filename, 'fields': guided_predict(image)}
