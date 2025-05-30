
from PIL import Image
import io
import fitz
import torch
import re
from pytesseract import image_to_string
from transformers import CLIPProcessor, CLIPModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

#for zero-shot classification
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
CLIP_LABELS = ["handwritten", "OTHER", "ID card", "form"]


#for image to text transformation
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model.eval()


def extract_text_from_pdf(file_data: bytes) -> str:
    try:
        with fitz.open(stream=file_data, filetype="pdf") as doc:
            text = " ".join(page.get_text() for page in doc).strip() # type: ignore
            return text.encode("ascii", errors="ignore").decode()  # strip non-ASCII
    except Exception:
        return ""
    

def extract_year(text: str) -> str:
    match = re.search(r'\b(19|20)\d{2}\b', text)
    return match.group(0) if match else "Unknown"


# this below is not much needed - it is only for detecting text from a picture (just in case)
def extract_text_with_trocr(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=200) # type: ignore
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values # type: ignore
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values) # type: ignore
        text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # type: ignore
        return text.strip()
    except Exception as e:
        print(f"TrOCR OCR failed: {e}")
        return ""

def extract_text(file_data: bytes) -> str:
    text = extract_text_from_pdf(file_data)
    if len(text.strip()) < 10:
        text = extract_text_with_trocr(file_data)
    return text


def detect_form_type(text: str) -> str:
    text = text.lower()
    positions = {}
    for form in ["form 1040", "form w-2", "form 1099"]:
        idx = text.find(form)
        if idx != -1:
            positions[form] = idx
    if not positions:
        return "OTHER"
    
    earliest_form = min(positions, key=positions.get) # type: ignore
    return {
        "form 1040": "1040",
        "form w-2": "W2",
        "form 1099": "1099"
    }[earliest_form]


def classify_with_clip(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200) # type: ignore
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    inputs = clip_processor(text=CLIP_LABELS, images=image, return_tensors="pt", padding=True) # type: ignore
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        pred_idx = probs.argmax(-1).item()
        label = CLIP_LABELS[pred_idx]
        #print("CLIP classification:", label)
        return label