# AccessLens AI

**Visual scene understanding for more accessible digital experiences.**

AccessLens AI is a hackathon prototype that combines object detection and image captioning to make image content easier to inspect and describe.

-----

---
<p align="center">
  <img src="Image.png" alt="MedVision AI Banner" width="100%">
</p>

-----

## Features
- Upload a JPG, PNG, or WEBP image.
- Detect common objects with YOLOv8n.
- Generate a natural-language caption with BLIP.
- Display an annotated image, object labels, and confidence scores.
- Explain model limitations and responsible use.

## Tech stack
- Python
- Streamlit
- Ultralytics YOLOv8
- Hugging Face Transformers / BLIP
- PyTorch

## Run locally
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads model weights and may take several minutes. A stable internet connection is required for initial downloads.

## Model behavior and limitations
- YOLOv8n detects objects from its supported label set; it may miss or misclassify objects.
- BLIP generates a caption and may hallucinate or omit details.
- Confidence scores are not guarantees of correctness.
- This prototype does not measure distance, establish whether a path is traversable, or provide certified mobility/safety guidance.
- Do not use its output as the sole basis for a safety-critical decision.

## Credits
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
- Hugging Face Transformers: https://github.com/huggingface/transformers
- BLIP model: https://huggingface.co/Salesforce/blip-image-captioning-base
- Streamlit: https://streamlit.io/

## Hackathon integrity
Add the exact tools/models you used and describe any AI-assisted development in your final submission. Ensure the code and project are created and committed within the event's allowed hacking window. Do not submit this same project to another hackathon.
