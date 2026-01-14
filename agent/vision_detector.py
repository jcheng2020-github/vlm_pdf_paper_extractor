from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI
from .utils import image_to_data_url

DETECT_SCHEMA: Dict[str, Any] = {
    "name": "page_objects",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string", "enum": ["table", "figure"]},
                        "bbox": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x0": {"type": "number"},
                                "y0": {"type": "number"},
                                "x1": {"type": "number"},
                                "y1": {"type": "number"},
                            },
                            "required": ["x0", "y0", "x1", "y1"],
                        },
                        "caption": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["type", "bbox", "caption", "confidence"],
                },
            }
        },
        "required": ["items"],
    }
}

class VisionDetector:
    """
    Wraps OpenAI vision calls. Returns structured detections:
      {"items":[{"type":"table|figure","bbox":{...},"caption":"...","confidence":0.0-1.0}, ...]}
    """

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def detect(self, page_png: Path) -> Dict[str, Any]:
        data_url = image_to_data_url(page_png)

        resp = self.client.responses.create(
            model=self.model,
            instructions=(
                "You are a document layout detector. "
                "Find all TABLES and FIGURES on the page. "
                "Return normalized bounding boxes (x0,y0,x1,y1) in [0,1]. "
                "Include nearest caption text if visible; else ''. "
                "Only include objects that are clearly tables or figures."
            ),
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Detect tables and figures on this page."},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            text={
                "format": {
                    "type": "json_schema",
                    "name": DETECT_SCHEMA["name"],
                    "schema": DETECT_SCHEMA["schema"],
                    "strict": True,
                }
            },
        )

        out = resp.output_text.strip()
        return json.loads(out)
