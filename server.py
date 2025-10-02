import os, io, json, tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=True)
app = FastAPI(title="Gopher Eye API")

from pipeline import classify_image, USE_MOCK
from cli.crop_send import crop_resize_512
import traceback

@app.post("/classify")
async def classify(
    prompt: str = Form(..., description="User text instruction"),
    image: UploadFile = File(..., description="Leaf photo (jpg/png)")
):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, detail=f"Only JPEG/PNG supported, got {image.content_type!r}")

    in_tmp = out_tmp = None
    try:
        try:
            raw = await image.read()
            Image.open(io.BytesIO(raw)).verify()
        except Exception as e:
            print("[server] verify image failed:", repr(e))
            print(traceback.format_exc())
            raise HTTPException(400, detail=f"Invalid image file: {type(e).__name__}")

        try:
            in_fd, in_tmp = tempfile.mkstemp(suffix=".png")
            with os.fdopen(in_fd, "wb") as f:
                f.write(raw)
        except Exception as e:
            print("[server] write temp input failed:", repr(e))
            print(traceback.format_exc())
            raise HTTPException(500, detail=f"Write temp input failed: {type(e).__name__}")

        try:
            out_fd, out_tmp = tempfile.mkstemp(suffix="_512.png")
            os.close(out_fd)
            crop_resize_512(in_tmp, out_tmp)
        except Exception as e:
            print("[server] crop/resize failed:", repr(e))
            print(traceback.format_exc())
            raise HTTPException(500, detail=f"Crop failed: {type(e).__name__}")

        try:
            out = classify_image(prompt, out_tmp)
        except Exception as e:
            print("[server] inference failed:", repr(e))
            print(traceback.format_exc())
            raise HTTPException(500, detail=f"Inference failed: {type(e).__name__}")

        try:
            data = json.loads(out)
        except Exception:
            data = {"raw": out}
        return JSONResponse(data)

    finally:
        for p in (in_tmp, out_tmp):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
