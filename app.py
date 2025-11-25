from fastapi import FastAPI, File, UploadFile, HTTPException
import io
from PIL import Image
from helper import predict_image


app = FastAPI(title="Car damage prediction", version="1.0")


@app.get("/")
def Status():
    return {"message" : "The api is live and running"}

@app.post("/predict")
async def predict_damage(image : UploadFile = File(...)):
    try:
        image = await image.read()
        image = Image.open(io.BytesIO(image)).convert("RGB")
        result = predict_image(image)
        return {"prediction" : result}
    except:
        raise HTTPException(status_code=400, detail="Something wrong with the image!")