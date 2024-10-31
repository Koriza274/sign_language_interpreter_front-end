import pandas as pd
import os


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile, HTTPException

from dm_api.prediction_function import predict_image



#to run this thing: uvicorn dm_api.api:app --reload --port 8333

#!!!!!!!!!!  pip install python-multipart !!!!!!!!!!


app = FastAPI()

#recommended:

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


#to request upload
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        #giving the directory for the uploaded file
        os.makedirs("dm_api/uploads", exist_ok=True)
        file_location = f"dm_api/uploads/{file.filename}"

        #creating the file:
        contents = file.file.read()
        with open(file_location, 'wb') as f:
            f.write(contents)

    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        try:
            #prediction using directory:
            letter = predict_image(file_location)
        except Exception:
            raise HTTPException(status_code=500, detail='Prediction failed')
        finally:
            try:
                #removing the created file
                os.remove(file_location)
            except Exception:
                print(f"Error deleting file: {e}")
            file.file.close()


    return {"message": f"This is {letter}"}




@app.get("/")
def root():
    return {'greeting': 'Hello'}
