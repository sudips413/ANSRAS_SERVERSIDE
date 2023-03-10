
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from uuid import UUID
from os.path import splitext
from nepalimodel.predict import predict_from_speech
from abstractive import abstractive_predict
from pydub import AudioSegment
from pythonfiles.main import get_summary_from_text_file
from pythonfiles import tokenizer
from pythonfiles import ranker
from pydantic import BaseModel
import os 
import subprocess 
from pydub import AudioSegment
from fastapi.responses import HTMLResponse
from nepalimodel import load_model
#####################import for custom model############################################
from ownmodel.predict import get_transcript
from ownmodel.configs import UNQ_CHARS
app = FastAPI()
##############################3

origins = [
    "*",
    
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    
    
# endpoint for textinput
class text(BaseModel):    
    texts: str
    
@app.get("/")
async def Hello():
    return "Hello World"
@app.post("/input-text")
async def create_upload_text(data: text):
    try:
        with open('static/input-text/input.txt', 'w',encoding="utf-8") as f:
            f.write(data.texts)
        filepath='static/input-text/input.txt'
        summary=get_summary_from_text_file(filepath)
        os.remove(filepath)
        return summary
    except:
        return "fail"
        
## load the model and processor 
@app.post("/loadmodel")  
async def loadthemodels():
    load_model.loadModelInitial()
    abstractive_predict.load_model()
    return True
    
#endpoint for fileinput-text
############################################################################################################
@app.post("/text")
async def create_upload_file(text: UploadFile = File(...)):
    try:       
        file_location = f"static/text/{uuid.uuid1()}{text.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(text.file.read())
            
        summary=get_summary_from_text_file(file_location)
        # print(summary)
        os.remove(file_location)
        return summary
        # return JSONResponse(content={"summary": summary})
         
        
    except:
        return "fail"


#endpoint for audioinput
##########################################################################################################
      
@app.post("/audio")
def create_upload_file(audio: UploadFile = File(...)):
    try:      
        ext=audio.filename.split('.').pop()        
        file_location = f"static/audio/{uuid.uuid1()}{audio.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(audio.file.read())   
        if ext == 'wav' or ext == 'flac':         
            transcript=predict_from_speech(file_location)
            os.remove(file_location)
        else:
            dest_path=f'static/audio/{uuid.uuid1()}coverted.flac'    
            command = f'ffmpeg -i {file_location} {dest_path}'
            subprocess.call(command,shell=True)  
            transcript=predict_from_speech(dest_path)
            os.remove(dest_path)
            os.remove(file_location)
        return transcript
    except:
        return "fail"
    
    
@app.post("/audio_live")
async def create_upload_file(audio: UploadFile = File(...)):    
    try:
        ext=audio.filename.split('.').pop()
        file_location = f"static/audio/{uuid.uuid1()}{audio.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(audio.file.read())
        dest_path=f'static/audio/{uuid.uuid1()}testme.flac'    
        command = f'ffmpeg -i {file_location} {dest_path}'
        subprocess.call(command,shell=True)    
        transcript=predict_from_speech(dest_path)
        if os.path.exists(dest_path):    
            print(dest_path)
        os.remove(dest_path)
        os.remove(file_location)
        return transcript
    except:
        return "couldnot handle the request, Try Again!"
    
    # return audio  
@app.post("/audio_live_own")
async def create_upload_file(audio: UploadFile = File(...)):
    # try:    
        ext=audio.filename.split('.').pop()
        file_location = f"static/audio/{uuid.uuid1()}{audio.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(audio.file.read())
        dest_path=f'static/audio/{uuid.uuid1()}testme.flac'    
        command = f'ffmpeg -i {file_location} {dest_path}'
        subprocess.call(command,shell=True)    
        transcript= get_transcript(dest_path)
        os.remove(dest_path)
        os.remove(file_location)
        return transcript     
    # except:
    #     return "couldnot handle the request, Try Again!"
@app.post("/abstract")
async def create_upload_text(data: text):    
    with open('static/input-text/input.txt', 'w',encoding="utf-8") as f:
        f.write(data.texts)
    filepath='static/input-text/input.txt'
    summary=abstractive_predict.abstractive_summarization_from_file(filepath)
    os.remove(filepath)
    return summary
    
        

    
       
    