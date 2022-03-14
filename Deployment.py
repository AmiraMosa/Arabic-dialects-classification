import uvicorn
from fastapi import FastAPI
import joblib
from DataPreProcessing import clean_tweet,normalize_char

app = FastAPI()

filename = 'ml_model.pkl'
tf_file = "tfidf.pkl"
ml_model = joblib.load(filename)
tfidf = joblib.load(open("tfidf.pkl", 'rb'))


def classify_dialect(text):
    text = clean_tweet(text)
    text = normalize_char(text)
    tf = tfidf.transform([text])
    prediction = ml_model.predict(tf)
    return prediction

map_dialects = {0:'IQ', 1:'LY', 2:'QA', 3:'PL',
       4:'SY', 5:'TN', 6:'JO', 7:'MA',
       8:'SA', 9:'YE', 10:'DZ', 11:'EG',
       12:'LB', 13:'KW', 14:'OM', 15:'SD',16: 'AE', 17:'BH'
       }


@app.get("/ml/{text}")
async def get_prediction(text: str):
    dialect = classify_dialect(text)
    dil = map_dialects[dialect[0]]
    return {f'{text}':f' is a  {dil} dialect'}


if __name__ =='__main__':
    uvicorn.run(app)
