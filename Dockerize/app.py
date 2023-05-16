from flask import *
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
import re

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def home():
	sentiment = ""
	if (request.method == "POST"):
		txt=(request.form["txt"])

		emj = emoji.distinct_emoji_list(txt)
		
		txt = clean_text(txt)
		# text_split = split_text(txt)
		emj = demojize(emj)

		context = txt+" ".join(emj)
		print(context)
		tfidf = pickle.load(open("tfidf.pkl", 'rb'))
		X = tfidf.transform([context]).toarray()

		filename = "LR_model.joblib"
		with open(filename, 'rb') as file:  
			LR_Model = pickle.load(file)		
			if(LR_Model.predict(X) == 0):
				sentiment = "positive 🙂"
			elif(LR_Model.predict(X) == 1):
				sentiment = "negative 🙁"
			elif(LR_Model.predict(X) == 2):
				sentiment = "natural"
			else:
				sentiment = "BiMani"
		
	return render_template("lr-form.html",sentiment = sentiment)


punctuation_signs = list("?:!.,;")
def clean_text(text):
    text=  emoji.demojize(text)
    text= re.sub(r'(:[!_\-\w]+:)', '', text)
    text= re.sub(r'@\w+', ' ', text)
    text =  text.replace("#", " ")
    text =  text.replace("-", " ")
    text =  text.replace("_", " ")
    
    for i in punctuation_signs:
         text = text.replace(i,'')
    return text

def split_text(sentence):
    tokens = sentence.strip().split(' ')
    return tokens

def demojize(liiist):
    z = []
    for item in liiist:
        z.append(emoji.demojize(item))
    return z




if __name__ == "__main__":
    app.run()
