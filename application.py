import pandas as pd
import numpy as np
import streamlit as st
import xgboost
import pickle
import base64
from PIL import Image
#from io import BytesIO
#import ssl

#ssl._create_default_https_context = ssl._create_unverified_context

def main():
	st.title('Default Predictor')
	st.subheader("Let us tell you if you're gonna default or not")
	uploadedFile = st.file_uploader('Upload data', type=['csv'],accept_multiple_files=False,key="fileUploader")
	if uploadedFile is not None:
		data = pd.read_csv(uploadedFile,delimiter=';', skiprows=0, low_memory=False)
		#data = data.iloc[:,2:]

		columns_drop = ['worst_status_active_inv','uuid']
		data = data.drop(columns_drop , axis = 1)

		model = pickle.load(open('finalized_model.sav', 'rb'))
		enc = pickle.load(open('encoder.pkl', 'rb'))
	
		for index in list(data.describe(include=['O']).columns):
			data[index] = enc.transform(np.array(data[index]).reshape(-1,1))
		
		predictions = model.predict_proba(data.iloc[:,1:])
		pred_default = []
		for pred in predictions:
			pred_default.append(pred[1])
		
		df = pd.DataFrame(list(pred_default))

		csvv = df.to_csv(index = False).encode()
		#b64 = base64.b64encode(csvv.encode()).decode() 
		b64 = base64.b64encode(csvv).decode()
		#href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
		href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
		st.markdown(href, unsafe_allow_html=True) 
if __name__ == '__main__':
	 main()