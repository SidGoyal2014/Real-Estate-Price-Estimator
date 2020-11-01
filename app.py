import streamlit as st
import joblib
import numpy as np
from flask import Flask,jsonify,request
import requests
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import os
import random

def main():
    st.title("Know the exact price of the house, if you are in Chennai")
    st.subheader("ML Based Real Estate Price Estimator")
    st.write("_______________________________________________________________________")
    
    st.sidebar.title("Know the exact price of the house, of you are in Chennai")
    st.sidebar.subheader("ML Based Real Estate Price Estimator")
    st.sidebar.write("__________________________________________________________")
    
    ##### First of all Display some text to give user a feel
    st.write("The Price of any Real Estate Property depends upon many factors like: ")
    st.write("1. Locality")
    st.write("2. Area")
    st.write("3. Distance from Mainroad")
    st.write("4. etc.")
    
    st.write("This ML software takes into account, various factors like Locality, Area, Quality score of rooms etc, for most accurate estimation.")
    
    # @st.cache(persist = True)
    def take_input():
        
        # Area selection
        area = st.sidebar.selectbox("Choose Area:",("Karapakkam","Adyar","Anna Nagar","Chrompet","KK Nagar","T Nagar","Velachery"))
        # st.write("The locality is : ",area)
        
        #"""
        ## def imag():
        ## Display the image
        #path = os.getcwd()
        ##path = path + "\Images" + "\\" + "Chennai\\"+ area
        #num = random.randint(1,5)
        #path = path + "\\" + str(num) + ".jpg"
        #print("path : ",path)
        #st.image(path,channels="RGB")
        #"""
        
        # Int SQFT area
        int_sqft = st.sidebar.number_input("Enter the area (in SQFT)",min_value=100)
        int_sqft = int(int_sqft)
        # st.write("The area is : ",int_sqft)
        
        # DIST_MAINROAD
        dist_mainroad = st.sidebar.number_input("Enter the distance from the mainroad (in KMs)",min_value=0.0)
        dist_mainroad = int(dist_mainroad)
        # st.write("The distance from the mainroad is : ",dist_mainroad)
        
        # Number of Bedrooms (N_BEDROOM)
        n_bedroom = st.sidebar.selectbox("The number of bedrooms:",(1,2,3,4,5))
        # st.write("The number of bedrooms: ",n_bedroom)
        
        # Number of Bedrooms (N_BATHROOM)
        n_bathroom = st.sidebar.selectbox("The number of bathrooms:",(1,2,3,4,5))
        # st.write("The number of bathrooms: ",n_bathroom)
        
        # Number of rooms (N_ROOMS)
        n_rooms = st.sidebar.selectbox("The number of Rooms:",(1,2,3,4,5,6,7,8,9,10))
        # st.write("Total number of rooms: ",n_rooms)
        
        # The sale condition (SALE_COND)
        sale_condition = st.sidebar.selectbox("Select the sale condition:",("AbNormal","AdjLand","Family","Normal Sale","Partial"))
        # st.write("The sale condition of the house: ",sale_condition)
        
        # Park Facil (PARK_FACIL)
        park_facing = st.sidebar.selectbox("Select if the house if park facing or not",("Yes","No"))
        # st.write("Park Facing : ",park_facing)
        
        # Buildtype (BUILDTYPE)
        buildtype = st.sidebar.selectbox("Select the buildtype:",("Commercial","House","Others"))
        # st.write("The buildtype is: ",buildtype)
        
        # utility (UTILITY)
        utility = st.sidebar.selectbox("Select the utility:",("AllPub","ELO","NoSeWa","NoSewr"))
        # st.write("The utility: ",utility)
        
        # Street type (STREET)
        street = st.sidebar.selectbox("Select the street:",("Gravel","No Access","Paved"))
        # st.write("The street is: ",street)
        
        # MZZONE
        mzzone = st.sidebar.selectbox("Select the mzzone:",("A","C","I","RH","RL","RM","If you don't have info about this"))
        # st.write("The mzzone: ",mzzone)
        
        if(mzzone == "If you don't have info about this"):
            # Seperate procedure
            print("There is a seperate procedure")
            
        # The quality score of rooms (QS_ROOMS)
        qs_rooms = st.sidebar.number_input("Enter the overall quality score of all rooms (0,5)",min_value=0.0,max_value=5.0)
        # st.write("The quality score of rooms: ",qs_rooms)
        
        # The quality score of bedrooms (QS_BEDROOM)
        qs_bedroom = st.sidebar.number_input("Enter the Quality score of bedrooms (0-5)",min_value=0.0,max_value=5.0)
        # st.write("The quality score of bedrooms: ",qs_bedroom)
        
        
        list_val = [[area,int_sqft,dist_mainroad,n_bedroom,n_bathroom,n_rooms,sale_condition,park_facing,buildtype,utility,street,mzzone,qs_rooms,qs_bedroom]]
        list_idx = ["AREA","INT_SQFT","DIST_MAINROAD","N_BEDROOM","N_BATHROOM","N_ROOM","SALE_COND","PARK_FACIL","BUILDTYPE","UTILITY","STREET","MZZONE","QS_ROOMS","QS_BEDROOM"]
        
        return list_val,list_idx
        
    list_val,list_idx = take_input()
    
    # print("List Value: ",list_val)
    # print("List Index: ",list_idx)
    
    @st.cache(persist = True)
    def dataframe_prep(list_val,list_idx):
        
        temp_df = pd.DataFrame(list_val, columns=list_idx)
        
        temp_df = pd.get_dummies(temp_df)
        
        ## print(temp_df)
        
        final_col = ['INT_SQFT', 'DIST_MAINROAD', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM','QS_ROOMS', 'QS_BEDROOM', 'AREA_Adyar','AREA_Anna Nagar', 'AREA_Chrompet', 'AREA_KK Nagar', 'AREA_Karapakkam','AREA_T Nagar', 'AREA_Velachery', 'SALE_COND_AbNormal','SALE_COND_AdjLand', 'SALE_COND_Family', 'SALE_COND_Normal Sale','SALE_COND_Partial', 'PARK_FACIL_No', 'PARK_FACIL_Yes','BUILDTYPE_Commercial', 'BUILDTYPE_House', 'BUILDTYPE_Others','UTILITY_AVAIL_AllPub', 'UTILITY_AVAIL_ELO', 'UTILITY_AVAIL_NoSeWa','UTILITY_AVAIL_NoSewr ', 'STREET_Gravel', 'STREET_No Access','STREET_Paved', 'MZZONE_A', 'MZZONE_C', 'MZZONE_I', 'MZZONE_RH','MZZONE_RL', 'MZZONE_RM']
        
        df_final= temp_df.reindex(columns=final_col,fill_value=0)
        
        poly = PolynomialFeatures(degree = 2)
        
        df_final = poly.fit_transform(df_final)
        
        # print(df_final)
        return df_final

    @st.cache(persist = True)
    def prediction(df_final):
        
        prediction = lr.predict(df_final)
        
        prediction = prediction[0]
        
        return prediction
    
    # Prepare the dataframe
    df_final = dataframe_prep(list_val,list_idx)
    
    # print(df_final)
    
    # Predct the price
    if(st.sidebar.button("Predict",key="Predict")):
        price = prediction(df_final)
        price = round(price,2)
        st.write("The price of the house is: ",price)
    
if __name__ == '__main__':
    lr = joblib.load("model.pkl")
    print("Model Loaded")
    model_columns = joblib.load("model_columns.pkl")
    print("Model Column Loaded")
    main()