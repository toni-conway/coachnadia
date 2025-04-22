import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

import folium
from streamlit_folium import st_folium
from folium import plugins

import geocoder
import geopy
from geopy.geocoders import Nominatim

from sklearn.metrics.pairwise import euclidean_distances

from datetime import date


#st.set_page_config(layout="wide") # Page expands to full width

#######################################################
# Initialize session state
#######################################################

# Q-CHAT Questions
Questions = [
    "Does your child look at you when you call his/her name?",
    "How easy is it for you to get eye contact with your child?",
    "Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)",
    "Does your child point to share interest with you? (e.g. pointing at an interesting sight)",
    "Does your child pretend? (e.g care for dolls, talk on a toy phone)",
    "Does your child follow where you're looking?",
    "If you or someone in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g stroking hair, hugging them)",
    "Would you describe your child's first word?",
    "Does your child use simple gestures? (e.g wave goodbye)",
    "Does your child stare at nothing with no apparent purpose?"
    ]

# Q-CHAT Options
Q_Options_List = [["Always", "Usually", "Sometimes", "Rarely", "Never"],
    ["Very easy", "Quite easy", "Quite difficult", "Very difficult", "Impossible"],
    ["Many times a day", "A few times a day", "A few times a week", "Less than once a week", "Never"],
    ["Very typical", "Quite typical", "Slighly typical", "Very unusual", "My child doesn't speak"],        
    ]

#Q-CHAT Question & Option Mapping
Q_Options = [
    Q_Options_List[0], Q_Options_List[1], Q_Options_List[2], Q_Options_List[2], Q_Options_List[2], 
    Q_Options_List[2], Q_Options_List[0], Q_Options_List[3], Q_Options_List[2], Q_Options_List[2]
    ]

# Symptoms Dictionary
Symptoms_dict = {
    1 : "Abnormalities in eye contact",
    2 : "Deficits in nonverbal communication understanding",
    3 : "Difficulties in sharing imaginative play",
    4 : "Poorly integrated nonverbal communication",
    5 : "Deficits in social-emotional reciprocity",
    6 : "Poorly integrated verbal and nonverbal communication",
    7 : "Highly restricted, fixated interests that are abnormal in intensity or focus"
    }

Symptoms_map = [1,1,2,2,3,4,5,6,6,7]   #Q-CHAT Questions to Symptoms Mapping

# Symptoms Infographic
Symptoms_Info = {
    1  : ["A1-A2.png"],
    2  : ["A3-A4.png"],
    3  : ["A5.png"],
    4  : ["A6 1of2.png","A6 2of2.png"],
    5  : ["A7 1of2.png","A7 2of2.png"],
    6  : ["A8-A9 1of2.png",'A8-A9 2of2.png'],
    7  : ["A10.png"],
    90 : ['TD.png'],
    91 : ['Parenting 1of2.png','Parenting 2of2.png'],
    92 : ['TD Parenting 1of2.png','TD Parenting 2of2.png'],
    }


# Header
st.image("images/ASD Header Option 2 (with kids).png")
#st.title("Quantitative Checklist for Autism in Toddlers (Q-CHAT)")

st.write("""    
    **Coach Nadia** utilizes Q-CHAT-10 (Quantitative Checklist for Autism in Toddlers) with proprietary machine learning algorithm to screen toddlers for signs of autism. 
    **Typically applicable to children aged between 18 and 48 months (less than 4 years old)**, the Q-CHAT-10 serves as a tool for 
    identifying children who may benefit a more comprehensive assessment for potential Autism Spectrum Disorders.
    """)

st.warning("""
        **Important Disclaimer:**\n
        This tool is not intended to provide an official diagnosis and is merely a screening resource to assess the potential need for further professional diagnostic evaluation.
        """, icon="⚠️")

# First Initialization
if "page_no" not in st.session_state:
    st.session_state.page_no      = 0
    st.session_state.pred         = 0
    st.session_state.qchat_result = 0
    st.session_state.qchat_resp   = []
    st.session_state.bg_result    = []
    st.session_state.model_result = ""
    st.session_state.child_name   = ""
    st.session_state.old_loc      = ""
    st.session_state.d_reco       = pd.DataFrame()
    st.session_state.t_reco       = pd.DataFrame()
    st.session_state.d_show       = True
    st.session_state.t_show       = True
    st.session_state.locgeo       = {"latitude": 0, "longitude": 0}
    st.session_state.error_flag   = False


#######################################################
# Function Name: load_pickle
# Description  : load pickle file
#######################################################
@st.cache_resource
def load_pickle():
    model = pickle.load(open('ada_hyper_f2.pkl', 'rb'))
    return model


#######################################################
# Function Name: get_yt_videos
# Description  : read YouTube Video Pool
#######################################################
@st.cache_data
def get_yt_videos():
    df = pd.read_csv('streamlit_yt_vids.csv', on_bad_lines='skip')
    video_dict = df.groupby('qid').apply(lambda group: dict(zip(group['id'], group['title']))).to_dict()
    return video_dict


#######################################################
# Function Name: format_phone_number
# Description  : Format PH Phone numbers
#######################################################
def format_phone_number(phone):

    if phone == "":
        formatted_number = ""
    else:
        phone_str = str(round(phone))
        if phone_str[0] == '9':
            formatted_number = f"0{phone_str[:3]} {phone_str[3:6]} {phone_str[6:]}"
        elif phone_str[0] == '2':
            formatted_number = f"(02) {phone_str[1:5]} {phone_str[5:]}"
        else:
            formatted_number = f"(0{phone_str[:2]}) {phone_str[2:5]} {phone_str[5:]}" 

    return formatted_number


#######################################################
# Function Name: get_distances
# Description  : Compute for Euclidean Distance
#######################################################
def get_distances(x,y):
    euclidean_dist =  euclidean_distances(x.values.reshape(1, -1), y.values.reshape(1, -1)).flatten()[0]
    return euclidean_dist


#######################################################
# Function Name: get_loc_distances
# Description  : Compute distance from location based on current location
#######################################################
def get_loc_distances(data, seed):
    seed_df = pd.Series({"Latitude": seed['latitude'], "Longitude": seed['longitude']})

    df = data.copy()
    df['distance'] = df.apply(lambda x: get_distances(x[["Latitude", "Longitude"]], seed_df), axis=1)
    df = df[df.distance < 0.03].sort_values("distance").reset_index()

    return df


#######################################################
# Function Name: get_dev_pedia
# Description  : read debelopment pediatrician pool
#######################################################
@st.cache_data
def get_dev_pedia():
    df = pd.read_csv('ph-dev-ped-2023-pool.csv', on_bad_lines='skip')
    df = df.fillna("")
    df = df.drop_duplicates(subset=['Developmental Pediatrician','Address'])
    df['Contact'] = df['Contact'].apply(format_phone_number)

    return df


#######################################################
# Function Name: get_therapy_center
# Description  : read therapy center pool
#######################################################
@st.cache_data
def get_therapy_center():
    df = pd.read_csv('ph-therapy-center-2023-pool.csv', on_bad_lines='skip')
    df = df.fillna("")
    df = df.drop_duplicates(subset=['Therapy Center','Address'])
    df['Contact'] = df['Contact'].apply(format_phone_number)

    return df


#######################################################
# Function Name: create_map
# Description  : create folium map
#######################################################
def create_folium_map(doc_data, therapy_data, seed):
    ph_map = folium.Map(location=[seed['latitude'], seed['longitude']], zoom_start=14, control_scale=True)

    # Current locations
    folium.Marker(location=[seed['latitude'], seed['longitude']], 
                icon=folium.Icon(color='red', icon='map-marker'),
                tooltip='Your Location').add_to(ph_map)

    # plot doctors locations
    if st.session_state.d_show == True:
        doc_data.apply(lambda row: folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                icon=folium.Icon(color='green', icon='plus-sign'),
                tooltip=row['Developmental Pediatrician']).add_to(ph_map), axis=1)
    
    # plot therapy center locations
    if st.session_state.t_show == True:
        therapy_data.apply(lambda row: folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                #icon=folium.Icon(color='yellow', icon='education'),
                icon=folium.Icon(color='blue', icon='flag'),
                tooltip=row['Therapy Center']).add_to(ph_map), axis=1)

    plugins.MiniMap(toggle_display=True).add_to(ph_map)
        
    # display map    
    st_folium(ph_map, width=725)


#######################################################
# Function Name: get_yt_videos
# Description  : read YouTube Video Pool
#######################################################
def predict_data():
#        ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons',
#        'Sex', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test',
#        'Ethnicity_Latino', 'Ethnicity_Native Indian',
#        'Ethnicity_Others', 'Ethnicity_Pacifica', 'Ethnicity_White European',
#        'Ethnicity_asian', 'Ethnicity_black', 'Ethnicity_middle eastern',
#        'Ethnicity_mixed', 'Ethnicity_south asian']

    input_resp = st.session_state.qchat_resp.copy()
    model      = st.session_state.model

    bg_resp     = [0] * 15
    bg_resp[0]  = st.session_state.bg_result[0]                                          # Age_Mons
    bg_resp[1]  = 1 if st.session_state.bg_result[1] == "Male"            else 0         # Sex
    bg_resp[2]  = 1 if st.session_state.bg_result[3] == True              else 0         # Jaundice
    bg_resp[3]  = 1 if st.session_state.bg_result[4] == True              else 0         # Family_mem_with_ASD
    bg_resp[4]  = 1 if st.session_state.bg_result[5] == "Family Member"   else 0         # Who completed the test
    bg_resp[5]  = 1 if st.session_state.bg_result[2] == "Latino"          else 0         # Ethnicity_Latino
    bg_resp[6]  = 1 if st.session_state.bg_result[2] == "Native Indian"   else 0         # Ethnicity_Native Indian
    bg_resp[7]  = 1 if st.session_state.bg_result[2] == "Others"          else 0         # Ethnicity_Others
    bg_resp[8]  = 1 if st.session_state.bg_result[2] == "Pacifica"        else 0         # Ethnicity_Pacifica
    bg_resp[9]  = 1 if st.session_state.bg_result[2] == "White European"  else 0         # Ethnicity_White European
    bg_resp[10] = 1 if st.session_state.bg_result[2] == "Asian"           else 0         # Ethnicity_asian
    bg_resp[11] = 1 if st.session_state.bg_result[2] == "Black"           else 0         # Ethnicity_black
    bg_resp[12] = 1 if st.session_state.bg_result[2] == "Middle Eastern"  else 0         # Ethnicity_middle eastern
    bg_resp[13] = 1 if st.session_state.bg_result[2] == "Mixed"           else 0         # Ethnicity_mixed
    bg_resp[14] = 1 if st.session_state.bg_result[2] == "South Asian"     else 0         # Ethnicity_south asian

    input_resp.extend(bg_resp)
    input_data = np.array(input_resp).reshape(1, -1)
    prediction = model.predict(input_data)

    # 0 - Typically Developing
    # 1 - Mild Symptoms
    # 2 - Severe Symptoms

    return prediction

    
#######################################################
# Function Name: evaluate_Qchat
# Description  : Evaluate Qchat Scores
#######################################################
def evaluate_Qchat(Responses, Q_Options):
    Answers = [0] * 10
    NoResponse = [i+1 for i, x in enumerate(Responses) if x is None]    # check for blank anwers

    for i, Response in enumerate(Responses):
        if i == 9: #Question 10
            Answers[i] = 1 if Response in Q_Options[i][0:3] else 0
        else:
            Answers[i] = 1 if Response in Q_Options[i][2:] else 0

    st.session_state.qchat_text   = Responses
    st.session_state.qchat_resp   = Answers
    st.session_state.qchat_result = sum(Answers)

    return NoResponse


#######################################################
# Function Name: demographics_questionnaire
# Description  : Show demographics questionnaire
#######################################################
def demographics_questionnaire():
    st.markdown("Please answer the following questions about your child")

    with st.form("Background Questionnaire"):
        st.info("##### Background Questions")
        st.session_state.child_name = st.text_input("Child Name", max_chars=50)
        
        B1 = st.number_input("B1. What age is your child (in months) [0-48]?", 0, 48, help="Select the age of your child in months.")
        B2 = st.radio("B2. What is your child's gender?", ["Male", "Female"], help="Choose the gender of your child.", horizontal=True, index=None)
        B3 = st.radio("B3. What is your child's ethnicity?", ["Asian", "Black", "Hispanic", "Latino", "Middle Eastern", "Native Indian", "Pacifica",
                                        "South Asian", "White European", "Mixed", "Others"],
                        help="Select the ethnicity that best describes your child.", horizontal=True, index=None)
        B4 = st.checkbox("B4. Has your child experienced jaundice?", help="Check this box if your child experienced jaundice")
        B5 = st.checkbox("B5. Do any of your child's immediate family members (siblings or parents) have a diagnosis of autism?", 
                          help="Check this box if any immediate family members have been diagnosed with autism")
        B6 = st.radio("B6. Who completed the test?", ["Family Member", "Health Care Professional"],
                        help="Select the person who administered the test.", horizontal=True, index=None)

        if B1 == 0: B1 = None
        Responses = [B1, B2, B3, B4, B5, B6]

        if st.form_submit_button(" Next Page "):
            NoResponse = [i+1 for i, x in enumerate(Responses) if x is None] 
            NoResponseText = ', '.join("B"+str(item) for item in NoResponse)
            
            if st.session_state.child_name == "": 
                NoResponseText = "Child Name, " + NoResponseText
 
            if len(NoResponse) > 0:
                st.error(f"Please answer questions {NoResponseText} then click Next Page button", icon="🚨")
                st.session_state.page_no = 0
            else:
                st.session_state.bg_result = Responses
                st.session_state.page_no += 1


#######################################################
# Function Name: qchat_questionnaire
# Description  : Show Q-CHAT-10 questionnaire
#######################################################
def qchat_questionnaire():
    st.markdown("Please answer the following questions about your child")

    with st.form("Q-CHAT Questionnaire"):
        st.info("##### Q-CHAT-10 Questions")

        Responses = [""] * 10
        for i, Question in enumerate(Questions):
            Responses[i] = st.radio(f"Q{i+1}. {Question}", Q_Options[i], index=None, horizontal=True) 

        if st.form_submit_button("Submit"):
            NoResponse = evaluate_Qchat(Responses, Q_Options)
            NoResponseText = ', '.join("Q"+str(item) for item in NoResponse)  

            if len(NoResponse) > 0:
                st.error(f"Please answer questions {NoResponseText} then resubmit", icon="🚨")
                st.session_state.page_no = 1
            else:
                st.session_state.page_no   += 1
    

#######################################################
# Function Name: show_results
# Description  : Show Results
#######################################################
def show_results():

    st.session_state.pred = predict_data()
     
    if st.session_state.pred == 0:      # 0 - Typically Developing
        st.session_state.model_result = f"""
The child is healthy and exhibiting appropriate behavior for their age. 
There is no need for further ASD screening.
 """
        
    elif st.session_state.pred == 1:    # 1 - Mild Symptoms
        st.session_state.model_result = f"""
The child has some risk factors and symptoms that may be associated to ASD. 
Keep monitoring your child closely and consider ASD screening by a medical professional.
"""
        
    else:                               # 2 - Severe Symptoms
        st.session_state.model_result = f"""
The child has several risk factors and symptoms that are usually associated to ASD. 
ASD screening by a medical professional is strongly recommended.
"""
    
    st.info(st.session_state.model_result)
        
        
#######################################################
# Function Name: get_yt_link
# Description  : format youtube link
#######################################################
def get_yt_link(video_id):
    return "https://www.youtube.com/watch?v="+video_id

#######################################################
# Function Name: show_yt_video
# Description  : Show Results
#######################################################
def show_yt_video(video_dict):

    st.info("Recommended videos")

    for video_id, title in video_dict.items():
        yt_column  = st.columns([0.5, 0.5])
        with yt_column[0]: st.video(get_yt_link(video_id))
        with yt_column[1]: st.write(title)

        # st.caption(title)
        # st.video(get_yt_link(video_id))


#######################################################
# Function Name: download_result
# Description  : download Results
#######################################################
def download_result():

    bg_resp  = st.session_state.bg_result
    B4 = "Yes" if bg_resp[3] == True else "No"
    B5 = "Yes" if bg_resp[4] == True else "No"

    qn_resp   = st.session_state.qchat_text.copy()
    mo_resp   = st.session_state.model_result
    chld_name = st.session_state.child_name
    assess_dt = date.today().strftime("%Y-%m-%d")

    result_text = f"""
Child Name: {chld_name}
Assessment Date: {assess_dt}

Background Responses
-----------------------------------------------------------------------------------------------------------------
B1. What age is your child (in months) [0-48]? {bg_resp[0]}
B2. What is your child's gender? {bg_resp[1]}
B3. What is your child's ethnicity? {bg_resp[2]}
B4. Has your child experienced jaundice? {B4}
B5. Do any of your child's immediate family members (siblings or parents) have a diagnosis of autism? {B5}
B6. Who completed the test? {bg_resp[5]}


Q-CHAT-10 Questionnaire Responses
-----------------------------------------------------------------------------------------------------------------
Q1. Does your child look at you when you call his/her name? {qn_resp[0]}
Q2. How easy is it for you to get eye contact with your child? {qn_resp[1]}
Q3. Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach) {qn_resp[2]}
Q4. Does your child point to share interest with you? (e.g. pointing at an interesting sight) {qn_resp[3]}
Q5. Does your child pretend? (e.g care for dolls, talk on a toy phone) {qn_resp[4]}
Q6. Does your child follow where you're looking? {qn_resp[5]}
Q7. If you or someone in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g stroking hair, hugging them) {qn_resp[6]}
Q8. Would you describe your child's first word? {qn_resp[7]}
Q9. Does your child use simple gestures? (e.g wave goodbye) {qn_resp[8]}
Q10. Does your child stare at nothing with no apparent purpose? {qn_resp[9]}


Child displays the following symptoms associated with ASD
-----------------------------------------------------------------------------------------------------------------
{Symptoms_List}


Recommendation:
-----------------------------------------------------------------------------------------------------------------
{mo_resp[1:]}
    """

    return result_text[1:]


#######################################################
# Function Name: process_location
# Description  : download Results
#######################################################
def process_location(dp_reco_df, tc_reco_df):
    if len(dp_reco_df) > 0 or len(tc_reco_df) > 0:

        map_column  = st.columns(2)
        with map_column[0]: st.session_state.d_show = st.toggle("Show Developmental Pediatrician",value=True)
        with map_column[1]: st.session_state.t_show = st.toggle("Show Therapy Center", value=True)

        create_folium_map(dp_reco_df, tc_reco_df, st.session_state.locgeo)

        if st.session_state.d_show == True:
            if len(dp_reco_df) == 0:
                st.info(f"No Developmental Pediatrician Found")
            else:
                st.dataframe(dp_reco_df[['Developmental Pediatrician','Address','Contact']], hide_index=True)

        if st.session_state.t_show == True:
            if len(tc_reco_df) == 0:
                st.info(f"No Therapy Center Found")
            else:
                #st.dataframe(tc_reco_df[['Therapy Center','Address','Contact','Link']], hide_index=True)

                st.data_editor(
                    tc_reco_df[['Therapy Center','Address','Contact','Link']],
                    column_config={
                        'Link': st.column_config.LinkColumn(
                            "Website",
                            help="click the link below",
                            max_chars=100, disabled=True
                        )},
                    hide_index=True,
                    )



#######################################################
# MAIN
#######################################################
st.session_state.model = load_pickle()

if st.session_state.page_no == 0:
    demographics_questionnaire()

elif st.session_state.page_no == 1:
    qchat_questionnaire()   
else:
    result_tab = st.tabs(["Results", "Recommendation", "Parenting Tips", "Locate Doctors & Theraphy Centers"])
    Symptoms_Ids = set(Symptoms_map[i] for i, x in enumerate(st.session_state.qchat_resp) if x == 1)

    Symptoms_List = "\n".join([f"{i+1}. {Symptoms_dict[x]}" for i, x in enumerate(Symptoms_Ids)])
    if Symptoms_List == "": Symptoms_List = "None"

    # --- Results --- #
    with result_tab[0]:
        show_results()
        st.download_button(label="Download Results", data=download_result(), file_name='Result.txt')

    # --- Recommendation --- #
    with result_tab[1]:

        video_dict = get_yt_videos()

        # ------- Mild/Severe Autism -----#
        if st.session_state.pred > 0:
            expand_section = st.toggle("Expand All")
            st.info("The child displays the following symptoms associated with ASD")

            for i, Symptoms_Id in enumerate(Symptoms_Ids):
                with st.expander(f"Symptom {i+1}: {Symptoms_dict[Symptoms_Id]}", expanded=expand_section):

                    for image_name in Symptoms_Info[Symptoms_Id]:
                        st.image('images/' + image_name)

                    if Symptoms_Id in video_dict.keys():
                        show_yt_video(video_dict[Symptoms_Id])

        else:
            # ------- Typically Developing -----#
            for image_name in Symptoms_Info[90]:
                st.image('images/' + image_name)

    # --- Parenting Tips --- #
    with result_tab[2]:
        # ------- Mild/Severe Autism -----#
        if st.session_state.pred > 0:
            Symptoms_Id = 91
            # subreddit_link = """
            # [Autism_Parenting](https://www.reddit.com/r/Autism_Parenting)\n
            # [AutisticParents](https://www.reddit.com/r/AutisticParents)
            # """

            st.info("Connect with Reddit communities to acquire valuable insights and shared experiences from parents and families raising children on the autism spectrum.")
            # st.markdown(subreddit_link)

            link_column  = st.columns(2)
            with link_column[0]: st.markdown("* [Autism_Parenting](https://www.reddit.com/r/Autism_Parenting)")
            with link_column[1]: st.markdown("* [AutisticParents](https://www.reddit.com/r/AutisticParents)")

        else:
            Symptoms_Id = 92

        for image_name in Symptoms_Info[Symptoms_Id]:
                st.image('images/' + image_name)

    # --- Locate Doctors --- #
    with result_tab[3]:
        devpedia_df = get_dev_pedia()
        therapy_df  = get_therapy_center()

        current_loc = st.text_input("Enter your current location", max_chars=100)

        if current_loc != "":
            if current_loc != st.session_state.old_loc:
                geolocator = Nominatim(user_agent="DSF C12 G3 Capstone")
                getLoc = geolocator.geocode(current_loc)
                st.session_state.old_loc = current_loc

                if hasattr(getLoc, 'address'):
                    st.session_state.locgeo = {"latitude": getLoc.latitude, "longitude": getLoc.longitude}
                    st.session_state.d_reco  = get_loc_distances(devpedia_df, st.session_state.locgeo)
                    st.session_state.t_reco  = get_loc_distances(therapy_df, st.session_state.locgeo)
                    st.session_state.error_flag = False
                else:
                    st.error(f"Current Location is not found.", icon="🚨")
                    st.session_state.error_flag = True

            if st.session_state.error_flag == False:
                process_location(st.session_state.d_reco, st.session_state.t_reco)



