import streamlit as st
from transformers import pipeline
from PIL import Image
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

st.set_page_config(
    page_title="AgroLens: Expert Diagnosis",
    page_icon="🌿",
    layout="wide"  
)



@st.cache_resource
def load_vision_model():
    """Core Vision Model: HuggingFace ViT"""
    pipe = pipeline("image-classification", model="wambugu71/crop_leaf_diseases_vit")
    return pipe

@st.cache_resource
def load_gemini_model():
    """Gemini Model for Chat & Remedy"""
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ Google API Key missing! Please check your .env file.")
        return None
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview", 
        temperature=0.7,
        convert_system_message_to_human=True 
    )
    return llm


def get_expert_remedy(disease, crop, llm):
    template = """
    You are an expert agriculturalist and plant doctor.
    
    The user has uploaded a photo of a {crop} leaf. 
    Our vision system has detected the condition: "{disease}".
    
    Please provide:
    1. A brief confirmation of what this disease looks like.
    2. A step-by-step recommended treatment (organic or chemical).
    3. A preventative measure for the future.
    
    Please keep it simple for a farmer to understand. Please follow the steps and make it under 75 words.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    response = chain.invoke({
        "crop": crop, 
        "disease": disease
    })
    
    content = response.content
    

    if isinstance(content, list):
       
        return content[0].get('text', '')
        
   
    return content

def parse_prediction(label):
    known_crops = ["Corn", "Potato", "Rice", "Wheat"]
    detected_crop = "Unidentified"
    clean_label = label.replace("___", " ").replace("_", " ")
    for crop in known_crops:
        if crop.lower() in clean_label.lower():
            detected_crop = crop
            break 
    detected_disease = clean_label.replace(detected_crop, "").strip() if detected_crop != "Unidentified" else clean_label
    return detected_crop, detected_disease.title()



# SIDEBAR 
st.sidebar.title("🌿 AgroLens Menu")
app_mode = st.sidebar.radio("Go to:", ["🍃 Leaf Diagnosis", "💬 AI Agri-Chatbot"])

#  MODE 1: LEAF DIAGNOSIS 
if app_mode == "🍃 Leaf Diagnosis":
    st.title("🌿 AgroLens: Plant Doctor")
    st.markdown("Upload a photo of a crop leaf to detect diseases and get expert cures.")
    st.markdown("### 🌱Supported Crops")
    st.markdown("""
    The system is optimized for the following crops:
    * Rice
    * Wheat
    * Corn
    * Potato
    """)
    
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", width=300)
        
        if st.button('Analyze Plant'):
            viz_pipe = load_vision_model()
            gemini_llm = load_gemini_model()
            
            if gemini_llm:
                with st.spinner('Scanning leaf...'):
                    results = viz_pipe(image)
                    top = results[0]
                    final_crop, final_disease = parse_prediction(top['label'])
                    confidence = top['score']
                    remedy = get_expert_remedy(final_disease, final_crop, gemini_llm)
                    
                    st.divider()
                    col1, col2 = st.columns(2)
                    col1.metric("Detected Crop", final_crop)
                    col2.metric("Condition", final_disease, delta="Healthy" if "Healthy" in final_disease else "-Infected")
                    
                    st.progress(confidence, text=f"AI Confidence: {confidence:.1%}")
                    
                    if "Healthy" in final_disease:
                        st.success(f"**Analysis:** {remedy}")
                    else:
                        st.error(f"**Confirmed Disease:** {final_disease}")
                        st.info(f"**Prescription:** {remedy}")
                    
                    
                    with st.expander("See Knowledge Base Match"):
                        st.write(f"**Lookup Key:** {final_crop.lower()} {final_disease.lower()}")
                        st.write(f"**Vision Model:** wambugu71/crop_leaf_diseases_vit")
                        st.json(results)

# MODE 2: AI CHATBOT 
elif app_mode == "💬 AI Agri-Chatbot":
    st.title("💬 AgroBot Assistant")
    st.caption("Ask me anything about farming, pests, soil health, or fertilizers ")

   
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

   
    user_query = st.chat_input("Ask your farming question here...")
    
    if user_query:
        
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
       
        llm = load_gemini_model()
        if llm:
            with st.chat_message("assistant"):
                with st.spinner("AgroBot is thinking..."):
                   
                    system_prompt = """
                    You are AgroBot, an expert AI Agricultural Consultant. 
                    Answer questions related to farming, crops, pests, soil, and weather.
                    If a user asks about non-farming topics, politely steer them back to agriculture.
                    Keep your answers practical, concise, and easy for farmers to understand.
                    """
                    
                   
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{input}")
                    ])
                    
                  # 1. Add the parser to the end of your chain
                    chain = prompt | llm | StrOutputParser() 
                    
                    # 2. Invoke now returns a clean string directly (no need for .content)
                    response_text = chain.invoke({
                        "history": st.session_state.chat_history,
                        "input": user_query
                    })
                    
                    # 3. Print and save the clean string
                    st.markdown(response_text)
            
            st.session_state.chat_history.append(AIMessage(content=response_text))
st.sidebar.markdown("---")
st.sidebar.caption("Made by Team AgroLens | Samsung Innovation Campus")
