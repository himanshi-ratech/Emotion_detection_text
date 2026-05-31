import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet = True)
stop_words= set(stopwords.words('english'))

EMOTION_MAP= {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

EMOTION_CONFIG={
    'sadness':{'emoji':'😢','color':'#4A90D9', 'bg':'#EBF4FF', 'bar':'#4A90D9'},
     'joy':      {'emoji': '😄', 'color': '#F5A623', 'bg': '#FFF8EC', 'bar': '#F5A623'},
    'love':     {'emoji': '❤️',  'color': '#E8507A', 'bg': '#FFF0F4', 'bar': '#E8507A'},
    'anger':    {'emoji': '😠', 'color': '#E84040', 'bg': '#FFF0F0', 'bar': '#E84040'},
    'fear':     {'emoji': '😨', 'color': '#8B5CF6', 'bg': '#F5F0FF', 'bar': '#8B5CF6'},
    'surprise': {'emoji': '😲', 'color': '#10B981', 'bg': '#F0FDF9', 'bar': '#10B981'},
}

@st.cache_resource
def load_model():
    with open('emotion_model.pkl','rb') as f:
        model = pickle.load(f)
    with open ('vectorizer.pkl', 'rb') as f:
        vectorizer=  pickle.load(f)
    return model,vectorizer

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    keep = {'not', 'no', 'never', 'nor', 'neither'}
    tokens = [w for w in tokens if (w not in stop_words or w in keep) and len(w) > 1]
    # tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

def predict(text, model, vectorizer):
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return EMOTION_MAP[pred], proba

st.set_page_config(
    page_title="Emotion Detector",
    layout='centered'
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
 
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #1a1a1a;
    margin-bottom: 0.2rem;
    line-height: 1.15;
}
.hero-sub {
    font-size: 1rem;
    color: #666;
    margin-bottom: 2rem;
}
.result-card {
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    border: 1px solid rgba(0,0,0,0.06);
}
.result-emotion {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-bottom: 0.25rem;
}
.result-conf {
    font-size: 0.9rem;
    color: #888;
    margin-bottom: 1.25rem;
}
.bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    margin-bottom: 3px;
    color: #444;
}
.bar-outer {
    background: #f0f0f0;
    border-radius: 99px;
    height: 8px;
    margin-bottom: 10px;
    overflow: hidden;
}
.bar-inner {
    height: 8px;
    border-radius: 99px;
}
.divider { border: none; border-top: 1px solid #eee; margin: 1.25rem 0; }
 
.stTextArea textarea {
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    border: 1.5px solid #e0e0e0 !important;
    padding: 14px !important;
}
.stButton > button {
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.95rem;
    padding: 0.55rem 1.75rem;
    background: #1a1a1a;
    color: white;
    border: none;
}
.stButton > button:hover { opacity: 0.85; color: white; }
</style>          
""", unsafe_allow_html= True)

# load model
try: 
    model, vectorizer = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model is not loaded: {e}\n\nCheck if emotion_model.pkl aur vectorizer.pkl are in same folder.")

st.markdown('<div class="hero-title">Emotion Detector</div>', unsafe_allow_html=True)

st.markdown('<div class="hero-sub">Type any text — see the emotion behind it.</div>', unsafe_allow_html=True)

# Example buttons
EXAMPLES = [
    "I am so happy today!",
    "I miss you so much",
    "This makes me really angry",
    "I'm scared of what happens next",
    "I love you with all my heart",
    "Wow, I did not see that coming!"
]

st.markdown("**Try an example:**")
cols = st.columns(3)
for idx, ex in enumerate(EXAMPLES):
    label = ex[:26] + ("…" if len(ex) > 26 else "")
    if cols[idx % 3].button(label, key=f"ex_{idx}"):
        st.session_state['input_text'] = ex
 
# Text input
user_input = st.text_area(
    "Your text",
    value=st.session_state.get('input_text', ''),
    placeholder="e.g. I feel so grateful for everything in my life...",
    height=110,
    label_visibility="collapsed"
)
 
analyze_btn = st.button("Analyze emotion", use_container_width=False)
 
# Prediction
# REPLACE karo pura prediction block with this:
if analyze_btn and user_input.strip() and model_loaded:
    with st.spinner("Analyzing..."):
        emotion, proba = predict(user_input, model, vectorizer)
        cfg = EMOTION_CONFIG[emotion]

    confidence = proba[list(EMOTION_MAP.values()).index(emotion)] * 100

    sorted_emotions = sorted(
        [(EMOTION_MAP[i], p) for i, p in enumerate(proba)],
        key=lambda x: x[1], reverse=True
    )

    # Header card alag
    st.markdown(f"""
    <div class="result-card" style="background:{cfg['bg']}">
        <div class="result-emotion" style="color:{cfg['color']}">{cfg['emoji']} {emotion.capitalize()}</div>
        <div class="result-conf">Confidence: {confidence:.1f}%</div>
        <hr class="divider">
    """, unsafe_allow_html=True)

    # Bars alag alag render karo
    for emo, prob in sorted_emotions[:2]:
        pct = prob * 100
        ecfg = EMOTION_CONFIG[emo]
        st.markdown(f"""
        <div class="bar-label">
            <span>{ecfg['emoji']} {emo}</span>
            <span>{pct:.1f}%</span>
        </div>
        <div class="bar-outer">
            <div class="bar-inner" style="width:{pct:.1f}%;background:{ecfg['bar']}"></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sirf top emotion
    # top_emo, top_prob = sorted_emotions[0]
    # pct = top_prob * 100
    # ecfg = EMOTION_CONFIG[top_emo]

    # st.markdown(f"""
    # <div class="bar-label">
    #     <span>{ecfg['emoji']} {top_emo}</span>
    #     <span>{pct:.1f}%</span>
    # </div>
    # <div class="bar-outer">
    #     <div class="bar-inner" style="width:{pct:.1f}%;background:{ecfg['bar']}"></div>
    # </div>
    # """, unsafe_allow_html=True)
    

    # Card close
    st.markdown("</div>", unsafe_allow_html=True)