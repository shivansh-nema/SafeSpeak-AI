import os
import json

import streamlit as st
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"

API_KEY = st.secrets["API_KEY"]
headers = {
    "authorization": API_KEY,
    "content-type": "application/json"
}

if not API_KEY:
    raise RuntimeError(
        "Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
    )

client = genai.Client(api_key=API_KEY)

RISK_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "risk_score": {
            "type": "NUMBER",
            "description": "Overall risk between 0 and 100. 0 = totally safe, 100 = extremely risky."
        },
        "risk_level": {
            "type": "STRING",
            "description": "Qualitative risk bucket.",
            "enum": ["low", "medium", "high", "critical"]
        },
        "categories": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of categories such as religion, caste, hate_speech, bullying, etc."
        },
        "explanations": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Short human-friendly explanations of why this text is risky or safe."
        },
        "suggested_rewrites": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Alternative polite / respectful rewrites of the original text."
        },
    },
    "required": ["risk_score", "risk_level", "suggested_rewrites"],
}

def call_gemini_for_text(text: str) -> dict:
    """
    Send raw text to Gemini and get back a structured JSON response
    containing risk_score, risk_level, categories, explanations, suggested_rewrites.
    """
    if not text or not text.strip():
        return None

    prompt = f"""
You are an AI ethics and safety assistant for social media, especially in diverse and sensitive contexts
like India and other multicultural societies.

Your job:
- Analyse the user text for risk of:
  - disrespecting or attacking religion, caste, culture, region, language or community
  - harassment, bullying, slurs or abusive language
  - gender, sexuality or minority hate
  - incitement to violence, self-harm or serious discrimination
- Be careful to respect free expression: disagreement and criticism are allowed,
  but you should still highlight if the tone is harsh or potentially hurtful.

Definitions:
- "risk_score": number 0–100. 0 means totally safe; 100 means extremely harmful.
- "risk_level": one of ["low", "medium", "high", "critical"].
- "categories": list of short labels, e.g. ["religion", "caste", "bullying"].
- "explanations": list of short bullet-point style sentences explaining your reasoning.
- "suggested_rewrites": list of alternative messages that keep the user's intent,
  but are more polite, respectful, and unlikely to hurt anyone's sentiments.

Very important:
- DO NOT invent new slurs or extra offensive content.
- DO NOT make the message harsher or more extreme.
- Focus on de-escalation, empathy, and respectful communication.

Now analyse this text and fill the JSON fields:

USER_TEXT:
\"\"\"{text.strip()}\"\"\"
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": RISK_SCHEMA,
        },
    )

    return json.loads(response.text)


def call_gemini_for_image(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    """
    Analyse an uploaded image (e.g., screenshot of a comment section) and return
    the same JSON structure: risk_score, risk_level, categories, explanations, suggested_rewrites.
    """
    if not image_bytes:
        return None

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type or "image/png",
    )

    prompt = """
You are an AI ethics and safety assistant for social media screenshots.

Steps:
1. Carefully read and transcribe any visible text in the image, especially comments,
   captions, replies, usernames and overlaid text.
2. Consider the screenshot as a bundle of social media comments / posts.
3. Evaluate the overall risk that this image's content could hurt someone's
   cultural, religious, caste, gender, regional or personal sentiments, or
   contain harassment, bullying, hate speech, or incitement.

Return a single JSON object with:
- "risk_score": number 0–100 (0 = safe, 100 = extremely harmful)
- "risk_level": one of ["low", "medium", "high", "critical"]
- "categories": list of short labels (e.g. ["religion", "caste", "bullying", "explicit_language"])
- "explanations": list of short sentences explaining the main concerns
- "suggested_rewrites": list of more respectful and polite alternative ways to express
  the main messages from these comments, while keeping the core meaning where possible.

Do NOT output anything except the JSON object.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[image_part, prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": RISK_SCHEMA,
        },
    )

    return json.loads(response.text)


def call_gemini_for_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> dict:
    """
    Analyse a spoken voice message (content + tone) and return the same JSON structure.

    The model should:
    - Transcribe what the user is saying.
    - Analyse meaning + tone (angry, mocking, sarcastic, calm, etc.)
    - Compute risk_score, risk_level, categories, explanations, suggested_rewrites.
    """
    if not audio_bytes:
        return None

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type=mime_type or "audio/wav",
    )

    prompt = """
You are an AI ethics and safety assistant for SPOKEN social media messages (voice notes).

Steps:
1. Accurately TRANSCRIBE the speech in the audio.
2. Analyse BOTH:
   - The CONTENT of what is said.
   - The TONE of the voice (angry, sarcastic, mocking, calm, threatening, etc.).
3. Decide how risky it would be to send this as a voice message or post it online,
   especially in sensitive contexts like religion, caste, gender, region, language,
   or towards specific communities or individuals.

Important:
- Someone can sound aggressive even if words are neutral.
- Sarcasm and mocking tone can make a message more hurtful.
- Do NOT invent new slurs or add extra offensive details that are not spoken.

Return a JSON object with:
- "risk_score": number 0–100 (0 = safe, 100 = extremely harmful)
- "risk_level": one of ["low", "medium", "high", "critical"]
- "categories": list of short labels (e.g. ["religion", "caste", "bullying", "threat"])
- "explanations": list of short sentences explaining the risk, including tone-related points
- "suggested_rewrites": list of respectful, polite alternative ways to say the same thing
  as a text message. Keep the user's main intent, but make it calm and non-hurtful.

Output ONLY the JSON object, no extra text.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[audio_part, prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": RISK_SCHEMA,
        },
    )

    return json.loads(response.text)

def render_risk_box(data: dict):
    if not data:
        st.warning("No analysis result available.")
        return

    risk_score = int(round(data.get("risk_score", 0)))
    risk_level = (data.get("risk_level") or "low").lower()

    level_display = {
        "low": "🟢 Low",
        "medium": "🟡 Medium",
        "high": "🟠 High",
        "critical": "🔴 Critical",
    }.get(risk_level, risk_level)

    st.subheader("Overall Risk Assessment")
    st.metric("Risk score (0–100)", value=risk_score)
    st.progress(min(max(risk_score, 0), 100))

    st.write("*Risk level:*", level_display)

    categories = data.get("categories") or []
    if categories:
        st.write("*Categories detected:*")
        st.write(", ".join(f"{c}" for c in categories))

    explanations = data.get("explanations") or []
    if explanations:
        st.write("*Why this might be risky / safe:*")
        for e in explanations:
            st.markdown(f"- {e}")

    rewrites = data.get("suggested_rewrites") or []
    if rewrites:
        st.subheader("Polite & respectful suggestions")
        for i, alt in enumerate(rewrites, start=1):
            st.markdown(f"*Option {i}:* {alt}")

st.set_page_config(
    page_title="SafeSpeak AI",
    page_icon="SafeSpeak_AI_Favicon.png",
    layout="wide",
)

st.title("SafeSpeak AI - Ethics Assistant for Social Media")
st.caption(
    "Helps you check if your text, screenshots, or voice notes could hurt someone's sentiments "
    "and suggests kinder alternatives."
)

tab_text, tab_image, tab_audio = st.tabs(
    ["Text input", "Image Input", "Voice Input"]
)

with tab_text:
    st.subheader("Analyse a text comment or post")

    user_text = st.text_area(
        "Type or paste your comment/post here:",
        height=180,
        placeholder="Example: \"People from XYZ community are ...\"",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_btn = st.button("Analyse text", type="primary")

    if analyze_btn:
        if not user_text.strip():
            st.error("Please enter some text to analyse.")
        else:
            with st.spinner("Analysing Text..."):
                try:
                    result = call_gemini_for_text(user_text)
                    render_risk_box(result)
                except Exception as e:
                    st.error(f"Error while analysing text: {e}")

with tab_image:
    st.subheader("Upload a screenshot of a comment section")

    uploaded_file = st.file_uploader(
        "Upload an image (PNG/JPEG)",
        type=["png", "jpg", "jpeg"],
        key="image_uploader",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

        if st.button("Analyse image"):
            try:
                image_bytes = uploaded_file.read()
                mime_type = uploaded_file.type or "image/png"

                with st.spinner("Analysing Image..."):
                    result = call_gemini_for_image(image_bytes, mime_type)
                    render_risk_box(result)
            except Exception as e:
                st.error(f"Error while analysing image: {e}")

with tab_audio:
    st.subheader("Speak and get a live risk analysis")

    st.write(
        "Click on the mic, speak your message, then stop.\n"
        "As soon as the recording finishes, Gemini will analyse what you said "
        "and how you said it (tone) and show a risk score."
    )

    audio_value = st.audio_input(
        "Press the mic button, speak, then stop recording",
        sample_rate=16000,
    )

    if audio_value:
        st.audio(audio_value)

        if "last_audio_size" not in st.session_state:
            st.session_state.last_audio_size = None

        current_size = len(audio_value.getvalue())

        if st.session_state.last_audio_size != current_size:
            with st.spinner("Analysing Voice..."):
                try:
                    audio_bytes = audio_value.read()
                    mime_type = audio_value.type or "audio/wav"

                    result = call_gemini_for_audio(audio_bytes, mime_type)
                    st.session_state.last_audio_size = current_size
                    st.session_state.last_audio_result = result
                except Exception as e:
                    st.error(f"Error while analysing audio: {e}")

        if "last_audio_result" in st.session_state and st.session_state.last_audio_result:
            render_risk_box(st.session_state.last_audio_result)
        else:
            st.info("Record a short message to see the risk analysis here.")