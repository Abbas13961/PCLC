import streamlit as st
import google.generativeai as genai
import PyPDF2
import re
import os

# Configure Gemini API from Streamlit secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
    st.stop()

# Initialize the generative model (cached for efficiency)
@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel('gemini-2.5-flash')

model = get_gemini_model()

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a given uploaded PDF file object.
    """
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None
    return text

def display_urdu_rtl_streamlit(text):
    """
    Displays the given text with right-to-left direction using HTML and CSS in Streamlit.
    """
    lines = text.split('\n')
    processed_lines = []
    list_item_pattern = re.compile(r'^\s*[-*+•]|\s*\d+\.')

    for line in lines:
        if list_item_pattern.match(line.strip()):
            line_without_marker = list_item_pattern.sub('', line.strip())
            processed_lines.append(f"• {line_without_marker}")
        else:
            processed_lines.append(line)

    processed_text = '\n'.join(processed_lines)

    rtl_html = f"""
    <div style='direction: rtl; text-align: right;'>
      <style>
        div, p, h1, h2, h3, h4, h5, h6, ul, ol, li, blockquote {{
          direction: rtl;
          text-align: right;
        }}
        ul {{
          list-style: none;
          padding-right: 20px;
          padding-left: 0;
        }}
         ol {{
          list-style: none;
          padding-right: 20px;
          padding-left: 0;
        }}
        li {{
          text-align: right;
          margin-right: 10px;
        }}
        * {{
          direction: rtl;
        }}
        div > ul > li, div > ol > li {{
            direction: rtl !important;
            text-align: right !important;
        }}
      </style>
      {processed_text}
    </div>
    """
    st.markdown(rtl_html, unsafe_allow_html=True)

def get_law_details(section_number, law_text, lang='ur'):
    """
    Retrieves and formats law details for a given section using Gemini.
    Lang: 'ur' for Urdu, 'en' for English, 'ro' for Roman Urdu.
    """
    if not law_text:
        return "معذرت، متعلقہ قانونی متن دستیاب نہیں ہے۔" if lang == 'ur' else "Sorry, relevant legal text is not available."

    lang_map_full = {
        'en': 'English',
        'ur': 'Urdu',
        'ro': 'Roman Urdu'
    }

    section_content = ""
    pattern = re.compile(
        r'(?:Section|SECTION|Sec\.؟|S\.|Dafaa|دفعہ)؟\s*' + re.escape(section_number) +
        r'[\.\s:-]*[\s\S]*؟(?=(?:Section|SECTION|Sec\.؟|S\.|Dafaa|دفعہ)؟\s*\d+|$)',
        re.IGNORECASE
    )
    match = pattern.search(law_text)

    if match:
        section_content = match.group(0)[:2000]  # Limit to avoid token issues
    else:
        section_content = ""

    prompt = f"""
    You are a legal assistant specializing in Pakistan Penal Code (PPC) and Code of Criminal Procedure (CrPC).
    A user is asking for details about Section {section_number}.

    Task:
    - If law text is provided below, extract information from it.
    - If law text is empty, search from your own knowledge/resources and provide accurate details.
    - If still no information is available, clearly respond with:
      \"{'اس کی تفصیل میرے پاس اس وقت موجود نہیں ہے۔' if lang == 'ur' else 'I do not have details for this section at the moment.'}\"

    Strictly provide the answer ONLY in the following format:
    دفعہ نمبر (Section Number)
    جرم (Offence)
    اردو عنوان (Urdu Title)
    تفصیل (Tafseel)
    زیادہ سے زیادہ سزا (Maximum Saza)
    کم سے کم سزا (Minimum Saza)
    ضمانت (Bailable / Non-bailable)
    قابل گرفتاری (Cognizable / Non-cognizable)
    کن عدالت میں سماعت ہوگی (Triable by)
    مثال (Example)
    کیا پولیس بغیر وارنٹ گرفتار کر سکتی ہے؟
    وارنٹ یا سمن (Warrant or Summons)
    کیا راضی نامہ ممکن ہے؟ (Compoundable or Not)
    سزا (Punishment)
    کس عدالت میں مقدمہ چلے گا؟ (Court by Which Triable)
    Suggestions (کسے بچا جا سکتا ہے)

    Provide the response in {lang_map_full[lang]} only.

    Law Text (if available):
    {section_content}
    """

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()

        if not answer or ("نامعلوم" in answer and section_content == ""):
            return "اس کی تفصیل میرے پاس اس وقت موجود نہیں ہے۔" if lang == 'ur' else "I do not have details for this section at the moment."

        return answer
    except Exception as e:
        return f"جیمنی API سے جواب حاصل کرنے میں خرابی: {e}" if lang == 'ur' else f"Error getting response from Gemini API: {e}"

def analyze_case(case_description, ppc_text, crpc_text, lang='ur'):
    """
    Analyzes a given case description and suggests relevant PPC/CrPC sections.
    Provides output in the requested language.
    """
    if not ppc_text or not crpc_text:
        return "معذرت، قانونی متن دستیاب نہیں ہے کیس کے تجزیے کے لیے۔" if lang == 'ur' else "Sorry, legal text is not available for case analysis."

    lang_map_full = {
        'en': 'English',
        'ur': 'Urdu',
        'ro': 'Roman Urdu'
    }

    prompt = f"""
    You are a legal assistant. Analyze the following case scenario and identify the most relevant sections from the Pakistan Penal Code (PPC) and Code of Criminal Procedure (CrPC).
    Explain why each section is relevant and then provide a summary of the potential charges.
    Also, include general suggestions on how one might be legally defended, clearly stating it's not legal advice.

    Case Scenario:
    \"{case_description}\"

    Relevant Sections (from PPC and CrPC texts provided, if needed, limit text to avoid exceeding token limits):
    PPC Text: {ppc_text[:7000]} # Adjusted limit
    CrPC Text: {crpc_text[:7000]} # Adjusted limit

    Provide the output in {lang_map_full[lang]} in a clear, conversational manner.
    Start the response with a clear disclaimer in the requested language:
    \"نوٹ: میں ایک مصنوعی ذہانت پر مبنی ماڈل ہوں اور آپ کو قانونی مشورہ نہیں دے سکتا۔ فراہم کردہ معلومات صرف عمومی آگاہی کے لیے ہیں۔ کسی بھی حقیقی قانونی معاملے کے لیے، یہ انتہائی ضروری ہے کہ آپ فوراً ایک مستند وکیل سے رابطہ کریں جو آپ کے کیس کا تفصیلی جائزہ لے کر درست قانونی رہنمائی فراہم کر سکے۔\"
    (Or its English/Roman Urdu equivalent)

    Then, after the disclaimer, provide the analysis and suggestions.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"جیمنی API سے جواب حاصل کرنے میں خرابی: {e}" if lang == 'ur' else f"Error getting response from Gemini API: {e}"

# Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("🇵🇰 Pakistan Criminal Law Chatbot ⚖️")
    st.markdown("Created by: Muhammad Ali Abbas -NED University")
    st.markdown("---")

    st.sidebar.header("Upload Law Files")
    ppc_uploaded_file = st.sidebar.file_uploader("Upload Pakistan Penal Code (PPC) PDF", type="pdf", key="ppc_uploader")
    crpc_uploaded_file = st.sidebar.file_uploader("Upload Code of Criminal Procedure (CrPC) PDF", type="pdf", key="crpc_uploader")

    ppc_text = None
    crpc_text = None

    if ppc_uploaded_file:
        with st.spinner("Extracting text from PPC PDF..."):
            ppc_text = extract_text_from_pdf(ppc_uploaded_file)
        if ppc_text:
            st.sidebar.success("PPC PDF loaded.")
        else:
            st.sidebar.error("Failed to load PPC PDF.")
    else:
        st.sidebar.info("Please upload the PPC PDF file.")


    if crpc_uploaded_file:
        with st.spinner("Extracting text from CrPC PDF..."):
            crpc_text = extract_text_from_pdf(crpc_uploaded_file)
        if crpc_text:
            st.sidebar.success("CrPC PDF loaded.")
        else:
            st.sidebar.error("Failed to load CrPC PDF.")
    else:
        st.sidebar.info("Please upload the CrPC PDF file.")


    st.markdown("""
    **سوال پوچھیں (Ask a Question):**
    مثالیں (Examples):
    - `420 کیا ہے؟`
    - `What is Section 302 PPC?`
    - `Chori ke baad qatal mein kya laws lagte hain?`
    """)

    user_input = st.text_input("آپ کا سوال (Your Question):", key="user_query")

    if user_input:
        if not ppc_text or not crpc_text:
            st.warning("Please upload both PPC and CrPC PDF files in the sidebar to use the chatbot.")
        else:
            response_lang = 'ur' # Default to Urdu
            # Language detection logic
            if "what is" in user_input.lower() or "section" in user_input.lower() or "law" in user_input.lower() and "urdu" not in user_input.lower() and "roman" not in user_input.lower():
                response_lang = 'en'
            elif any(char.isalpha() for char in user_input) and not any(char.isascii() for char in user_input): # Basic check for Urdu script
                response_lang = 'ur'
            elif any(char.isalpha() for char in user_input) and (
                "kya" in user_input.lower() or "kiya" in user_input.lower() or
                "kaise" in user_input.lower() or "mein" in user_input.lower() or
                "mujhe" in user_input.lower() or "batao" in user_input.lower() or
                "lagte" in user_input.lower() or "hogaya" in user_input.lower()
            ): # Basic check for Roman Urdu
                response_lang = 'ro'

            section_match = re.search(r'\b(PPC|CrPC)؟\s*(\d+)\b', user_input, re.IGNORECASE)

            if section_match:
                law_prefix = section_match.group(1)
                section_num = section_match.group(2)

                law_text_to_use = None
                if law_prefix and law_prefix.lower() == 'crpc':
                    law_text_to_use = crpc_text
                    st.info(f"CrPC کی دفعہ {section_num} کی تفصیلات نکال رہا ہوں۔")
                else: # Default to PPC
                    law_text_to_use = ppc_text
                    st.info(f"PPC کی دفعہ {section_num} کی تفصیلات نکال رہا ہوں۔")

                if law_text_to_use:
                    with st.spinner("Generating details..."):
                        details = get_law_details(section_num, law_text_to_use, lang=response_lang)
                    if response_lang == 'ur':
                        display_urdu_rtl_streamlit(details)
                    else:
                        st.write(details)
                else:
                    st.error("Error: Law text not available for the requested section.")

            else:
                st.info("کیس کا تجزیہ کر رہا ہوں...")
                with st.spinner("Analyzing case..."):
                    analysis = analyze_case(user_input, ppc_text, crpc_text, lang=response_lang)
                if response_lang == 'ur':
                    display_urdu_rtl_streamlit(analysis)
                else:
                    st.write(analysis)

    st.markdown("---")
    st.markdown("""
    <div style='direction: rtl; text-align: right;'>
    📌 **ڈس کلیمر:**
    یہ معلومات صرف قانونی رہنمائی کے لیے ہے، یہ کسی قسم کی قانونی مشورہ نہیں ہے۔
    تفصیلی مشورہ کسی لائسنس یافتہ وکیل سے لینا ضروری ہے۔
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    **Disclaimer:**
    This information is for legal guidance only, it is not legal advice of any kind.
    Detailed advice should be sought from a licensed lawyer.
    """)
    st.write("شکریہ! اللہ حافظ۔")

if __name__ == "__main__":

    main()
