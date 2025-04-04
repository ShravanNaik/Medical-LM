
import sys
import importlib
importlib.import_module('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import threading
import uuid
import streamlit as st
import openai
from langchain_openai import ChatOpenAI
import os
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from crewai import Agent, Task, Crew 
from openai import OpenAI 
from pydub import AudioSegment 
import speech_recognition as sr
import wave
import sounddevice as sd
import numpy as np 
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pyttsx3
import sys
from io import BytesIO
import sounddevice as sd



load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
client = OpenAI()


AGENT_PERSONALITIES = {
    "clinical": {
        "tone": "professional and precise",
        "vocabulary": "medical terminology with clear explanations",
        "pacing": "methodical and thorough"
    },
    "educational": {
        "tone": "instructive and encouraging",
        "vocabulary": "accessible medical terms with definitions",
        "pacing": "step-by-step with reinforcement"
    },
    "conversational": {
        "tone": "friendly and reassuring",
        "vocabulary": "everyday language with occasional medical terms",
        "pacing": "natural dialogue flow"
    }
}

def generate_summary(text,temperature,personality="clinical"):

    model = ChatOpenAI(model="gpt-4o", temperature=temperature)
    personality_traits = AGENT_PERSONALITIES.get(personality, AGENT_PERSONALITIES["clinical"])
    summarizer = Agent(
        role="Senior Medical Document Synthesizer",
        goal="Generate a concise, actionable, and medically accurate summary of the provided document",
        backstory=f"You are a medical documentation expert with years of experience distilling complex clinical information into clear summaries. You communicate in a {personality_traits['tone']} style, using {personality_traits['vocabulary']} with a {personality_traits['pacing']} approach.",
        verbose=True,
        llm=model,
    )

    summary_task = Task(
        description=f"""Analyze and summarize the following medical document:
        
        {text}
        
        Your summary should:
        1. Identify the document type (discharge summary, consultation note, etc.)
        2. Extract key patient demographics
        3. Outline primary diagnosis and relevant medical history
        4. Detail critical lab values and imaging results
        5. Highlight medication changes and dosing instructions
        6. Emphasize post-discharge/follow-up care plans with specific timelines
        7. Flag any urgent warning signs patients should monitor
        8. Organize information in a clinical hierarchy of importance
        9. Include a brief "Patient Instructions" section in simplified language
        
        Format the summary with clear headings, bullet points where appropriate, and ensure all critical information is prioritized.
        """,
        agent=summarizer,
        expected_output="A comprehensive, structured summary of the medical document with all clinically relevant information organized for quick reference and patient safety."
    )

    crew = Crew(agents=[summarizer],tasks=[summary_task],verbose=True)
    result = crew.kickoff()
    return result

def generate_audio_overview(text,temperature):

    llm = ChatOpenAI(model="gpt-4o", temperature=temperature)


    audio_agent = Agent(
        role="Medical Audio Narrator",
        goal="Transform complex medical information into clear, engaging audio content",
        backstory="You are an expert in medical communication who specializes in converting technical documentation into accessible audio summaries. You understand how to structure information for listening comprehension rather than reading.",
        llm=llm,  
        verbose=True,
    )


    audio_task = Task(
        description=f"""Create an audio-optimized version of this medical document:

        {text}
        
        Your audio script should:
        1. Begin with a clear introduction identifying the document type
        2. Use natural speech patterns with appropriate transitions
        3. Include verbal signposts ("Next, let's discuss...", "An important point to note...")
        4. Emphasize critical information through verbal cues ("This is crucial...")
        5. Avoid complex numerical sequences that are difficult to remember when heard
        6. Use repetition strategically for key points
        7. Include natural pauses (indicated by [pause] in your script)
        8. End with a summary of the 2-3 most critical points
        
        Keep the total duration to approximately 3-5 minutes when read aloud at a natural pace.
        """,
        agent=audio_agent,
        expected_output="A conversational, clearly structured audio script that effectively communicates the medical information in a format optimized for listening.",
    )


    crew = Crew(agents=[audio_agent], tasks=[audio_task], verbose=True)
    result = crew.kickoff()
    return result

def generate_audio(text, voice,output_file="output.mp3"):

    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text,
)
        # response = openai_client.audio.speech.create(
        #     model="tts-1",  # Use "tts-1" for standard quality or "tts-1-hd" for high quality
        #     voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
        #     input=text,
        # )
        response.stream_to_file(output_file)
        return output_file
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def split_text_into_chunks(text, max_length=4096):
    """Split text into chunks of max_length characters."""
    chunks = []
    print(text)
    while len(text) > max_length:
        # Find the last space within the limit to avoid splitting words
        split_index = text.rfind(" ", 0, max_length)
        if split_index == -1:
            split_index = max_length  # If no space is found, split at max_length
        chunks.append(text[:split_index])
        text = text[split_index:].strip()
    chunks.append(text)  
    print("chunks",chunks)# Add the remaining text
    return chunks

def generate_audio_in_chunks(text, voice, output_file="output.mp3"):
    """Generate audio from long text by splitting it into chunks."""
    try:
        chunks = split_text_into_chunks(text)
        audio_segments = []

        for i, chunk in enumerate(chunks):
            # Generate audio for each chunk
            chunk_file = f"chunk_{i}.mp3"
            response = client.audio.speech.create(
                model="tts-1",  # Use "tts-1" for standard quality or "tts-1-hd" for high quality
                voice=voice,  # Options: alloy, echo, fable, onyx, nova, shimmer
                input=chunk,
            )
            response.stream_to_file(chunk_file)
            audio_segments.append(AudioSegment.from_file(chunk_file))

        # Combine all audio segments
        combined_audio = sum(audio_segments)
        combined_audio.export(output_file, format="mp3")

        # Clean up temporary chunk files
        for chunk_file in [f"chunk_{i}.mp3" for i in range(len(chunks))]:
            os.remove(chunk_file)

        return output_file
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def generate_podcast(text,temparature):

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=temparature)  # Higher temperature for creativity


    host = Agent(
        role="Medical Podcast Host 'Dr. Sarah Chen'",
        goal="Guide an informative and engaging discussion about complex medical topics",
        backstory="You are Dr. Sarah Chen, a charismatic podcast host with a background in internal medicine and health communication. You have a knack for making complex medical information accessible through thoughtful questions and relatable examples.",
        llm=llm,
        verbose=True,
    )

    expert = Agent(
        role="Medical Specialist 'Dr. William Rodriguez'",
        goal="Provide expert analysis and context to medical information while maintaining accessibility",
        backstory="You are Dr. William Rodriguez, a respected specialist with 15 years of clinical experience and a passion for patient education. You're known for your ability to explain complex conditions using metaphors and practical examples that patients remember.",
        llm=llm,
        verbose=True,
    )

    podcast_task = Task(
        description=f"""Create an engaging podcast-style discussion about this medical document:

        Document Content:
        {text}
        
        Script requirements:
        1. Begin with host Dr. Sarah Chen welcoming listeners and introducing today's topic
        2. Introduce expert guest Dr. William Rodriguez with relevant credentials
        3. Structure the conversation around 3-4 main discussion points extracted from the document
        4. Include natural dialogue elements:
           - Thoughtful questions from Dr. Chen
           - Detailed but accessible explanations from Dr. Rodriguez
           - Occasional professional disagreements or additional perspectives
           - Personal anecdotes that illustrate key points (while maintaining privacy)
        5. Include practical takeaways for patients or caregivers
        6. End with a summary of key points and farewell to listeners
        
        The conversation should feel natural, educational, and engaging while covering all critical medical information from the document.
        """,
        agent=expert,
        expected_output="A realistic, engaging podcast transcript that effectively communicates medical information through natural dialogue between two medical professionals.",
    )

    crew = Crew(agents=[host, expert], tasks=[podcast_task], verbose=True)
    result = crew.kickoff()
    return result


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)


# Function to create a vector store
def get_vector_store(text_chunks):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error saving vector store: {str(e)}")


# Function to generate FAQs and answers using CrewAI
def generate_faqs(text,temparature):
    # Define the LLM to be used by CrewAI
    llm = ChatOpenAI(model="gpt-4o", temperature=temparature)  # Moderate temperature for creativity

    # Define the FAQ agent
    faq_agent = Agent(
        role="Patient Education Specialist",
        goal="Create comprehensive, patient-centered FAQs that address common concerns and knowledge gaps",
        backstory="You are an expert in patient education with years of experience anticipating what questions patients have after receiving medical information. You excel at identifying information gaps and providing clear, actionable answers.",
        llm=llm,
        verbose=True,
    )

    faq_task = Task(
        description=f"""Generate a comprehensive FAQ based on this medical document:
        
        {text}
        
        Your FAQ should:
        1. Address at least 8-12 questions covering different aspects of care
        2. Include questions about:
           - Diagnosis and condition specifics
           - Treatment plans and medication management
           - Follow-up care requirements
           - Warning signs requiring medical attention
           - Daily lifestyle adaptations
           - Recovery expectations and timelines
        3. Organize questions in order of typical patient priority
        4. Ensure answers are:
           - Concise (2-4 sentences per answer)
           - Actionable with specific guidance
           - Written at approximately 8th-grade reading level
           - Sensitive to common concerns and anxieties
        
        Format each FAQ as:
        Q: [Question phrased from patient perspective]
        A: [Clear, concise answer with actionable information]
        """,
        agent=faq_agent,
        expected_output="A comprehensive set of FAQs addressing the most important patient questions raised by the medical document, with clear and actionable answers.",
    )

    crew = Crew(agents=[faq_agent], tasks=[faq_task], verbose=True)
    result = crew.kickoff()
    return result

def generate_flashcards(text,temparature):

    llm = ChatOpenAI(model="gpt-4o", temperature=temparature)  

   
    quiz_master = Agent(
        role="Quiz Master",
        goal="Create fun and educational flashcards to help users learn about the document content.",
        backstory="You are an expert in creating engaging and interactive learning materials. You love designing quizzes and flashcards that make learning enjoyable.",
        llm=llm,
        verbose=True,
    )

   
    flashcard_task = Task(
        description=f"""Create a set of flashcards based on the following document content:
        Document Content:
        {text}

        Guidelines:
        1. Generate multiple-choice questions with 4 options.
        2. Ensure the questions are fun, engaging, and educational.
        3. Format the output as follows:
           Q: [Question]
           A: [Option 1]
           B: [Option 2]
           C: [Option 3]
           D: [Option 4]
           Correct Answer: [Correct Option]
        """,
        agent=quiz_master,
        expected_output="A list of flashcards, each containing a question, 4 options, and the correct answer.",
    )

 
    crew = Crew(agents=[quiz_master], tasks=[flashcard_task], verbose=True)
    result = crew.kickoff()
    return result

def parse_flashcards(flashcard_text):
    flashcards = []
    current_flashcard = {}
    lines = flashcard_text.split("\n")
    for line in lines:
        if line.startswith("Q:"):
            if current_flashcard: 
                flashcards.append(current_flashcard)
            current_flashcard = {"question": line.replace("Q:", "").strip(), "options": [], "correct_answer": ""}
        elif line.startswith("A:") or line.startswith("B:") or line.startswith("C:") or line.startswith("D:"):
            current_flashcard["options"].append(line.strip())
        elif line.startswith("Correct Answer:"):
            current_flashcard["correct_answer"] = line.replace("Correct Answer:", "").strip()
    if current_flashcard:  
        flashcards.append(current_flashcard)
    return flashcards

def display_flashcards(flashcards):
    for i, flashcard in enumerate(flashcards):
        st.write(f"**Question {i + 1}:** {flashcard['question']}")
        user_answer = st.radio(
            "Choose the correct answer:",
            options=flashcard["options"],
            key=f"question_{i}",
        )
        if st.button(f"Submit Answer for Question {i + 1}"):
            if user_answer.startswith(flashcard["correct_answer"]):
                st.success("Correct! 🎉")
            else:
                st.error(f"Wrong! The correct answer is: {flashcard['correct_answer']}")
        st.write("---")

def combine_audio_files(file1, file2, output_file="podcast.mp3"):
    audio1 = AudioSegment.from_file(file1)
    audio2 = AudioSegment.from_file(file2)
    combined = audio1 + audio2
    combined.export(output_file, format="mp3")
    return output_file

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error saving vector store: {str(e)}")

def get_conversational_chain(temparature):
    prompt_template = """
    You are an intelligent AI with reasoning capabilities. Answer using both document context and logical reasoning.
    If the answer is not found in the document, infer using logic.
    Context:\n {context}\n
    question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-4-turbo", temperature=temparature,streaming=True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question,temparature):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain(temparature)

        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error during query: {str(e)}")

def generate_recommendations(text,temparature):
    # Define the LLM to be used by CrewAI
    llm = ChatOpenAI(model="gpt-4o", temperature=temparature)

    # Define the research assistant agent
    research_agent = Agent(
        role="Clinical Evidence Specialist",
        goal="Provide evidence-based recommendations and current best practices relevant to the medical case",
        backstory="You are a medical researcher with expertise in translating current clinical evidence into practical recommendations. You stay current with medical literature and can identify which guidelines and research are most relevant to specific cases.",
        llm=llm,
        verbose=True,
    )

    recommendation_task = Task(
        description=f"""Analyze this medical document and provide evidence-based recommendations:
        
        {text}
        
        Your response should:
        1. Identify 3-5 key clinical topics from the document requiring evidence-based guidance
        2. For each topic, provide:
           - Current best practice recommendations from major medical societies
           - Reference to relevant clinical guidelines with year of publication
           - Brief summary of any recent developments or changes in approach
           - Practical implementation considerations for clinicians
        3. Include recommendations for:
           - Diagnostic approach refinements
           - Treatment optimization strategies
           - Monitoring parameters and follow-up timing
           - Patient education resources
        4. Note any areas where evidence is evolving or controversial
        
        Format your response with clear headings for each clinical topic and structured recommendations.
        """,
        agent=research_agent,
        expected_output="A set of evidence-based clinical recommendations directly relevant to the medical case, including references to current guidelines and best practices.",
    )

    crew = Crew(agents=[research_agent], tasks=[recommendation_task], verbose=True)
    result = crew.kickoff()
    return result

def generate_conversational_response(user_input,raw_text):
    # Define the LLM to be used by CrewAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # Define the conversational agent
    conversational_agent = Agent(
        role="Conversational AI",
        goal="Provide instant answers and engage in a two-way conversation with the user.",
        backstory="You are a friendly and knowledgeable AI assistant who can answer questions, provide explanations, and engage in natural conversations.",
        llm=llm,
        verbose=True,
    )

    # Define the conversational task
    conversational_task = Task(
        description=f"""Respond to the following user input:
        User Input:
        {user_input}
        try to provide answers from the Document Context:
        {raw_text}

        Guidelines:
        1. Provide a clear and concise answer to the user's question.
        2. If the user is dictating notes, summarize and save them.
        3. Maintain a natural and engaging conversational tone.
        """,
        agent=conversational_agent,
        expected_output="A natural and informative response to the user's input from the Document Context.",
    )

    # Create a crew and execute the task
    crew = Crew(agents=[conversational_agent], tasks=[conversational_task], verbose=2)
    result = crew.kickoff()
    return result

def record_audio(filename="output.wav", duration=5, samplerate=44100):
    st.write("🎤 Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()  # Wait until recording is finished

    # Save as WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    st.success("✅ Recording complete!")
    return filename

def transcribe_audio(audio_file):
    with open(audio_file, "rb") as file:
        transcript=openai.audio.transcriptions.create(
                            model="whisper-1",
                            file=file,
                        )
    return transcript.text

def generate_conversational_response(user_question,context):
    # Define the LLM to be used by CrewAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # Define the conversational agent
    conversational_agent = Agent(
                role="Medical Conversational Agent",
                goal="Provide accurate and context-aware answers to user questions based on the uploaded medical documents.",
                backstory="You are an expert in medical documentation and can provide detailed answers to user queries based on the content of the uploaded documents.",
                llm=llm,
                verbose=True,
            )

            # Define the conversation task
    conversation_task = Task(
                description=f"""Answer the following user question based on the provided context:
                User Question: {user_question}
                Context: {context}
                """,
                agent=conversational_agent,
                expected_output="A detailed and accurate answer to the user's question.",
            )

            # Create a crew and execute the task
    crew = Crew(agents=[conversational_agent], tasks=[conversation_task], verbose=True)
    response = crew.kickoff()
    return response.raw


def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulates a standalone question which can be understood"
        "without the chat history. Do not answer the question,"
        "just reformulate it if needed and otherwise return it as it"
    )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("system", system_prompt),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(llm):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever_chain = _get_context_retriever_chain(vector_db, llm)
    


    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer "
        "the question.If the answer is not found in the document,say that you dont know as it is not in the context."\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_rag_response(messages,temparature):
    llm = ChatOpenAI(model="gpt-4o", temperature=temparature,streaming=True)
    conversation_rag_chain = get_conversational_rag_chain(llm)
    response_message = "*(AI Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})


def _get_context_retriever_chain_voice(vector_db, llm):
    retriever = vector_db.as_retriever()
    system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulates a standalone question which can be understood"
        "without the chat history. Do not answer the question,"
        "just reformulate it if needed and otherwise return it as it,"
    )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("system", system_prompt),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_voice_chain(llm):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever_chain = _get_context_retriever_chain_voice(vector_db, llm)
    


    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you dont know the answer, say that you dont know.
          keep the answer concise.If the answer is not found in the document,say that you dont know as it is not in the context. \n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt) 
    

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_rag_voice_response(messages,temparature):
    llm = ChatOpenAI(model="gpt-4o", temperature=temparature,streaming=True)
    conversation_rag_chain = get_conversational_rag_voice_chain(llm)
    response_message = ""
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk
    threading.Thread(target=speak_response(response_message), daemon=True).start()
    # speak_response(response_message)
    st.session_state.messages_voice.append({"role": "assistant", "content": response_message})



def speak(text):
    """Convert text to speech using pyttsx3."""
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)  # Adjust speed (default ~200)
    engine.say(text)
    engine.runAndWait()

def speak_response(response_text, voice="alloy"):
    """Converts text to speech and streams it in real-time."""
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=response_text
    )
    audio_data = BytesIO(response.read()) 
    play_audio(audio_data) 
    

def play_audio(audio_data):
    """Plays audio data in real-time."""
    audio = AudioSegment.from_file(audio_data, format="mp3")
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(audio.array_type).max  # Normalize audio
    st.success("Speaking...")
    sd.play(samples, samplerate=audio.frame_rate)
    sd.wait() 

def listen():
    """Listen to the user's voice input and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        audio = recognizer.listen(source)
        try:
            st.success("Processing...")
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand what you said.")
        except sr.RequestError:
            st.error("Sorry, there was an issue with the speech recognition service.")
    return None

def main():

    st.markdown("<h1 style='text-align: center;'>Medical Document Assistant 🩺</h1>", unsafe_allow_html=True)
    st.sidebar.header("Settings")
    temparature = st.sidebar.slider("Temparature",min_value=0.0,max_value=2.0,value=0.7)

    if "messages" not in st.session_state:
       st.session_state.messages = [
        ]
    if "messages_voice" not in st.session_state:
       st.session_state.messages_voice = [
        ]
    with st.sidebar:
        st.button("Clear Convesations",on_click=lambda : (st.session_state.messages.clear(),st.session_state.messages_voice.clear()),type="primary")


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    for message1 in st.session_state.messages_voice:
        with st.chat_message(message1["role"]):
            st.markdown(message1["content"])


    if user_question:= st.chat_input("Please enter your message"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
            st.write_stream(stream_llm_rag_response(messages,temparature=temparature))


    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
    st.sidebar.header("Features")
    # with st.sidebar.("Features", expanded=False):

    st.markdown(
        """
        <style>
            .stButton>button {
                width: 100%;
                margin-bottom: 8px;
                border: none;
                background-color: #444; /* Dark background (adjust as needed) */
                color: white;
                font-size: 16px;
                text-align: center;
                border-radius: 5px;
                padding: 10px;
            }
            .stButton>button:hover {
                background-color: #444;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    

    if st.sidebar.button("Generate Summary"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Generating summary..."):
                    raw_text = get_pdf_text(pdf_docs)
                    summary = generate_summary(raw_text,temparature)
                st.subheader("Document Summary")
                st.write(summary.raw)
    
    if st.sidebar.button("Generate FAQs"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Generating FAQs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    faqs_summary = generate_faqs(raw_text,temparature)
                st.subheader("FAQS")
                st.write(faqs_summary.raw)

    if st.sidebar.button("Generate Audio Overview"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Generating audio overview..."):
                    raw_text = get_pdf_text(pdf_docs)
                    audio_summary = generate_audio_overview(raw_text,temparature)
                st.subheader("Audio Overview Summary")

                    # Generate audio file
                audio_file = generate_audio(audio_summary.raw,"alloy")
                if audio_file:
                        # Automatically play the audio
                    st.audio(audio_file, format="audio/mp3", start_time=0)
                    st.success("Audio overview generated and playing!")

    
    if st.sidebar.button("Generate Podcast"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Generating podcast..."):
                    raw_text = get_pdf_text(pdf_docs)
                    podcast_transcript = generate_podcast(raw_text,temparature)
                st.subheader("Podcast Transcript")
                    # st.write(podcast_transcript)

                    # Generate audio file
                audio_file = generate_audio(podcast_transcript.raw,"alloy")
                if audio_file:
                        # Automatically play the audio
                    st.audio(audio_file, format="audio/mp3", start_time=0)
                    st.success("Podcast generated and playing!")

    if st.sidebar.button("Play Flashcard Game"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Generating flashcards..."):
                    raw_text = get_pdf_text(pdf_docs)
                    flashcard_text = generate_flashcards(raw_text,temparature)
                st.subheader("Flashcard Game")
                st.write("Let's test your knowledge! Here are some fun questions:")

                    # Parse and display flashcardss
                flashcards = parse_flashcards(flashcard_text.raw)
                display_flashcards(flashcards)

    if st.sidebar.button("Smart Search & Recommendations"):
                if not pdf_docs:
                    st.error("Please upload at least one PDF file.")
                else:
                    with st.spinner("Generating recommendations..."):
                        raw_text = get_pdf_text(pdf_docs)
                        recommendations = generate_recommendations(raw_text,temparature)
                    st.subheader("Smart Search & Recommendations")
                    st.write(recommendations.raw)
    
    if st.sidebar.button("Start Voice Conversation"):
         while True:
             with st.spinner("Listening..."):
            # Record audio from the user
                # audio_file = record_audio()
                # user_text = transcribe_audio(audio_file)
                # if user_text:
                #     st.write(user_text)

                user_t = listen()
                if user_t and user_t.lower() =="stop the conversation":
                    st.write("Thank you for the conversation")
                    break
                elif user_t:
                    st.session_state.messages_voice.append({"role": "user", "content": user_t})
                    with st.chat_message("user"):
                        st.markdown(user_t)
               

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""
                        messages_voice = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages_voice]
                        st.write_stream(stream_llm_rag_voice_response(messages_voice,temparature=temparature))
    

    if st.sidebar.button("Read Aloud"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Reading..."):
                    raw_text = get_pdf_text(pdf_docs)
                audio_file = speak(raw_text)





if __name__ == "__main__":
    main()
    
