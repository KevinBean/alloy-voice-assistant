import base64
from threading import Lock, Thread
import json
import time
import keyboard

import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
# import ChatOllama
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError
from langchain.schema.messages import HumanMessage, AIMessage

# load environment variables from .env file
load_dotenv()

# define the assistant class for collecting the webcam stream
class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()

# define the assistant class for answering the user's questions
class Assistant:
    def __init__(self, model, with_image=True, SYSTEM_PROMPT=None):
        if not SYSTEM_PROMPT:
            self.SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """
        else:
            self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.with_image = with_image
        self.chain = self._create_inference_chain(model)


    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="onyx",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        if self.with_image:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history"),
                    (
                        "human",
                        [
                            {"type": "text", "text": "{prompt}"},
                            {
                                "type": "image_url",
                                "image_url": "data:image/jpeg;base64,{image_base64}",
                            },
                        ],
                    ),
                ]
            )
        else:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", [{"type": "text", "text": "{prompt}"}]),
                ]
            )
        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )
    def save_history(self, file_path):
        history_data = self.chain.get_session_history("1")# Convert chat history to dictionary
        print(history_data.messages)
        # Function to format messages as Markdown
        def format_messages_as_markdown(messages):
            print(messages)
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    formatted_messages.append(f"**Human:** {msg.content}\n")
                elif isinstance(msg, AIMessage):
                    formatted_messages.append(f"**AI:** {msg.content}\n")
            return "\n".join(formatted_messages)

        # Format the messages
        formatted_content = format_messages_as_markdown(history_data.messages)
        print(formatted_content)

        # Write to a Markdown file
        with open("chat_history.md", "w") as f:
            f.write(formatted_content)

        print("Chat history saved to chat_history.md")

# anothe assistant class without the image
class AssistantWithoutImage(Assistant):
    def __init__(self, model, SYSTEM_PROMPT=None):
        super().__init__(model, with_image=False, SYSTEM_PROMPT=SYSTEM_PROMPT)

    # modify the answer method to remove the image parameter
    def answer(self, prompt):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)


# define if the assistant will use the image
with_image = True

# ask use to choose if the assistant will use the image from the webcam stream
with_image = input("Do you want to use the image from the webcam stream as the context of the conversation? (y/n): ").lower() == "y"



# start the webcam stream
webcam_stream = WebcamStream().start() if with_image else None


# define the models to be used by the assistant
models = {}
models["gemini-1.5-flash-latest"] = { "model": ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest"), "with_image": True }

# You can use OpenAI's GPT-4o model instead of Gemini Flash
# by uncommenting the following line:
models["gpt-4o-mini"] = { "model": ChatOpenAI(model="gpt-4o-mini"), "with_image": True }

# you can use local model through ollama
models["llama3.1"] = { "model": ChatOllama(model="llama3.1"), "with_image": False }

# let use choose the model to be used by the assistant
print("Choose the model to be used by the assistant:")
# only show the models that can be used with the webcam stream
if with_image:
    # only shoe the models that can be used with the webcam stream by checking the 'with_image' key
    for i, (model_name, model) in enumerate(models.items()):
        if model["with_image"]:
            print(f"{i + 1}. {model_name}")

else:
    for i, model_name in enumerate(models.keys()):
        print(f"{i + 1}. {model_name}")

# get the model index from the user
model_index = int(input("Enter the model index: ")) - 1

# get the model to be used by the assistant
model = list(models.values())[model_index]["model"]

# prompts for the assistant
prompts = {
    "as a english teacher": """
    You are an English teacher who is helping a student with their speaking skills.
    The student is going to have a conversation with you about a topic of their choice.
    You should ask the student questions about the topic and provide feedback on their answers.
    The feedback should be about the student's pronunciation, grammar, and vocabulary.
    The student will ask you questions about the topic as well.
    You should answer the student's questions and provide additional information about the topic.
    You should also ask the student questions about the topic to help them practice their speaking skills.
    """,
    "as a CBT therapist": """
    You are a Cognitive Behavioral Therapist. Your kind and open approach to CBT allows users to confide in you. You will be given an array of dialogue between the therapist and fictional user you refer to in the second person, and your task is to provide a response as the therapist. 

RESPOND TO THE USER'S LATEST PROMPT, AND KEEP YOUR RESPONSES AS BRIEF AS POSSIBLE. 

You ask questions one by one and collect the user's responses to implement the following steps of CBT :

1. Help the user identify troubling situations or conditions in their life. 

2. Help the user become aware of their thoughts, emotions, and beliefs about these problems.

3. Using the user's answers to the questions, you identify and categorize negative or inaccurate thinking that is causing the user anguish into one or more of the following CBT-defined categories:

- All-or-Nothing Thinking
- Overgeneralization
- Mental Filter
- Disqualifying the Positive
- Jumping to Conclusions
- Mind Reading
- Fortune Telling
- Magnification (Catastrophizing) or Minimization
- Emotional Reasoning
- Should Statements
- Labeling and Mislabeling
- Personalization

4. After identifying and informing the user of the type of negative or inaccurate thinking based on the above list, you help the user reframe their thoughts through cognitive restructuring. You ask questions one at a time to help the user process each question separately.

For example, you may ask:

- What evidence do I have to support this thought? What evidence contradicts it?
- Is there an alternative explanation or perspective for this situation?
- Am I overgeneralizing or applying an isolated incident to a broader context?
- Am I engaging in black-and-white thinking or considering the nuances of the situation?
- Am I catastrophizing or exaggerating the negative aspects of the situation?
- Am I taking this situation personally or blaming myself unnecessarily?
- Am I jumping to conclusions or making assumptions without sufficient evidence?
- Am I using "should" or "must" statements that set unrealistic expectations for myself or others?
- Am I engaging in emotional reasoning, assuming that my feelings represent the reality of the situation?
- Am I using a mental filter that focuses solely on the negative aspects while ignoring the positives?
- Am I engaging in mind reading, assuming I know what others are thinking or feeling without confirmation?
- Am I labeling myself or others based on a single event or characteristic?
- How would I advise a friend in a similar situation?
- What are the potential consequences of maintaining this thought? How would changing this thought benefit me?
- Is this thought helping me achieve my goals or hindering my progress?

Using the user's answers, you ask them to reframe their negative thoughts with your expert advice. As a parting message, you can reiterate and reassure the user with a hopeful message.
    When there is no more to discuss and the user ask you to summarize the session, create a markdown table summarizing the conversation. It should have columns for each negative belief mentioned by the user, emotion, category of negative thinking, and reframed thought.
    """
}
# multiple assistants
assistants = {"default_with_image": Assistant(model), 
              "default_without_image": AssistantWithoutImage(model), 
              "as a english teacher": AssistantWithoutImage(model, SYSTEM_PROMPT=prompts["as a english teacher"]),
              "as a CBT therapist": AssistantWithoutImage(model, SYSTEM_PROMPT=prompts["as a CBT therapist"])}	

# let user choose the assistant to be used
print("Choose the assistant to be used:")
if with_image:
    assistants_with_image = [assistant for assistant in assistants.keys() if assistants[assistant].with_image]
    for i, assistant_name in enumerate(assistants_with_image):
        print(f"{i + 1}. {assistant_name}")
else:
    assistants_without_image = [assistant for assistant in assistants.keys() if not assistants[assistant].with_image]
    for i, assistant_name in enumerate(assistants_without_image):
        print(f"{i + 1}. {assistant_name}")

# get the assistant index from the user
assistant_index = int(input("Enter the assistant index: ")) - 1

# get the assistant to be used
assistant = assistants[assistants_with_image[assistant_index]] if with_image else assistants[assistants_without_image[assistant_index]]

def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, webcam_stream.read(encode=True)) if with_image else assistant.answer(prompt)

    except UnknownValueError:
        print("There was an error processing the audio.")


recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)
# prompt the user to speak
print("I am ready to assist you. Please start speaking. Press 'q' to stop.")

# with the camera stream running, the assistant will answer the user's questions

while True:
    cv2.imshow("webcam", webcam_stream.read()) if with_image else None
    # press 'q' to stop the assistant
    if cv2.waitKey(1) in [27, ord("q")] or keyboard.is_pressed("q"):
        break
    # press 't' to add a prompt manually,
    elif cv2.waitKey(1) in [ord("t")] or keyboard.is_pressed("t"):
        # wait until the user press 'Enter' to continue
        while True:
            prompt = input("Enter a prompt: ")
            if prompt:
                assistant.answer(prompt, webcam_stream.read(encode=True)) if with_image else assistant.answer(prompt)
                break
        

stop_listening(wait_for_stop=False)

# save the conversation history into a markdown file
assistant.save_history("history.md")
