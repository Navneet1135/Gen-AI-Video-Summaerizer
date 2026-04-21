import yt_dlp
from faster_whisper import WhisperModel
from transformers import pipeline
import gradio as gr
import whisper

model = whisper.load_model("base") 
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def download_youtube_audio(youtube_url):
    output_file = "video_audio.mp3"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'video_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    return output_file

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    
    segments = result["segments"]
    transcript = " ".join(segment["text"] for segment in segments)

    return transcript

def chunk_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def summarize_text(text):

    # Split text into chunks to avoid token limit issues
    text_chunks = chunk_text(text, chunk_size=400)

    summaries = []
    for chunk in text_chunks:
        summary = summarizer(chunk, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    # Combine all summaries into a final text
    full_summary = " ".join(summaries)

    return full_summary

def process_video(youtube_url):
    # Step 1: Download audio
    audio_path = download_youtube_audio(youtube_url)

    # Step 2: Convert audio to text
    transcript = transcribe_audio(audio_path)

    # Step 3: Summarize text
    summary = summarize_text(transcript)

    return summary

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Generative AI Video Summarizer")
    gr.Markdown("### Paste a YouTube link to generate a summarized transcript.")

    youtube_url = gr.Textbox(label="YouTube Video URL")
    output_text = gr.Textbox(label="Summarized Key Points", lines=10)

    summarize_button = gr.Button("Summarize Video")
    summarize_button.click(process_video, inputs=[youtube_url], outputs=[output_text])

demo.launch(share = True, debug = True)
