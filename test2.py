import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QTextEdit
from PyQt5.QtGui import QFont
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

class YouTubeTranscriptApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('YouTube Transcript and Summarization')
        self.setGeometry(100, 100, 800, 600)

        # Fonts
        font = QFont("Poppins", 12)

        # Widgets
        self.url_label = QLabel('Enter YouTube URL:')
        self.url_entry = QLineEdit()
        self.process_button = QPushButton('Process')
        self.transcript_label = QLabel('Transcript:')
        self.transcript_text = QTextEdit()
        self.summary_label = QLabel('Summary:')
        self.summary_text = QTextEdit()

        # Styling
        self.url_label.setFont(font)
        self.url_entry.setFont(font)
        self.process_button.setFont(font)
        self.transcript_label.setFont(font)
        self.transcript_text.setFont(font)
        self.summary_label.setFont(font)
        self.summary_text.setFont(font)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.url_label)
        layout.addWidget(self.url_entry)
        layout.addWidget(self.process_button)
        layout.addWidget(self.transcript_label)
        layout.addWidget(self.transcript_text)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.summary_text)

        self.setLayout(layout)

        # Connect button click event to function
        self.process_button.clicked.connect(self.process_video)

        self.show()

    def process_video(self):
        video_url = self.url_entry.text()

        if video_url:
            result = self.transcribe_and_summarize(video_url)
            transcript = "\n".join([segment["text"] for segment in result["transcript"]])
            summary = result["summary"]

            # Display results in QTextEdit widgets
            self.transcript_text.setPlainText(transcript)
            self.summary_text.setPlainText(summary)
        else:
            self.transcript_text.setPlainText("Please enter a valid YouTube URL.")
            self.summary_text.setPlainText("")

    def transcribe_and_summarize(self, video_url, summary_length=0.2):
        video_id = video_url.split("=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([segment["text"] for segment in transcript])

        # Split transcript into chunks of a certain length (e.g., 512 tokens)
        chunk_size = 512
        chunks = [transcript_text[i:i + chunk_size] for i in range(0, len(transcript_text), chunk_size)]

        # Initialize summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


        # Summarize each chunk and concatenate the results
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, min_length=int(len(chunk) * summary_length),
                                 max_length=int(len(chunk) * summary_length) + 10)[0]["summary_text"]
            summaries.append(summary)

        # Concatenate the summaries
        summary = " ".join(summaries)

        return {"transcript": transcript, "summary": summary}

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YouTubeTranscriptApp()
    sys.exit(app.exec_())
