# You Talk Too Much App 🎙️

Feeling like you can't keep up with all the meeting discussions lately?

This app records, transcribes, and summarizes your conversations using state-of-the-art ML models, then automatically syncs the summary to your Microsoft OneNote.

## Features

- ⚡ **Local High-Performance Transcription**: Uses `mlx-whisper` optimized for Apple Silicon.
- 👥 **Speaker Diarization**: Identifies who said what using `pyannote-audio` 4.x.
- 🔍 **Global Speaker Tracking**: Tracks speakers across different segments within a session using embedding similarity.
- 🧠 **AI-Powered Summarization**: Generates professional meeting notes, executive summaries, and action items via Vertex AI (Gemini).
- 📓 **OneNote Integration**: Automatically creates a new page in your specified OneNote section with the formatted summary.
- 💻 **Clean CLI Interface**: Real-time conversation logging with proper speaker labels and alignment.

## Tech Stack

- **Transcription**: `mlx-whisper`
- **Diarization**: `pyannote-audio` (v4.0.4+)
- **LLM**: Google Vertex AI (Gemini 2.5 Flash)
- **Audio**: `sounddevice` & `silero-vad`
- **Dependency Management**: `uv`

## How it Works

```mermaid
flowchart TD
   A[User]:::defClass
   APP[App]:::appClass
   MENU[CLI Menu]:::appClass

   STEP1A[Start Audio Capture]:::toolClass
   STEP1B[VAD-based Batching]:::toolClass
   STEP1C[MLX-Whisper Transcription]:::toolClass
   STEP1D[Pyannote Diarization]:::toolClass
   STEP1E[Log Conversation]:::toolClass

   STEP2A[Stop Audio Capture]:::toolClass
   STEP2B[Export Final Transcript]:::toolClass
   STEP2C[Vertex AI Summarization]:::toolClass
   STEP2D[Upload to OneNote]:::toolClass

   FILESYSTEM:::artifactClass@{ shape: lin-cyl, label: "fa:fa-file Raw Conversation" }
   ONENOTE:::artifactClass@{ shape: lin-cyl, label: "fa:fa-file MS OneNote" }

   A -- runs --> APP
   APP -- displays --> MENU

   MENU -- Press '1' --> DEC1{Recording?}:::defClass
   DEC1 -- N --> STEP1A
   STEP1A --> STEP1B
   STEP1B --> STEP1C
   STEP1C --> STEP1D
   STEP1D --> STEP1E
   STEP1E --> FILESYSTEM

   MENU -- Press '2' --> DEC2{Recording?}:::defClass
   DEC2 -- Y --> STEP2A
   STEP2A --> STEP2B
   STEP2B --> STEP2C
   STEP2C --> STEP2D
   STEP2D --> ONENOTE
   STEP2D --> MENU

   MENU -- Press '3' --> STEP3A[Quit]:::toolClass

   line@{ animate: true }

   classDef defClass fill:#FFFFFF,stroke:#666666,color:#666666
   classDef appClass fill:lightgreen,stroke:green,color:#666666
   classDef toolClass fill:lightblue,stroke:blue,color:#666666
   classDef artifactClass fill:#CCCCCC,stroke:#666666,color:#666666
```

```mermaid
flowchart TD
    B:::userClass@{ shape: processes, label: "fa:fa-user Manager" }
    U["fa:fa-user Employee"]:::userClass
    AR["fa:fa-robot Root Agent"]:::agentClass
    AS:::agentClass@{ shape: processes, label: "fa:fa-robot Solution Architecture Agent" }
    AC:::agentClass@{ shape: processes, label: "fa:fa-robot C4 Agent" }
    DM:::artifactClass@{ shape: lin-cyl, label: "fa:fa-file 1 x Solution Architecture\nDocument" }
    DD:::artifactClass@{ shape: lin-cyl, label: "fa:fa-file 1 x C4 Context Diagram \n 2 x C4 Container Diagrams" }
    TL:::toolClass@{ shape: processes, label: "fa:fa-hammer Local Tool" }
    TM("fa:fa-hammer Mermaid MCP Tool"):::toolClass

    U--impresses-->B
    B--gives job back-->U
    U e0@--interacts-->AR
    AR e1@--> AS
    AR e2@--> AC
    AS e3@--writes--> DM
    AS --uses--> TL
    AC --uses--> TL
    AC --uses--> TM
    AC e4@--writes--> DD

    e0@{ animate: true }
    e1@{ animate: true }
    e2@{ animate: true }
    e3@{ animate: true }
    e4@{ animate: true }

    classDef userClass fill:#FFFFFF,stroke:#666666,color:#666666
    classDef toolClass fill:lightblue,stroke:blue,color:#666666
    classDef agentClass fill:lightgreen,stroke:green,color:#666666
    classDef artifactClass fill:#CCCCCC,stroke:#666666,color:#666666
```

## Prerequisites

1. **Hardware**: Apple Silicon Mac (M3, etc) is highly recommended for `mlx` performance.
2. **System Libraries**: Install `portaudio` via Homebrew:
   ```bash
   brew install portaudio
   ```
3. **API Access**:
   - **Hugging Face**: A token with read access. You **must** visit the following model pages and "Accept and Approve" their terms while logged in:
     - [mlx-community/whisper-large-v3-mlx](https://huggingface.co/mlx-community/whisper-large-v3-mlx)
     - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
     - [pyannote/embedding](https://huggingface.co/pyannote/embedding)
   - **Google Cloud**: A Vertex AI project and a service account key JSON file.
   - **Microsoft Azure**: An App Registration for OneNote API access (`Notes.ReadWrite.All`).

## Installation

1. Install dependencies using `uv`:

   ```bash
   uv sync
   ```

2. Set up your environment variables by copying `.env.sample` to `.env` and filling in your details:
   ```bash
   cp .env.sample .env
   ```

## Usage

### Run via Command Line

```bash
uv run you-talk-too-much
```

### Run via AppleScript App

Under the [applescript/](applescript) folder, you can use the bundled `.app` file to launch the application in a dedicated iTerm window.

### Menu Options

1. **Start new capture**: Begins recording audio and processing it in real-time batches.
2. **Stop existing capture**: Finalizes the current session, generates the summary, and uploads it to OneNote.
3. **Quit program**: Exits the application.

## License

MIT
