# You Talk Too Much App

Feeling like you can't keep up with all the meeting discussions lately? There's an app for that. 

This app will record the conversation, summarize it, and write it to OneNote.

# Getting Started

1. On Mac, install portaudio via brew. See https://stackoverflow.com/questions/33513522/when-installing-pyaudio-pip-cannot-find-portaudio-h-in-usr-local-include
2. Clone the repo.
3. Under [applescript/](applescript/) folder, use the app file to run the app.
4. ???
5. Profit.

# How does it Work

```mermaid
flowchart TD
    A[User] -- runs --> APP[App]
    APP -- opens iTerm terminal and displays --> MENU[ASCII Menu]

    MENU -- Press '1' --> DEC1{Audio Capture\nNot Running?}
    DEC1 -- Y --> 1A[Start Speech Recording]
    DEC1 -- N --> 1B[Do Nothing]

    MENU -- Press '2' --> DEC2{Audio Capture\n Running?}

    DEC2 -- Y --> 2A[Stop Speech Recording]
    2A --> 2B[Export Speech to File]
    2B --> 2C[Export Raw Data to File]
    2C --> 2D[Summarize Speech]
    2D --> 2E[Write to OneNote]
    2E --> DEC3{2 or 3 \nPressed?}
    DEC3 -- 2 --> MENU
    DEC3 -- 3 --> 3A[Quit]

    DEC2 -- N --> DEC3

    MENU -- Press '3' --> DEC2
```
