# you-talk-too-much-app

```mermaid
flowchart TD
    A[User] -- uses --> B[App]
    B --> C[Record Speech]
    C --> D[Export Speech to File]
    C --> E[Export Raw Data to File]
    D --> F[Summarize Speech]
    C --> H[Generate Smart Questions Based on Recent Speech]
    F --> G[Write to OneNote]
```