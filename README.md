
Movie Script Generator

    A custom Small Language Model (SLM) trained on 5,000 movie scripts and 10+ million lines of dialogue to generate cinematic scenes.

<img width="1029" height="963" alt="image" src="https://github.com/user-attachments/assets/0971f56f-fd31-4084-9448-41202297aae4" />



About The Project

    This project is a full-stack generative AI application powered by a custom-trained Transformer model. Unlike wrappers around existing APIs (like OpenAI), this utilizes a custom GPT architecture trained specifically on screenplay formatting and dialogue.

    The model was trained on a massive dataset of 5,000 movie scripts, parsing over 10 million individual lines of dialogue to learn the nuances of character interaction, scene setting, and dramatic pacing.

Key Features

    Massive Dataset: Trained on 10M+ lines of dialogue from 5,000 Hollywood scripts.

    Custom GPT Architecture: Implements Causal Self-Attention, LayerNorm, and Multi-Head Attention from scratch using PyTorch.

    Full-Stack Integration: A modern React frontend communicating via the Gradio Client with a Python backend.

    Optimized Inference: Runs locally on NVIDIA GPUs using Mixed Precision (Autocast) for rapid token generation.

    Creative Controls:

        Temperature: Control the "creativity" or randomness of the output.

        Top-K Sampling: Limit predictions to the top K likely next tokens.

        Max Tokens: Define the length of the generated scene.

🛠️ Tech Stack

Machine Learning & Backend

    Python 3.11

    PyTorch (CUDA) - Custom Transformer implementation.

    Gradio - API serving and model interface.

    TikToken - Efficient tokenization.

Frontend

    React.js

    Vite

    @gradio/client - Bridge between React and the Python backend.

    CSS Modules/Glassmorphism - Custom UI design.

 The Model Architecture

The core is a decoder-only Transformer (GPT-style) featuring:

    Context Window: Trained to handle standard scene lengths.

    Positional Embeddings: Learnable positional encodings for sequence order.

    Optimization: Weights saved and loaded securely; inference optimized with torch.float16.

Installation & Usage

Prerequisites

    Python 3.11+

    Node.js & npm

    


Dataset Stats

    Source: 5,000+ Movie Scripts.

    Volume: >10,000,000 lines of dialogue.

    Preprocessing: Cleaned and formatted for screenplay structure (Scene Headings, Character Names, Dialogue).

Future Improvements

    Fine tune further to generate a more cohesive dialog. 
    
    Implement "Stop Sequences" to prevent mid-sentence cutoffs.

    Fine-tune on specific genres (Horror, Sci-Fi, etc.).

License

Distributed under the MIT License. See LICENSE for more information.
