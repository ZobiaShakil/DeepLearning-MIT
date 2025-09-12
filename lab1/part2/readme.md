
#  AI Music Generation with RNNs

This project is an implementation of a Recurrent Neural Network (RNN) trained to generate classical piano music in the style of Bach chorales. It was built as part of the MIT Introduction to Deep Learning (6.S191) course.

##  Listen to the AI's Music
The model has successfully generated original musical compositions. Check out the `music(0,1,2)` files in this directory to hear the results!

## Project Overview

This model is a character-level RNN that learns to predict the next character in a sequence of ABC notation, a text-based music representation. By learning from the patterns in Bach's music, it can generate new, original musical scores.

**Key Techniques Used:**
-   **Recurrent Neural Network (RNN)** with LSTM cells for sequence modeling.
-   **Character-level text generation** applied to musical notation.
-   **Hyperparameter optimization** (learning rate, hidden size, sequence length) to achieve coherent output.
-   **Temperature sampling** to control the creativity vs. coherence of the generated music.

##  Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ZobiaShakil/DeepLearning-MIT.git
    cd DeepLearning-MIT/lab1
    ```

##  How to Run

1.  **Run the notebook cells in sequence.** The notebook is structured to:
    -   Install dependencies and configure the environment.
    -   Load and preprocess the ABC music dataset.
    -   Define and train the RNN model.
    -   Generate new music from the trained model.
    -   Synthesize the generated ABC text into playable audio files (`.wav`).

2.  **Training:** The model will train for a predefined number of iterations. You can monitor the training progress through the plotted loss curve.

3.  **Generation:** After training, the final cells will generate multiple musical sequences and save them as audio files.

## Results

After tuning hyperparameters and the text generation function, the model successfully learned the structure of ABC notation and began producing coherent musical pieces. The key to success was:
-   i changed the iterations to 10000 from 3000
-   batchsize from 16 to 32
-   sequencelength from 50 to 100 and learningrate to 1e - 3

Key Training Metrics (Logged via Comet.ml):

Final Loss: ~0.15

Initial Loss: ~4.43

Total Training Iterations: 10,000

Total Training Duration: ~776 seconds (~13 minutes)

Loss Curve Analysis:
The model's loss dropped dramatically from an initial value of 4.43 to a final value of ~0.15, demonstrating a strong ability to learn the patterns and structure of the musical ABC notation.
<img width="371" height="369" alt="image" src="https://github.com/user-attachments/assets/f2bca56d-a154-49ce-92c6-56177eb33c2f" />

