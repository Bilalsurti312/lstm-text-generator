# Shakespeare Text Generation with LSTM

This project is a character-level text generator trained on William Shakespeare's works using TensorFlow and Keras. It learns the style, vocabulary, and structure of Shakespeare's writing and generates new text from a given seed line.


## What Does This Project Do?

1. **Loads and cleans text**: Reads Shakespeare's plays, converts everything to lowercase, strips punctuation, and builds a character vocabulary (28 unique characters: `a-z`, space, and newline).
2. **Prepares training sequences**: Breaks the text into overlapping chunks of 50 characters to predict the 51st character (over 95,000 sequence pairs created).
3. **Trains an LSTM model**: Uses an Embedding layer followed by two stacked LSTM layers with Dropout to learn patterns without overfitting.
4. **Generates text**: Takes any starting sentence (seed) and generates new text character by character using temperature sampling.


## Project Structure
├── artifacts/
│   ├── best_lstm_model.keras        # Best model saved during training
│   ├── char_to_idx.json             # Character to number mapping
│   └── idx_to_char.json             # Number to character mapping
├── notebooks/
│   └── lstm_text_generation.ipynb   # Main notebook with code, training, and plots
├── outputs/
│   ├── sample_generations.txt       # Generated text outputs
│   └── training_curves.png          # Loss and accuracy graphs
├── requirements.txt                 # Required libraries
└── README.md


## Dataset

* **Source**: Shakespeare's Complete Works from [Project Gutenberg](https://www.gutenberg.org/ebooks/100).
* **Direct link**: [shakespeare.txt (100-0.txt)](https://www.gutenberg.org/files/100/100-0.txt)
* **Characters used**: First ~286,000 characters (sufficient for quick, coherent training).


## Model Architecture

* **Input**: Sequence of 50 character indices
* **Embedding Layer**: 64 dimensions
* **LSTM Layer 1**: 128 units (`return_sequences=True`)
* **Dropout**: 0.2 (prevents overfitting)
* **LSTM Layer 2**: 128 units
* **Dropout**: 0.2
* **Dense Layer**: Softmax output over 28 vocabulary characters
* **Total Parameters**: ~235,000


## Training Results

The model was trained for 12 epochs with a batch size of 128 (90% train / 10% validation split):

* **Loss**: Dropped smoothly from **2.40** down to **1.43** (validation loss: **1.54**).
* **Accuracy**: Reached **~52%** on validation data (which is a strong benchmark for next-character language modeling).
* **No Overfitting**: Validation loss closely tracked training loss throughout, showing that the dropout layers worked properly.

The training curves are saved in `outputs/training_curves.png`.


## Sample Outputs

We tested the model with 3 different seed prompts across low, medium, and high temperatures:

### Prompt: *"shall i compare thee to a summers day"* (Temp: 0.5)
> `shall i compare thee to a summers day`  
> `cleopatra`  
> `for the like the irmis with her love with him can strange shall well foul men to the hongiers being done with the beauty he hand`  
> `countess`  
> `yet i be her love dear is dear will seen me which`

### Prompt: *"friends romans countrymen lend me your ears"* (Temp: 0.5)
> `friends romans countrymen lend me your earser in his love in seen`  
> `would have he is all the dear caesar and excellenter of my gracely we are some to my love the agriment may i the wide the ring and beauty bears all the sur our not the count`

### Prompt: *"to be or not to be that is the question"* (Temp: 0.8)
> `to be or not to be that is the question of with the sond in not that doth live the kingg and countloor and bear my beauty your good of the world to against sir you do forget no set of he my done but my love me wibeth who have to the praise`

**Key Observations**:
* At **temperature 0.2**, the text is safe and creates proper English words, but repeats phrases.
* At **temperature 0.5**, the output is well-balanced. Interestingly, the model even learned theatrical formatting on its own—putting character names like `cleopatra` and `countess` on new lines followed by dialogues.
* At **temperature 0.8**, the output is much more creative and varied, though it occasionally invents archaic-sounding words.

All test outputs are stored in `outputs/sample_generations.txt`.


## Bonus: Architectural Experiments

1. **Sequence Length**: 
   * When tested with a shorter sequence length (25 characters), the model struggled with longer sentence context.
   * Increasing to 50 characters gave the model enough context to learn word combinations, spacing, and play dialogues.
2. **Number of LSTM Layers**: 
   * A single LSTM layer stalled around 1.78 loss and produced broken syllables.
   * Stacking 2 LSTM layers with dropout allowed the model to reach 1.54 loss and learn both word structure and line breaks.


## How to Run

1. Clone this repository:
   git clone <your-repo-link>
   cd <repo-folder>

2. Install dependencies:
   pip install -r requirements.txt

3. Open and run the notebook:
   jupyter notebook notebooks/lstm_text_generation.ipynb