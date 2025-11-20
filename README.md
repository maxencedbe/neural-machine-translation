# Neural Machine Translation with Seq2Seq & Attention 🇬🇧 🇫🇷

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)

This repository contains a "from-scratch" implementation of a **Neural Machine Translation (NMT)** system capable of translating English sentences into French.

This project was developed as part of the **ALTeGraD (Advanced Learning for Text and Graph Data)** course, part of the **Master MVA (Mathématiques Vision Apprentissage)** at **École Polytechnique** within the *Institut Polytechnique de Paris*. It explores Recurrent Neural Networks (RNNs) and the implementation of Global Attention mechanisms.

## 🧠 Model Architecture

The model is built using **PyTorch** and follows the **Sequence-to-Sequence (Seq2Seq)** paradigm with the following components:

* **Encoder:** A non-stacked unidirectional **GRU** (Gated Recurrent Unit) that processes the source sentence.
* **Decoder:** A GRU-based language model conditioned on the source context.
* **Attention Mechanism:** Implementation of **Global Attention** (using the *concat* scoring function) as proposed by [Luong et al. (2015)](https://arxiv.org/abs/1508.04025). This allows the decoder to focus on different parts of the source sentence at each generation step.
* **Embeddings:** Learned from scratch during training.

## 📂 Project Structure

The implementation is consolidated in a Jupyter Notebook that covers:

1.  **Data Preprocessing:** Handling vocabulary, padding, and special tokens (`<SOS>`, `<EOS>`, `<PAD>`, `<OOV>`).
2.  **Model Implementation:** Custom `nn.Module` classes for the Encoder, Decoder, and Attention.
3.  **Training Loop:** SGD optimization using CrossEntropyLoss.
4.  **Inference:** Greedy decoding strategy to generate translations.
5.  **Visualization:** Plotting alignment weights (attention maps).

## 📊 Dataset

* **Source:** [Tatoeba Project](https://tatoeba.org/) (via http://www.manythings.org/anki/).
* **Size:** ~136k training pairs, ~34k testing pairs.
* **Vocabulary:** ~5,000 English words, ~7,000 French words.

## 🚀 Key Results

The model successfully learns syntactic alignments, such as the inversion of nouns and adjectives between English and French.

### Example:
> **Input:** "I have a red car."
> **Output:** "j ai une voiture rouge ."

During inference, the model computes **attention weights** ($\alpha_{t}$), determining which source words the decoder focuses on when generating each target word. By analyzing these weights, we can confirm the model learns correct word-to-word alignment (e.g., matching "red" with "rouge" and "car" with "voiture").

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/maxencedbe/neural-machine-translation-rnn.git](https://github.com/maxencedbe/neural-machine-translation-rnn.git)
    cd neural-machine-translation-rnn
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch numpy matplotlib nltk tqdm
    ```

3.  **Run the notebook:**
    Open `notebooks/NMT_Seq2Seq_Attention.ipynb` in Jupyter or Google Colab to train the model or run inference using pre-trained weights.

## 📚 Theoretical Insights

The project includes a detailed analysis of NMT limitations:
* **Greedy Decoding vs. Beam Search:** Discussion on why greedy methods are locally optimal but globally suboptimal.
* **The Coverage Problem:** Analyzing repetition and omission errors inherent to standard attention mechanisms.
* **Contextual Embeddings:** Comparison with modern approaches (ELMo, BERT) to handle polysemy (e.g., the word "mean" in different contexts).

---
*This lab was originally designed by Prof. Michalis Vazirgiannis, Dr. Hadi Abdine, and Yang Zhang for the APM 53674 course.*