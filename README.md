# clip_for_goods

# Semantic Fashion Search with Fine-Tuned CLIP

This project implements a text-to-image search engine specifically tailored for the fashion domain. By fine-tuning OpenAI's **CLIP (Contrastive Language-Image Pre-Training)** model, the system understands fashion-specific attributes (textures, styles, cuts) significantly better than the base model.

![Search Result Demo](assets/search_demo.png)
*(Search results demonstrating the model's ability to link text queries to product images)*

## 🚀 Key Features

* **Domain Adaptation:** Fine-tuned `clip-vit-base-patch32` on a fashion product dataset to align visual and textual representations.
* **High Accuracy:** Achieved a **CLIP Score > 30** on the validation set, ensuring high confidence in matching correct pairs.
* **Vector Search Engine:** Implemented an indexing system that pre-computes image embeddings for instant retrieval.
* **Contrastive Learning:** Utilized symmetric cross-entropy loss to maximize cosine similarity between matching image-text pairs.

## 🛠️ Tech Stack

* **Deep Learning:** PyTorch, Hugging Face Transformers
* **Data Processing:** Pandas, PIL (Python Imaging Library)
* **Visualization:** Matplotlib
* **Environment:** Google Colab (GPU)

## 📊 Methodology

1.  **Data Preparation:** * Cleaned and preprocessed the Fashion Product Images dataset.
    * Implemented a custom PyTorch `Dataset` class to handle image loading and tokenization on the fly.
2.  **Fine-Tuning:** * Model: `openai/clip-vit-base-patch32`.
    * Optimizer: AdamW (`lr=5e-5`).
    * Loss Function: Contrastive Loss (Symmetric Cross-Entropy).
    * Training Duration: 4 Epochs (converged with stable validation metrics).
3.  **Inference & Search:**
    * Generated embeddings for the entire image catalog.
    * Performed Nearest Neighbor search using Cosine Similarity between the query vector and image vectors.

## 📈 Results

The model demonstrates a strong ability to understand specific fashion queries.
* **Baseline:** The pre-trained model had a general understanding of objects.
* **Fine-tuned:** The model distinguishes between nuanced queries (e.g., "floral dress" vs. "striped dress") with high confidence scores (**>30 logits**).
