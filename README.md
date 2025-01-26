Supervised Fine-Tuning (SFT) for Sentiment Analysis
Overview


![fb9d0144-3952-42db-8382-8e2eb37d917e_1670x640](https://github.com/user-attachments/assets/8572789f-56ca-4541-8e7d-2e486d9f4373)

This project demonstrates Supervised Fine-Tuning (SFT) of a pre-trained DistilBERT model (distilbert-base-uncased) on the IMDB movie review dataset for binary sentiment classification (positive or negative). The SFT process adapts a general-purpose language model to perform a specialized task with high accuracy.
Objective

The goal of the SFT phase is to:

    Fine-tune the DistilBERT model to classify movie reviews as positive or negative.
    Use the IMDB dataset to adapt the model to the specific task of sentiment analysis.
    Create a baseline model that serves as the foundation for further optimization (e.g., RLHF).

Dataset

We used the IMDB dataset, which contains:

    25,000 training samples of labeled movie reviews.
    25,000 test samples for evaluation.

The dataset is loaded and processed using the Hugging Face datasets library.
Process
1. Data Preparation

    Tokenized the IMDB reviews using the DistilBERT tokenizer.
    Applied padding and truncation to ensure a consistent sequence length of 512 tokens.

2. Model Initialization

    Used the pre-trained DistilBERT model (distilbert-base-uncased).
    Added a classification head for binary classification (num_labels=2).

3. Training Configuration

    Optimizer: AdamW.
    Loss Function: CrossEntropyLoss.
    Learning Rate: 2e-5.
    Batch Size: 16.
    Epochs: 2–3 (adjustable based on performance).
    Device: CUDA (GPU) for accelerated training.

4. Checkpointing

    Checkpoints were saved at the end of each epoch to allow resumption of training in case of interruptions.

5. Evaluation

    Evaluated the model on the IMDB test set after each epoch.
    Achieved ~90% validation accuracy.

Key Results

    Fine-tuned DistilBERT achieved high accuracy (~90%) on the IMDB test dataset.
    Created a specialized sentiment analysis model based on general-purpose pre-trained weights.
    Enabled resumption of training through effective checkpointing.

How to Run the SFT Pipeline
Prerequisites

    Python 3.8+
    A virtual environment with the following installed:
        transformers
        datasets
        torch
        tqdm

Install required packages:

pip install transformers datasets torch tqdm

Steps to Fine-Tune

    Clone the repository and navigate to the project folder:

cd Fine_tuning_projectg

Run the fine-tuning script:

python main.py

To resume training from a checkpoint, use the --resume flag:

    python main.py --resume --resume_checkpoint ./checkpoints/model_epoch_1.pt

Folder Structure

Fine_tuning_projectg/
├── data/
│   ├── raw/                 # Raw IMDB dataset files (if applicable)
│   ├── processed/           # Tokenized data
├── checkpoints/             # Model checkpoints saved during training
├── microservices/
│   ├── data_service/        # Data loading and preprocessing scripts
│   ├── model_service/       # Model definition and training scripts
│   └── orchestrator/        # Pipeline to coordinate the process
├── main.py                  # Entry point for the SFT process
└── README.md                # Documentation (this file)

Future Enhancements

    Reinforcement Learning from Human Feedback (RLHF): Optimize the fine-tuned model further using human preferences.
    Hyperparameter Tuning: Experiment with learning rates, batch sizes, and more for improved performance.
    Model Deployment: Deploy the fine-tuned model for real-time sentiment analysis.

Contact

For questions or feedback, please contact [Rakshit Bhardwaj] at [rbhardwaj2186@gmail.com].
