# Hate Speech Detection with HateBERT

This project builds upon the research paper introducing HateBERT, which demonstrated improved performance over general language models in hate speech and abusive language detection tasks.


Dataset Used: Hate Speech and Offensive Language Dataset (Not present here but will work for any similar dataset)

The dataset consists of 24,783 Twitter posts labeled for hate speech detection.
Each tweet is classified into three categories:
Hate Speech (label 0): 1,430 tweets
Offensive Language (label 1): 19,190 tweets

This dataset is used to develop and evaluate models that detect hate speech and offensive language, enhancing the ability to create safer online environments.

Tweet Text: The content of the Twitter post
Class: The classification label (hate speech, offensive language, or neither).


1. Data Preparation:

Dataset Cleaning: The dataset was preprocessed by removing missing values, cleaning text (removing URLs, mentions, hashtags, special characters, etc.), and converting text to lowercase.

Label Mapping: Labels were standardized into three categories: "hate speech," "offensive language," and "neither."

Data Augmentation: Synonym-based augmentation was performed for underrepresented hate speech examples to balance class distribution.

2. Model Selection and Tokenization:

Used HateBERT, a pre-trained BERT model tailored for hate speech detection.

Tokenized text data using Hugging Face’s AutoTokenizer for compatibility with the BERT model.

3. Training Strategy:

Split data into training, validation, and test sets to ensure robust evaluation. Training (20,073 examples), Validation (2,231 examples), and Test (2,479 examples) sets.

Fine-tuned HateBERT with class weights to handle imbalanced data.

Applied weighted loss to mitigate the effects of class imbalance during training.

Defined a custom metric function to calculate accuracy, precision, recall, and F1 score.

Evaluated the model on the test set both before and after data augmentation.


Introduced augmented data to enhance the detection of hate speech (synonym-based augmentation).Hate Speech: 8,321 examplesOffensive Language: 19,190 examples
Neither: 1,430 examples

Fine-tuned the model further using the augmented dataset and re-evaluated performance.




