import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups


class TextClassifier:
    def __init__(self):
        """
        Initializes the text classification pipeline.
        The pipeline consists of a TF-IDF vectorizer and a Multinomial Naive Bayes classifier.
        """
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=3, max_df=0.95)),
            ('clf', MultinomialNB(alpha=0.1)),
        ])
        self.target_names = None
        self.is_fitted = False

    def train(self, texts, labels, target_names=None):
        """
        Trains the text classification model.

        Args:
            texts (list): A list of text documents (strings).
            labels (list): A list of corresponding labels (integers or strings).
            target_names (list, optional): A list of human-readable names for the categories.
                                           Defaults to None.
        """
        if not texts or not labels:
            raise ValueError("Texts and labels cannot be empty for training.")
        if len(texts) != len(labels):
            raise ValueError("The number of texts and labels must be the same.")

        print(f"Starting training with {len(texts)} documents.")
        self.model.fit(texts, labels)
        self.target_names = target_names
        self.is_fitted = True
        print("Training complete.")

    def predict(self, texts):
        """
        Predicts the categories for a list of text documents.

        Args:
            texts (list): A list of text documents (strings) to classify.

        Returns:
            list: A list of predicted category labels.
            list: A list of predicted category names if target_names were provided during training.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "TextClassifier is not fitted. Call train() or load_model() successfully before predicting."
            )
        if not texts:
            return [], []

        predictions = self.model.predict(texts)
        if self.target_names:
            predicted_names = [self.target_names[p] for p in predictions]
            return predictions, predicted_names
        return predictions, [str(p) for p in predictions]

    def predict_proba(self, texts):
        """
        Predicts the probability of each category for a list of text documents.

        Args:
            texts (list): A list of text documents (strings) to classify.

        Returns:
            numpy.ndarray: An array where rows correspond to documents and columns to classes,
                           containing the probability of each class.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "TextClassifier is not fitted. Call train() or load_model() successfully before predicting."
            )
        if not texts:
            return []
        return self.model.predict_proba(texts)

    def save_model(self, path="text_classifier_model.joblib"):
        """
        Saves the trained model to a file.

        Args:
            path (str): The path to save the model file.
        """
        if not self.is_fitted:
            print("Warning: Attempting to save a model that has not been fitted.")
        joblib.dump({'model': self.model, 'target_names': self.target_names, 'is_fitted': self.is_fitted}, path)
        print(f"Model saved to {path}")

    def load_model(self, path="text_classifier_model.joblib"):
        """
        Loads a trained model from a file.

        Args:
            path (str): The path to the model file.
        """
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.target_names = data['target_names']
            self.is_fitted = data.get('is_fitted', True)
            if not self.is_fitted:
                 print(f"Warning: Loaded model from {path} was marked as not fitted.")
            else:
                print(f"Model loaded from {path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}. A new model will be created if you train.")
            self.is_fitted = False
        except Exception as e:
            print(f"Error loading model: {e}. A new model will be created if you train.")
            self.is_fitted = False


def load_and_preprocess_data(categories=None, subset='train'):
    """
    Loads and preprocesses the 20 Newsgroups dataset.

    Args:
        categories (list, optional): Specific categories to load. If None, loads a few common ones.
        subset (str): 'train', 'test', or 'all'.

    Returns:
        tuple: (texts, labels, target_names)
    """

    print(f"Loading {subset} data for categories: {categories}")
    try:
        dataset = fetch_20newsgroups(subset=subset, categories=categories,
                                     shuffle=True, random_state=42,
                                     remove=('headers', 'footers', 'quotes'))

        texts = []
        labels = []
        for i, doc in enumerate(dataset.data):
            if doc and len(doc.strip()) > 20:
                texts.append(doc.strip())
                labels.append(dataset.target[i])

        print(f"Loaded {len(texts)} documents for {subset} set.")
        return texts, labels, dataset.target_names
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], [], []


if __name__ == "__main__":
    classifier = TextClassifier()

    category_map = {
        'comp.graphics': 'Computing & Technology',
        'comp.os.ms-windows.misc': 'Computing & Technology',
        'comp.sys.ibm.pc.hardware': 'Computing & Technology',
        'comp.sys.mac.hardware': 'Computing & Technology',
        'comp.windows.x': 'Computing & Technology',
        'sci.crypt': 'Science & Medicine',
        'sci.electronics': 'Science & Medicine',
        'sci.med': 'Science & Medicine',
        'sci.space': 'Science & Medicine',
        'rec.autos': 'Recreation & Sports',
        'rec.motorcycles': 'Recreation & Sports',
        'rec.sport.baseball': 'Recreation & Sports',
        'rec.sport.hockey': 'Recreation & Sports',
        'alt.atheism': 'Religion',
        'soc.religion.christian': 'Religion',
        'talk.religion.misc': 'Religion',
        'talk.politics.guns': 'Politics',
        'talk.politics.mideast': 'Politics',
        'talk.politics.misc': 'Politics',
        'misc.forsale': 'Commerce & Sales'
    }

    custom_target_names = sorted(list(set(category_map.values())))
    print(f"Custom target categories: {custom_target_names}")

    original_categories_to_load = list(category_map.keys())

    train_texts, original_train_labels, original_train_target_names = \
        load_and_preprocess_data(categories=original_categories_to_load, subset='train')
    test_texts, original_test_labels, original_test_target_names = \
        load_and_preprocess_data(categories=original_categories_to_load, subset='test')

    if not train_texts or not original_train_labels:
        print("Failed to load training data or labels. Exiting example.")
    elif not test_texts or not original_test_labels:
        print("Failed to load test data or labels. Exiting example.")
    else:
        def map_labels_to_custom(original_labels, fetched_original_target_names,
                                 current_category_map, custom_categories_list):
            new_labels = []
            valid_texts_indices = []
            for i, orig_label_idx in enumerate(original_labels):
                original_category_name = fetched_original_target_names[orig_label_idx]
                custom_category_name = current_category_map.get(original_category_name)
                if custom_category_name:
                    try:
                        new_labels.append(custom_categories_list.index(custom_category_name))
                        valid_texts_indices.append(i)
                    except ValueError:
                        print(f"Warning: Custom category '{custom_category_name}' derived from original "
                              f"'{original_category_name}' not found in custom_target_names list. Skipping.")
                else:
                    print(f"Warning: Original category '{original_category_name}' "
                          f"could not be mapped to a custom category. Skipping this item.")
            return new_labels, valid_texts_indices

        custom_train_labels, valid_train_indices = map_labels_to_custom(
            original_train_labels, original_train_target_names, category_map, custom_target_names
        )
        train_texts = [train_texts[i] for i in valid_train_indices]

        custom_test_labels, valid_test_indices = map_labels_to_custom(
            original_test_labels, original_test_target_names, category_map, custom_target_names
        )
        test_texts = [test_texts[i] for i in valid_test_indices]


        if not custom_train_labels or not train_texts:
            print("No training data/labels after mapping. Cannot train. Exiting.")
        else:
            classifier.train(train_texts, custom_train_labels, target_names=custom_target_names)

            if test_texts and custom_test_labels:
                predicted_labels_numeric, predicted_labels_named = classifier.predict(test_texts)
                print("\nClassification Report on Test Set (Mapped Categories):")
                print(classification_report(custom_test_labels, predicted_labels_numeric,
                                            target_names=classifier.target_names))
            else:
                print("\nNo test data or labels available for evaluation after mapping.")

            classifier.save_model("text_classifier_model.joblib")

            loaded_classifier = TextClassifier()
            loaded_classifier.load_model("text_classifier_model.joblib")

            if loaded_classifier.is_fitted and test_texts:
                sample_texts_to_predict = test_texts[:min(5, len(test_texts))]
                if sample_texts_to_predict:
                    print(f"\nPredicting categories for {len(sample_texts_to_predict)} sample texts using loaded mapped model:")
                    _, predictions_named = loaded_classifier.predict(sample_texts_to_predict)
                    for text, category in zip(sample_texts_to_predict, predictions_named):
                        print(f"\nText: {text[:100]}...")
                        print(f"Predicted Custom Category: {category}")
                else:
                    print("\nNo sample texts from test set to predict with (after filtering).")
            elif not loaded_classifier.is_fitted:
                print("\nLoaded model is not fitted. Cannot predict.")
            else:
                print("\nNo test data was available to demonstrate prediction with the loaded model (after filtering).")