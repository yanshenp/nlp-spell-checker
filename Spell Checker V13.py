import os
import re
import string
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.lm.models import Laplace
from nltk.util import ngrams
from collections import Counter
import fitz  # PyMuPDF
from language_tool_python import LanguageTool

# Directory to store the .pkl file
file_path = 'C:/Users/pangy/Downloads/'  # 'C:/Users/NEW HP/Desktop/Testing'

class SpellingCheckerApp:
    def __init__(self, master):
        self.master = master
        master.title("Spelling and Grammar Checker")

        # Load or preprocess the data
        if self.load_preprocessed_data():
            print("Preprocessed data loaded successfully")
        else:
            print("Preprocessed data not found or could not be loaded, preprocessing...")
            self.preprocess_data()
            print("Preprocessed data loaded successfully")

        # Load or build the bigram model
        if self.load_bigram_model():
            print("Bigram model loaded successfully")
        else:
            print("Bigram model not found or could not be loaded, building...")
            self.build_bigram_model()
            print("Bigram model loaded successfully")

        # LanguageTool instance for grammar checking
        self.grammar_checker = LanguageTool('en-US')

        # Create frames
        self.file_frame = tk.Frame(master)
        self.file_frame.pack(fill=tk.BOTH, expand=False)
        self.input_frame = tk.Frame(master)
        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.suggestion_frame = tk.Frame(master)
        self.suggestion_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # File Management Buttons
        self.save_button = tk.Button(self.file_frame, text="Save", command=self.save_as_file)
        self.save_button.pack(side=tk.LEFT, anchor="nw")
        self.open_button = tk.Button(self.file_frame, text="Open", command=self.open_file)
        self.open_button.pack(side=tk.LEFT, anchor="nw")

        # Status Label
        self.status_label = tk.Label(self.file_frame, text="", padx=20, pady=10)
        self.status_label.pack(pady=10)

        # Input frame widgets
        self.label = tk.Label(self.input_frame, text="Enter text:")
        self.label.pack()
        self.text_entry = tk.Text(self.input_frame, height=10, width=30)
        self.text_entry.pack(fill=tk.BOTH, expand=True)
        self.clear_button = tk.Button(self.input_frame, text="Clear Input", command=self.clear_input)
        self.clear_button.pack()
        self.suggestion_label = tk.Label(self.suggestion_frame, text="Suggestions:")
        self.suggestion_label.pack()
        self.suggestion_text = tk.Text(self.suggestion_frame, height=10, width=50)
        self.suggestion_text.pack(fill=tk.BOTH, expand=True)

        # Frame for search bar and search button
        self.search_frame = tk.Frame(self.suggestion_frame)
        self.search_frame.pack(pady=10)
        # Search label
        self.search_label = tk.Label(self.search_frame, text='Search:', font=('Arial', 12))
        self.search_label.pack(side=tk.LEFT)
        # Search entry field
        self.search_entry = tk.Entry(self.search_frame, width=20, font=('Arial', 10))
        self.search_entry.pack(side=tk.LEFT)
        # Search button
        self.search_button = tk.Button(self.search_frame, text='Search', command=self.search_word, font=('Arial', 12))
        self.search_button.pack(side=tk.LEFT)

        # Show list
        self.show_word_list_button = tk.Button(self.suggestion_frame, text="Show Word List", command=self.show_word_list)
        self.show_word_list_button.pack(side=tk.TOP, pady=10)

        # Maximum word limit
        self.max_word_limit = 500
        # Current word limit
        self.current_word_limit = self.max_word_limit
        self.text_entry.bind("<KeyRelease>", self.on_key_release)  # Bind to key release event

        # Dictionary to store misspelled words and their suggestions
        self.misspelled_words = {}
        self.show_vocab_length_button = tk.Button(self.suggestion_frame, text="Show Vocabulary Length", command=self.show_vocab_length)
        self.show_vocab_length_button.pack(side=tk.TOP, pady=10)

    def preprocess_data(self):
        # Directory containing PDF articles
        pdf_directory = os.path.join(file_path, "Corpus")

        # List to store text extracted from PDF files
        corpus = []

        # Iterate through PDF files
        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                pdf_file_path = os.path.join(pdf_directory, filename)
                # Extract text from PDF
                text = self.extract_text_from_pdf(pdf_file_path)
                corpus.append(text)

        # Preprocess corpus
        self.preprocessed_corpus = self.preprocess_corpus(corpus)

        # Create vocabulary
        self.vocabulary, self.word_counts, self.word_probas = self.create_vocabulary(self.preprocessed_corpus)

        # Save preprocessed data
        self.save_preprocessed_data()

    def extract_text_from_pdf(self, pdf_file_path):
        try:
            text = ''
            with fitz.open(pdf_file_path) as pdf_file:
                for page_num in range(pdf_file.page_count):
                    page = pdf_file.load_page(page_num)
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"Error occurred while extracting text from PDF: {e}")
            return ''

    def preprocess_corpus(self, corpus):
        try:
            preprocessed_abstracts = []
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))

            for abstract in corpus:
                # Remove HTML tags
                abstract = re.sub(r'<.*?>', ' ', abstract)
                abstract = abstract.lower()
                # Modify the regex pattern to exclude non-alphabetic characters while preserving underscores
                abstract = re.sub(r'[^a-zA-Z\s]', ' ', abstract)
                words = word_tokenize(abstract)
                words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
                preprocessed_text = ' '.join(words)
                preprocessed_abstracts.append(preprocessed_text)

            return preprocessed_abstracts
        except Exception as e:
            print(f"Error occurred during corpus preprocessing: {e}")

    def create_vocabulary(self, corpus):
        try:
            words = [word.strip() for text in corpus for word in text.split() if not re.search('[^a-zA-Z_]', word.strip())]
            total_word_count = len(words)
            word_counts = Counter(words)
            word_probas = {word: count / total_word_count for word, count in word_counts.items()}
            return set(words), word_counts, word_probas
        except Exception as e:
            print(f"Error occurred while creating vocabulary: {e}")

    def save_preprocessed_data(self):
        directory_path = file_path
        data = {
            'vocabulary': self.vocabulary,
            'word_counts': self.word_counts,
            'word_probas': self.word_probas
        }
        with open(os.path.join(directory_path, 'preprocessed_data.pkl'), 'wb') as f:
            pickle.dump(data, f)

    def load_preprocessed_data(self):
        try:
            directory_path = file_path
            with open(os.path.join(directory_path, 'preprocessed_data.pkl'), 'rb') as f:
                data = pickle.load(f)
                self.vocabulary = data['vocabulary']
                self.word_counts = data['word_counts']
                self.word_probas = data['word_probas']
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"Error occurred while loading preprocessed data: {e}")
            return False

    def build_bigram_model(self):
        try:
            bigrams = list(ngrams(self.preprocessed_corpus, 2))
            self.bigram_model = Laplace(order=2)
            self.bigram_model.fit([bigrams], vocabulary_text=self.preprocessed_corpus)
            self.save_bigram_model()
        except Exception as e:
            print(f"Error occurred while building bigram model: {e}")

    def save_bigram_model(self):
        directory_path = file_path
        with open(os.path.join(directory_path, 'bigram_model.pkl'), 'wb') as f:
            pickle.dump(self.bigram_model, f)

    def load_bigram_model(self):
        try:
            directory_path = file_path
            with open(os.path.join(directory_path, 'bigram_model.pkl'), 'rb') as f:
                self.bigram_model = pickle.load(f)
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"Error occurred while loading bigram model: {e}")
            return False

    def correct_spelling(self, word):
        try:
            if word in self.vocabulary:
                return f"{word} is already correctly spelled"

            suggestions_edit_distance_1 = self.edit_distance_operations(word)[0]  # Suggestions with edit distance 1
            suggestions_edit_distance_2 = self.edit_distance_operations(word)[1]  # Suggestions with edit distance 2
            possible_suggestions = suggestions_edit_distance_1.union(suggestions_edit_distance_2, {word})  # Combine suggestions with edit distances 1 and 2

            best_guesses = [w for w in possible_suggestions if w in self.vocabulary]

            # Calculate weighted probabilities based on word probabilities
            corrections = [(w, self.word_probas.get(w, 0)) for w in best_guesses]

            # Group suggestions by edit distance and sort each group by probability
            grouped_corrections = {}
            for suggestion, probability in corrections:
                edit_distance = nltk.edit_distance(word, suggestion)
                if edit_distance not in grouped_corrections:
                    grouped_corrections[edit_distance] = []
                grouped_corrections[edit_distance].append((suggestion, probability))

            # Sort each group by probability
            for key in grouped_corrections:
                grouped_corrections[key] = sorted(grouped_corrections[key], key=lambda x: x[1], reverse=True)

            # Combine all groups and flatten the list
            sorted_corrections = [correction for distance in sorted(grouped_corrections) for correction in grouped_corrections[distance]]

            if sorted_corrections:
                return sorted_corrections
            else:
                return None
        except Exception as e:
            print(f"Error occurred during spelling correction: {e}")

    def edit_distance_operations(self, word):
        def split(word):
            return [(word[:i], word[i:]) for i in range(len(word))]

        def delete(word):
            return [l + r[1:] for l,r in split(word) if r]

        def swap(word):
            return [l + r[1] + r[0] + r[2:] for l, r in split(word) if len(r)>1]
        def replace(word):
            letters = string.ascii_lowercase
            return [l + c + r[1:] for l, r in split(word) if r for c in letters]
        def insert(word):
            letters = string.ascii_lowercase
            return [l + c + r for l, r in split(word) for c in letters]
        def transpose(word):
            return [l + r[1] + r[0] + r[2:] for l, r in split(word) if len(r) > 1]
        def edit1(word):
            return set(delete(word) + swap(word) + replace(word) + insert(word) + transpose(word))
        def edit2(word):
            return set(e2 for e1 in edit1(word) for e2 in edit1(e1))
        return edit1(word), edit2(word)

    def on_key_release(self, event):
        # Get the current text in the text entry field
        current_text = self.text_entry.get("1.0", tk.END)
        # Count the number of words in the current text
        word_count = len(word_tokenize(current_text))
        # Calculate the remaining word limit
        remaining_word_limit = max(self.max_word_limit - word_count, 0)
        # Update the current word limit
        self.current_word_limit = remaining_word_limit
        # Update the label to display the remaining word limit
        self.label.config(text=f"Enter text (Word Limit: {self.current_word_limit})")
        # If the word count exceeds the maximum limit, disable further input
        if word_count > self.max_word_limit:
            # Disable text entry
            self.text_entry.config(state=tk.DISABLED)
        else:
            # Enable text entry
            self.text_entry.config(state=tk.NORMAL)
        # Highlight misspelled words
        self.highlight_misspelled_words()

    def clear_input(self):
        self.text_entry.delete("1.0", tk.END)
        self.suggestion_text.delete("1.0", tk.END)
        self.misspelled_words = {}  # Clear misspelled words dictionary

    def search_word(self):
        search_word = self.search_entry.get()
        self.suggestion_text.delete("1.0", tk.END)
        if search_word in self.vocabulary:
            # Display the search word in the suggestion text box
            self.suggestion_text.insert(tk.END, f"Search Word: {search_word}\n\n")
            probability = self.word_probas.get(search_word, 0)   
            self.suggestion_text.insert(tk.END, f"{search_word} (Probability: {probability:.8f})")
        else:
            self.suggestion_text.insert(tk.END, f"Word '{search_word}' not found in the dictionary.")

    def show_word_list(self):
        self.suggestion_text.delete("1.0", tk.END)
        english_words = set(words.words())
        sorted_words = sorted(self.vocabulary.intersection(english_words))
        for word in sorted_words:
            self.suggestion_text.insert(tk.END, f"{word}\n")
    
    def show_vocab_length(self):
        vocab_length = len(self.vocabulary)
        tk.messagebox.showinfo("Vocabulary Length", f"The length of the vocabulary is: {vocab_length}")

    def show_suggestions(self, word):
        # Get the index of the clicked word
        index = self.text_entry.index(tk.CURRENT)

        # Find the word at that index
        word = self.text_entry.get(index + " wordstart", index + " wordend").strip()

        # Show suggestions for the clicked word
        self.suggestion_text.delete("1.0", tk.END)
        suggestions = self.correct_spelling(word)
        if suggestions:
            grouped_suggestions = {}  # Group suggestions by edit distance
            for suggestion, probability in suggestions:
                edit_distance = nltk.edit_distance(word, suggestion)
                if edit_distance not in grouped_suggestions:
                    grouped_suggestions[edit_distance] = []
                grouped_suggestions[edit_distance].append((suggestion, probability))
        
            # Display only top 5 suggestions for each edit distance group
            for distance in sorted(grouped_suggestions.keys()):
                self.suggestion_text.insert(tk.END, f"Edit Distance: {distance}\n")
                top_5_suggestions = sorted(grouped_suggestions[distance], key=lambda x: x[1], reverse=True)[:5]
                for rank, (suggestion, prob) in enumerate(top_5_suggestions, start=1):
                    self.suggestion_text.insert(tk.END, f"{rank}. {suggestion} (Probability: {prob:.8f})\n")
                self.suggestion_text.insert(tk.END, "\n")
        else:
            self.suggestion_text.insert(tk.END, "No suggestions found")

    def highlight_misspelled_words(self):
        self.suggestion_text.delete("1.0", tk.END)

    # Clear previous tags
        self.text_entry.tag_remove("misspelled", "1.0", tk.END)

        text = self.text_entry.get("1.0", tk.END)
        words = word_tokenize(text)
        tagged_words = nltk.pos_tag(words)  # Perform POS tagging
        for word, pos_tag in tagged_words:
        # Check if word is a stopword, punctuation, or common word, or if it's a digit
            if (word.strip() not in stopwords.words('english') and
                word not in string.punctuation and
            #word.lower() not in ['the', 'he', 'is', 'her'] and
                not word.isdigit() and
                pos_tag not in ['CC', 'DT', 'IN', 'PRP', 'PRP$', 'TO', 'WRB']):
                if not re.search(r'[^a-zA-Z0-9_]', word.strip()):  # Check if word contains only alphanumeric characters or underscores
                # Check if the word is in the vocabulary or its lowercase version is in the vocabulary
                    if word not in self.vocabulary and word.lower() not in self.vocabulary and not self.is_plural(word):
                        self.highlight_word(word.strip())

                    # Bind suggestion event to misspelled words
                        self.text_entry.tag_bind("misspelled", "<Button-1>", lambda event, word=word.strip(): self.show_suggestions(word))

    def highlight_word(self, word):
        start_index = "1.0"
        while True:
            start_index = self.text_entry.search(word, start_index, tk.END)
            if not start_index:
                break
            end_index = f"{start_index}+{len(word)}c"
            self.text_entry.tag_add("misspelled", start_index, end_index)
            start_index = end_index

        self.text_entry.tag_config("misspelled", underline=True, foreground="red")

    def is_plural(self, word):
        return word != word.lower() and word.lower() in self.vocabulary

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, "r") as file:
                    content = file.read()
                    self.text_entry.insert(tk.END, content)
                    self.status_label.config(text=f"File opened: {file_path}")
            except Exception as e:
                self.status_label.config(text=f"Error opening file: {e}")

    def save_as_file(self):
        content = self.text_entry.get("1.0", tk.END)
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, "w") as file:
                    file.write(content)
                    self.status_label.config(text=f"File saved as: {file_path}")
            except Exception as e:
                self.status_label.config(text=f"Error saving file: {e}")
    
    def clear_input(self):
        self.text_entry.delete("1.0", tk.END)
        self.suggestion_text.delete("1.0", tk.END)
        self.misspelled_words = {}  # Clear misspelled words dictionary
    
    # Reset word limit to maximum
        self.current_word_limit = self.max_word_limit
    # Update the label to display the remaining word limit
        self.label.config(text=f"Enter text (Word Limit: {self.current_word_limit})")

root = tk.Tk()
app = SpellingCheckerApp(root)
root.mainloop()
