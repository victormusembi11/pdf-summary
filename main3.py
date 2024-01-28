import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import PyPDF2

# TODO: add argparse to allow user to specify file path


stopwords = list(STOP_WORDS)

nlp = spacy.load("en_core_web_sm")


def convert_pdf_to_text(file_path):
    pdf_file_obj = open(file_path, "rb")
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page_obj = pdf_reader.pages[page_num]
        text += page_obj.extract_text()
    pdf_file_obj.close()
    return text


text = convert_pdf_to_text("pdfs/bitcoin.pdf")

doc = nlp(text)

tokens = [token.text for token in doc]
# print(tokens)

punctuation = punctuation + "\n"
# print(punctuation)


def get_word_frequencies():
    word_frequencies = {}

    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    return word_frequencies


word_frequencies = get_word_frequencies()

max_frequency = max(word_frequencies.values())
# print(max_frequency)

for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word] / max_frequency

# print(word_frequencies)

sentence_tokens = [sent for sent in doc.sents]
# print(sentence_tokens)


def calculate_sentence_scores():
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    return sentence_scores


sentence_scores = calculate_sentence_scores()
# print(sentence_scores)

select_length = int(len(sentence_tokens) * 0.3)
# print(select_length)

summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
# print(summary)

final_summary = [word.text for word in summary]
summary = " ".join(final_summary)
# print(summary)

# print(text)

print("\n\n\n")

print(summary)
