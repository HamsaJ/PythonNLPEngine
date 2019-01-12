from spacy.tokenizer import Tokenizer

def num_of_tokens(nlp, text):
    tokenizer = Tokenizer(nlp.vocab)
    return len(tokenizer(text))

def num_of_sentences(nlp):
    sentence_list = []
    for sent in nlp.sents:
        sentence_list.append(sent)
    return sentence_list
