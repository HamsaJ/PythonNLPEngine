import spacy
from spacy.lang.en import English
from .pipelines import RESTCountriesComponent

ner_model = spacy.load("en_core_web_sm")

nlp = English()
rest_countries = RESTCountriesComponent(nlp, label="COUNTRY")
nlp.add_pipe(rest_countries)
