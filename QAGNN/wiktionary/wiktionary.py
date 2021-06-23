import os
import urllib.request
import json
import spacy

class Wiktionary:

    def __init__(self):

        json_path = "wiktionary/wiktionary_en.json"

        if not os.path.exists(json_path):
            database_url = "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.json"
            print("Downloading wiktionary_en")
            urllib.request.urlretrieve(database_url, json_path)
            print("Downloading wiktionary_en completed")

        self.data = {}

        print("Indexing wiktionary")
        for line in open(json_path, 'r'):
            json_data = json.loads(line)
            word = json_data['word']

            if word in self.data.keys():
                continue # Use the first definition per concept

            senses = json_data[word] = json_data['senses']              

            if 'glosses' in senses[0]:
                self.data[word] = senses[0] # store str of 1st sense in dict

                if not word.lower() in self.data.keys():
                    self.data[word.lower()] = senses[0] # store lower case version as backup


        print(f"Indexing wiktionary completed. Found {len(self.data)} concepts.")

        self.lemmatizer = spacy.load("en")

    def find_entry(self, query):

        # if the query directly appears, return the entry
        value = self.data.get(query)
        if value != None:
            return value

        # iteratively replace the tokens in the query with lemmatized versions
        # if at any point the new query appears, return the entry
        tokens = self.lemmatizer(query)
        for num_tokens_to_lemmatize in range(1, len(tokens)+1):
            new_query = ""
            for i in range(num_tokens_to_lemmatize):
                new_query += " " + tokens[i].lemma_ # lemmatized token
            for i in range(num_tokens_to_lemmatize, len(tokens)):
                new_query += " " + str(tokens[i]) # original token
            new_query = new_query[1:] # drop white space at position 0

            value = self.data.get(new_query)
            if value != None:
                return value

        # iteratively leave out 1 more token of the lematized query, starting with the front token
        # if at any point the new query appears, return the entry
        for num_tokens_to_leave_out in range(1, len(tokens)):
            new_query = ""
            for i in range(num_tokens_to_leave_out, len(tokens)):
                new_query += " " +tokens[i].lemma_
            new_query = new_query[1:] # drop white space at position 0

            value = self.data.get(new_query)
            if value != None:
                return value

        if not query.isLower():
            return find_entry(query.lower())
        
        raise KeyError

    def __getitem__(self, query):
        """Returns the wiktionary entry for a query.

        For every concept, we choose its first definition entry in Wiktionary as the description.
        For every question/choice concept, we find its closest match in Wiktionary by using 
        the following forms in order: 
            i) original form;
            ii) lemmaform by Spacy (Honnibal and Montani, 2017);
            iii) base word (last word). 
        """

        entry = self.find_entry(query)

        while 'form_of' in entry.keys():
            entry = self.data[entry['form_of'][0]]

        return entry['glosses'][0]
