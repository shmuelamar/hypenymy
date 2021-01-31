import json
from functools import lru_cache

from nltk.corpus import wordnet as wn
import inflect
import spacy
import wikidata_queries as wiki
from country_to_adjective import country2adj

excluded_classes = ['object', 'thing', 'attribute', 'abstraction', 'entity', 'object', 'whole', 'artifact', 'physical_entity', 'substance', 'psychological_feature', 'matter', 'cognitive_state', 'causal_agent', 'living_thing', 'event', 'act',
                    'being', 'state']
singular_errors = []


COUNTRY_LIST_FILENAME = 'country_list.json'
LOCAL_WIKI_FILENAME = 'local_location_wiki.json'
LOCAL_WIKI_FEATURES_FILENAME = 'local_features_wiki.json'


print("** Loading inflect.engine()...")
p = inflect.engine()

print("**Loading Spacy...")
spacy_nlp_parser = spacy.load('en_core_web_lg')


@lru_cache(maxsize=int(2**20))
def nlp(s):
    return spacy_nlp_parser(s)
print("Done loading Spacy")


def save_file(data, fname):
    with open(fname, 'w') as fp:
        json.dump(data, fp, indent=4, sort_keys=True)


def load_file(fname):
    with open(fname, 'r') as fp:
        return json.load(fp)


country_list = load_file(COUNTRY_LIST_FILENAME)
local_wiki = load_file(LOCAL_WIKI_FILENAME)
local_wiki_features = load_file(LOCAL_WIKI_FEATURES_FILENAME)


### general utils
def word_peice_connected(tokens, input_ids):
    # Input: tokens of wordpiece tokenizer output and their id's
    # Output: the connected token list + new2old_ind converter
    # For example:
    # word_to_find = 'broccoli'
    # tokens =  "[CLS] '  mom  ##will send you gifts   bro  ##cco ##li  and  will ##send  you  bro ##cco ##li fruits for  the  return journey , ' shouted the quartermaster . [SEP] ' mom will send you gifts and bro ##cco ##li for the return journey , ' shouted the quartermaster . [SEP] ".split()
    # input_ids =[101, 18, 202, 313,   414, 515, 616,   717,  838,  949, 111, 222,  333,   444, 555, 666,  18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, 101, 18, 8231,  313, ]
    # old_ind =  [0,   1,   2,   3,    4,   5,   6,      7,   8,    9,   10,  11,   12,    13,  14,  15,   16,  17     18  ]
    # new_ind=   [0,   1,   2,         3,   4,   5,      6,               7,   8,           9,  10,             11,    12,    13]

    old_ind = list(range(len(input_ids)))
    new_ind = []
    new2old_ind = []
    s_new = []
    for i in old_ind:
        if tokens[i][:2] != '##':
            new_ind.append(input_ids[i])
            new2old_ind.append(i)
            s_new.append(tokens[i])
        else:
            wt = s_new.pop()
            wt += tokens[i][2:]
            s_new.append(wt)
    return s_new, new2old_ind


def is_plural(word):
    force_singular = ['process', 'bedclothes', 'address', 'compass', 'class', 'business', 'religious', 'judiciousness', 'attentiveness', 'access', 'cross',
                      'abbess', 'relatedness', 'illness', 'sameness', 'resoluteness', 'gas', 'kindness', 'orderliness', 'trustworthiness', 'activeness', 'inattentiveness',
                      'carelessness', 'loss']
    if not word.isalpha(): return word
    if word in force_singular:
        return False
    return not p.singular_noun(word) == False

def singular_noun_fixed(word):
    fixes = {'proces':'process', 'bedclothe':'bedclothes', 'due_proces': 'due_process', 'addres':'address', 'compas':'compass', 'clas':'class', 'busines':'business', '':''
        , 'religiou': 'religious', 'judiciousnes':'judiciousness', 'attentivenes':'attentiveness', 'acces':'access', 'cros':'cross'}
    if not word.isalpha(): return word
    sing = p.singular_noun(word)
    if sing in fixes:
        return fixes[sing]
    return sing

def to_singular(word_list):
    if type(word_list)==list:
        return [singular_noun_fixed(w) if (is_plural(w) and w.isalpha()) else w for w in word_list]
    else:
        return singular_noun_fixed(word_list) if ( word_list.isalpha() and is_plural(word_list)) else word_list


### hypernymy
def pick_nouns_for_hypernyms(sentence, mode='noun'):
    doc = nlp(sentence)
    entities = [a.text for a in doc.ents]   # entities from SpiCy NER
    nouns = []
    pos_list = dict(include_propn=['NOUN', 'PROPN'], noun=['NOUN'])
    if True:
        for d in doc:
            cand_is_upper = (d.text != sentence[0:len(d.text)] and d.text[0].isupper())
            if d.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not cand_is_upper:
                nouns.append(d.text.lower())
    else:
        for chunk in doc.noun_chunks:   # looking for NP chunks
            cand = chunk.root
            cand_is_upper = (cand.text != sentence[0:len(cand.text)] and cand.text[0].isupper())
            loc_after_chunk = (' ' + sentence + ' ').find(chunk.root.text) +len(chunk.root.text) -1
            loc_after_chunk = loc_after_chunk-1 if loc_after_chunk >= len(sentence) else loc_after_chunk    # In case the chunk ends where the sentence ends (so no character after it)
            has_dash = (sentence[(' ' + sentence + ' ').find(chunk.root.text) -2] == '-') or (sentence[loc_after_chunk] == '-')
            if cand.text not in entities and not cand_is_upper and cand.pos_ in pos_list[mode] and not has_dash:
                nouns.append(chunk.root.text.lower())
    return list(set(to_singular(nouns))), doc


def all_hypernyms2(word, noun_only=False):
    results = []
    if wn.synsets(word):
        for sense in wn.synsets(word):
            if not noun_only or sense.pos()=='n':
                for path in sense.hypernym_paths():
                    results += path[:-1]
        return list(set(results))
    return []


def synset_name(synset):
    if synset is None:
        return ''
    return synset.name().split('.')[0]


def list_hypernyms2(word_list, is_print=False, noun_only=False):
    global excluded_classes
    output = []
    for w in word_list:
        h = [hyper for hyper in all_hypernyms2(w, noun_only=noun_only) if synset_name(hyper) not in excluded_classes and synset_name(hyper).find('_')<0]
        if h:
            output.append((w, h))
            if is_print:
                print('*** ' + w + ':')
                for hi in h:
                    print(hi)
                print('\n')
    return output


def get_hypernym_candidates2(sentence_string, is_print=True):
    # look for the last word in each NP and returns all of its hypernyms, for all senses
    nouns, doc = pick_nouns_for_hypernyms(sentence_string)
    words_hypernyms = list_hypernyms2(nouns, is_print=is_print, noun_only=True)
    return words_hypernyms, doc, nouns


def find_all_ind(val, val_list):
    return [i for i, x in enumerate(val_list) if x.lower() == val.lower()]


def find_hypernymy_pairs(s1, s2, tokens1, tokens2, is_print=False, search_only_nouns=True, filter_repeat_word=False):
    # find hyponym-hypernym pairs in this order ONLY, so one is in s1 and the other in s2.
    # Input:
    #   s1 and s2, two sentences to locate hyponym in one and hypernyum at the other.
    # Output:
    #   pairs: a list of the founded pairs. each element is a tuple:   (hypo, hyper_s_orig, "hypo->hyper", (hypo_ind, hyper_s_ind))
    #       - hypo string (e.g. 'dog'), converted to singular
    #       - hyper string (converted to singular)
    #       - direction: "hypo->hyper" for hypo in s1 and hyper in s2
    #       - hypo_ind: all indices of hypo in the sentence where it appears. Note! indices are w.r.t the second sentence if direction is "hyper->hypo"
    #       - hyper_s_ind: all indices of hyper in the sentence where it appears in its original form

    global singular_errors
    global excluded_classes

    both_in_p_and_h = 0

    pairs_p_h, pairs_h_p = {}, {}
    duplicates = ""
    not_printed = True
    words_hypernyms_s1, doc1, nouns1 = get_hypernym_candidates2(s1, is_print=False) # look for the last word in each NP and returns all of its hypernyms, for all senses
    words_hypernyms_s2, doc2, nouns2 = get_hypernym_candidates2(s2, is_print=False) #
    tokesn2_only_nouns = nouns2 if search_only_nouns else tokens2
    tokesn1_only_nouns = nouns1 if search_only_nouns else tokens1

    def update_d(d, values):
        hypo, hyper, direction, hypo_ind, hyper_ind = values
        if hyper not in d:
            d[hyper] = {'hypo':[hypo] * len(hypo_ind), 'direction':direction, 'hypo_ind':hypo_ind.copy(), 'hyper_ind':hyper_ind.copy()}
        elif hypo not in d[hyper]['hypo']:      # we only want each hypo once for each hyper
            d[hyper]['hypo'] += [hypo] * len(hypo_ind)
            d[hyper]['hypo_ind'] += hypo_ind
        assert len(d[hyper]['hypo'])==len(d[hyper]['hypo_ind']), f"d[hyper]['hypo']={d[hyper]['hypo']} but d[hyper]['hypo_ind']={d[hyper]['hypo_ind']}"
        return d


    # looking for hypo in s1 and hyper in s2
    for (hypo, hs) in words_hypernyms_s1:       # words_hypernyms_s1 = [('dog', ['Synset(animal.n.4)', 'Synset(mammal.n.1)'...]), ]
        hypo_ind = find_all_ind(hypo, to_singular(tokens1))
        # print('*** ' + hypo + ':')
        for hi in hs:
            hyper_orig = synset_name(hi)
            hyper_s = to_singular(hyper_orig)
            # # assert hyper_s == to_singular(hyper_s) , f"hypers='{hyper_s}' should be singular, but is seems to be plural"
            # ### TODO: bring back the assert above instead of the 'if' below
            # if hyper_s != to_singular(hyper_s) and hyper_s not in singular_errors:
            #     print(hyper_s)
            #     warnings.warn(f'hypernymy {hyper_s} (of {hypo}) seems to be plural')

            # print(hyper_s + ' -> ')
            # if s2.find(hyper_s) > -1 and s1.find9(hyper_s) < 0 and s2.find(hypo) < 0 and hyper_s not in excluded_classes:
            do_filter = filter_repeat_word and ((hyper_s in to_singular(tokesn1_only_nouns)) or (hypo in to_singular(tokesn2_only_nouns)))
            if hyper_s in to_singular(tokesn2_only_nouns) and not do_filter:   # looking for the hyper in s2
                hyper_s_ind = find_all_ind(hyper_s, to_singular(tokens2))
                if len(hyper_s_ind) > 0:
                    duplicates += "hyper appears twice in S2. "
                # hyper_s_orig = tokens2[hyper_s_ind[0]]
                # pairs_p_h.append((hypo, hyper_s, "hypo->hyper", hypo_ind, hyper_s_ind))
                pairs_p_h = update_d(pairs_p_h, (hypo, hyper_s, "hypo->hyper", hypo_ind, hyper_s_ind))
                if is_print:
                    if not_printed:
                        print('\n' + s1)
                        print(s2)
                        not_printed = False
                    else:
                        print("-----------Multiple----------")
                    print('*** Found in H:%s -> %s' % (hypo, hyper_s))
    return pairs_p_h #, (doc1, doc2)


### location
def location2country_or_state(local_wiki, location):
    if location.lower() not in local_wiki:
        countries = wiki.get_wikidata_us_state_and_country(location)
        local_wiki[location] = [c.lower() for c in countries]    # it's a list for backward compatability
    return local_wiki[location]


def get_location_candidates(sentence, include_ORG=True):
    locations = []
    labels_to_include = ['LOC', 'GPE', 'ORG'] if include_ORG else ['LOC', 'GPE']
    doc = nlp(sentence)
    for e in doc.ents:
        if e.label_ in labels_to_include and e.text.lower() not in country_list and e.text.lower().find('.')==-1 and e.text.isalpha():      # not '.' in the location & if can't be a country name
            locations.append(e.text.lower())
    return locations


def find_all_country_ind(country, tokens):
    # example: country_orig='united states of america' ; tokens = ['i', 'used', 'to', 'live', 'in', 'the', 'u', '.', 's', '.', 'a', 'when', 'i', 'was', 'young']
    synonyms = {'united states of america': ['united states of america', 'the united states of america', 'the united states', 'united states', 'us', 'usa', 'america', 'the us', 'the usa', 'u . s .', 'u . s', 'the u . s .', 'the u . s . a .',
                                              'the u . s . a', 'u . s . a', 'u . s . a .'], 'united kingdom':['kingdom of england', 'england', 'the united kingdom', 'united kingdom', 'uk', 'u . k .', 'the u . k .', 'great britain', 'britain']}
    indices = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if country in synonyms:
            for c_syn in synonyms[country]:
                equal = True
                for j, ct in enumerate(c_syn.split()):
                    if i + j >= len(tokens) or tokens[i + j] != ct:
                        equal = False
                        break
                if equal:
                    indices.append(i)
                    i += len(c_syn.split())
                    break
        else:
            if country == t:
                indices.append(i)
        i += 1
    return indices


def find_location_country_pairs(s1, s2, tokens1, tokens2, local_wiki, is_print=False,  filter_repeat_word=False, include_ORG=True):
    # find location-country pairs, so one is in s1 and the other in s2
    # Input:
    #   s1 and s2, two sentences to find a location in s1 and a country in s2
    #   filter_repeat_word - if the location or the country appear both in s1 and in s2, don't count that pair (since it's probably not a location-country phenomenon)
    # Output:
    #   pairs: a list of the founded pairs. each element is a tuple:   (location, country, loc_ind, country_ind)
    #       - hypo string (e.g. 'dog'), converted to singular
    #       - hyper string (converted to singular)
    #       - direction: "hypo->hyper" for hypo in s1 and hyper in s2
    #       - hypo_ind: all indices of hypo in the sentence where it appears. Note! indices are w.r.t the second sentence if direction is "hyper->hypo"
    #       - hyper_s_ind: all indices of hyper in the sentence where it appears in its original form
    global singular_errors
    global excluded_classes

    pairs = {}
    duplicates = ""
    not_printed = True
    locations_s1 = get_location_candidates(s1, include_ORG) # returns all entities (using Spacy) that are LOC or GPE in lower()

    def update_d(d, values):
        location, country, location_ind, country_ind = values
        if country not in d:
            d[country] = {'location':[location] * len(location_ind), 'location_ind':location_ind.copy(), 'country_ind':country_ind.copy()}
        elif location not in d[country]['location']:      # we only want each location once for each country
            d[country]['location'] += [location] * len(location_ind)
            d[country]['location_ind'] += location_ind
        assert len(d[country]['location'])==len(d[country]['location_ind']), f"d[country]['location']={d[country]['location']} but d[country]['location_ind']={d[country]['location_ind']}"
        return d

    # looking for location in s1 and a matching country in s2
    for location in locations_s1:       # locations_s1 = ['kamakura', 'jerusalem']
        # countries = location2country(local_wiki, location)
        countries = location2country_or_state(local_wiki, location)
        location_ind = find_all_ind(location, tokens1)

        if location_ind == []: continue     # in case of location with multiple words, skip this location.
        for country in countries:
            do_filter = filter_repeat_word  and ((country in tokens1) or (location in tokens2))
            country_ind = find_all_country_ind(country, tokens2)
            if country_ind != [] and not do_filter:
                if len(country_ind) > 1:
                    duplicates += f"country {country} appears twice in S2. "
                pairs = update_d(pairs, (location, country, location_ind, country_ind))
                if is_print:
                    if not_printed:
                        print('\n' + s1)
                        print(s2)
                        not_printed = False
                    else:
                        print("-----------Multiple----------")
                    print('*** Found in H:%s -> %s' % (location, country))
    save_file(local_wiki, LOCAL_WIKI_FILENAME)
    return pairs


### color
def pick_nouns_for_color(sentence):
    doc = nlp(sentence)
    nouns = []
    for chunk in doc.noun_chunks:   # looking for NP chunks
        cand = chunk.root
        loc_after_chunk = (' ' + sentence + ' ').find(chunk.root.text) +len(chunk.root.text) -1
        loc_after_chunk = loc_after_chunk-1 if loc_after_chunk >= len(sentence) else loc_after_chunk    # In case the chunk ends where the sentence ends (so no character after it)
        has_dash = (sentence[(' ' + sentence + ' ').find(chunk.root.text) -2] == '-') or (sentence[loc_after_chunk] == '-')
        if cand.pos_ in ['NOUN', 'PROPN'] and not has_dash:
            nouns.append(chunk.root.text)
    return list(set(to_singular(nouns))), doc


def noun2features(local_wiki_features, noun):
    noun = to_singular(noun.lower())
    # if to_singular(noun) not in local_wiki_features:
    if noun not in local_wiki_features:
        features = wiki.get_wikidata_features(noun)
        local_wiki_features[noun] = features    # it's a list for backward compatability
    assert noun in local_wiki_features, f"the noun '{noun}' doesn't exist in the dictionary local_wiki_features with len {len(local_wiki_features)}"
    return local_wiki_features[noun]


def features_synonyms(feature, feature_type):
    # converting feature name to its synonyms as appear in Wikidata.
    #           E.g. 'sphere' -> 'round', ...
    d = {'color': {}, 'shape': {'round':'sphere'}, 'material': {}, }
    if feature in d[feature_type]:
        return d[feature_type][feature]
    return feature


def find_all_ind2(target, tokens):
    """
    finds all appearances of a target phrase (could be multiple words, e.g. 'dark brown') within a list of tokens.
    :param target: a String of the feature/country. E.g. 'dark brown', 'Italian wine', etc.
    :param tokens: the List of tokens of the sentence to be looked at
    :return: all indices where a match was found
    """

    featur_tokens = target.split()

    indices = []
    i = 0
    while i < len(tokens):
        equal = True
        for j, ft in enumerate(featur_tokens):
            if i + j >= len(tokens) or tokens[i + j] != ft:
                equal = False
                break
        if equal:
            indices.append(i)
            i += len(featur_tokens)
        i += 1
    return indices


def find_color_pairs(s1, s2, tokens1, tokens2, local_wiki_features, is_print=False,  filter_repeat_word=False):
    # find noun-country pairs, so one is in s1 and the other in s2
    # Input:
    #   s1 and s2, two sentences to find a noun in s1 and a country in s2
    #   filter_repeat_word - if the noun or the country appears both in s1 and in s2, don't count that pair (since it's probably not a noun-country phenomenon)
    # Output:
    #   pairs: a list of the founded pairs. each element is a tuple:   (noun, country, loc_ind, country_ind)
    #       - hypo string (e.g. 'dog'), converted to singular
    #       - hyper string (converted to singular)
    #       - direction: "hypo->hyper" for hypo in s1 and hyper in s2
    #       - hypo_ind: all indices of hypo in the sentence where it appears. Note! indices are w.r.t the second sentence if direction is "hyper->hypo"
    #       - hyper_s_ind: all indices of hyper in the sentence where it appears in its original form
    global singular_errors
    global excluded_classes

    duplicates = ""
    not_printed = True
    get_hypernym_candidates2
    nouns, doc = pick_nouns_for_color(s1)   # Same principle as in hypernyms - picks the last noun in an NP constituency.
    pairs = {}
    # locations_s1  = get_location_candidates(s1, include_ORG) # returns all entities (using Spacy) that are LOC or GPE in lower()
    feature_types = ('color', 'shape', 'material')

    def update_d(d, values):
        noun, feature, noun_ind, feature_ind, feature_type = values
        if feature not in d:
            d[feature] = {'noun':[noun] * len(noun_ind), 'noun_ind':noun_ind.copy(), 'feature_ind':feature_ind.copy(), 'type': feature_type}
        elif noun not in d[feature]['noun']:      # we only want each noun once for each feature
            d[feature]['noun'] += [noun] * len(noun_ind)
            d[feature]['noun_ind'] += noun_ind
        assert len(d[feature]['noun'])==len(d[feature]['noun_ind']), f"d[feature]['noun']={d[feature]['noun']} but d[feature]['noun_ind']={d[feature]['noun_ind']}"
        return d

    ##TODO: I think that there is a bug with location of the period at end of sentence ==> 'LastWordToken.' instead of 'LastWordToken'
    # looking for noun in s1 and a matching feature in s2
    for noun in nouns:       # nouns = ['basketball', 'table']
        features_d = noun2features(local_wiki_features, noun)
        noun_ind = find_all_ind(noun, to_singular(tokens1))

        if noun_ind == []: continue     # in case of noun with multiple words, skip this noun.
        for feature_type in features_d:      # features_d = ((colors...), (shapes...), (material used...))
            if feature_type == 'color' and len(features_d['color']) > 1: continue      # ignore objects with more than one color
            for feature in features_d[feature_type]:
                feature = features_synonyms(feature, feature_type)      # converts to the synonym as appears in wikidata, if exists. e.g round -> sphere
                do_filter = filter_repeat_word and ((feature in tokens1) or (noun in to_singular(tokens2)))
                feature_ind = find_all_ind2(feature, tokens2)       # looks for the actual words of the feature in the second sentence (e.g. lookse for "dark brown" in s2)
                if feature_ind != [] and not do_filter:
                    if len(feature_ind) > 1:
                        duplicates += f"feature {feature} appears twice in S2. "
                    pairs = update_d(pairs, (noun, feature, noun_ind, feature_ind, feature_type))
                    if is_print:
                        if not_printed:
                            print('\n' + s1)
                            print(s2)
                            not_printed = False
                        else:
                            print("-----------Multiple----------")
                        print('*** Found in H:%s -> %s' % (noun, feature))
    save_file(local_wiki_features, LOCAL_WIKI_FEATURES_FILENAME)
    return pairs


### trademarks
def get_capitalized_candidates(sentence):
    locations = []
    doc = nlp(sentence)
    for d in doc:
        if d.pos_ in ['PROPN', 'NOUN', 'X', 'ADJ'] and any([c.isupper() for c in d.text]):
            locations.append(d.text.lower())
    return locations


def get_wikidata_country_wrap(location):
    keep_trying = True
    tic = wiki.Tic()
    time = 0

    while keep_trying:
        try:
            countries = wiki.get_wikidata_country(location)
            keep_trying = False
        except:
            time = tic.toc(False)
            print(f'failed. Time = {time}')
            keep_trying = time < 15
    return countries


def location2country(local_wiki, location):
    if location.lower() not in local_wiki:
        countries = get_wikidata_country_wrap(location)
        local_wiki[location] = [c.lower() for c in countries]
    return local_wiki[location]


def find_trademark_country_pairs(s1, s2, tokens1, tokens2, local_wiki, is_print=False,  filter_repeat_word=False, include_ORG=True):
    # find company-country pairs, so one is in s1 and the other in s2
    # Input:
    #   s1 and s2, two sentences to find a company in s1 and a country in s2
    #   filter_repeat_word - if the company or the country appear both in s1 and in s2, don't count that pair (since it's probably not a company-country phenomenon)
    # Output:
    #   pairs: a list of the founded pairs. each element is a tuple:   (company, country, loc_ind, country_ind)
    #       - hypo string (e.g. 'dog'), converted to singular
    #       - hyper string (converted to singular)
    #       - direction: "hypo->hyper" for hypo in s1 and hyper in s2
    #       - hypo_ind: all indices of hypo in the sentence where it appears. Note! indices are w.r.t the second sentence if direction is "hyper->hypo"
    #       - hyper_s_ind: all indices of hyper in the sentence where it appears in its original form
    global singular_errors
    global excluded_classes

    pairs = {}
    duplicates = ""
    not_printed = True
    companies_s1 = get_capitalized_candidates(s1)  # returns all potential entities in lower(), by taking all words with capitalized first letter

    def update_d(d, values):
        company, country, company_ind, country_ind = values
        if country not in d:
            d[country] = {'company':[company] * len(company_ind), 'company_ind':company_ind.copy(), 'country_ind':country_ind.copy()}
        elif company not in d[country]['company']:      # we only want each company once for each country
            d[country]['company'] += [company] * len(company_ind)
            d[country]['company_ind'] += company_ind
        assert len(d[country]['company'])==len(d[country]['company_ind']), f"d[country]['company']={d[country]['company']} but d[country]['company_ind']={d[country]['company_ind']}"
        return d

    def capi(word):
        return word[0].upper() + word[1:]

    def country2adj_func(country):
        if country in country2adj:
            return country2adj[country]
        else:
            return country

    # looking for company in s1 and a matching country in s2
    for company in companies_s1:       # companies_s1 = ['anobit', 'Israel']
        # countries = location2country(local_wiki, company)
        countries = location2country(local_wiki, company)
        company_ind = find_all_ind(company, tokens1)

        if company_ind == []: continue     # in case of company with multiple words, skip this company.
        for country in countries:
            do_filter = filter_repeat_word and ((country in tokens1) or (company in tokens2))
            country_ind = find_all_country_ind(country, tokens2)
            country_ind += find_all_country_ind(country2adj_func(capi(country)).lower(), tokens2)
            if country_ind != [] and not do_filter:
                if len(country_ind) > 1:
                    duplicates += f"country {country} appears twice in S2. "
                pairs = update_d(pairs, (company, country, company_ind, country_ind))
                if is_print:
                    if not_printed:
                        print('\n' + s1)
                        print(s2)
                        not_printed = False
                    else:
                        print("-----------Multiple----------")
                    print('*** Found in H:%s -> %s' % (company, country))
    save_file(local_wiki, LOCAL_WIKI_FILENAME)
    return pairs


def main(debug):
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if debug == 'hypernymy':
        a = "the frog and the cats playing outside in the sand on chair"
        b = "i see other animal inside with a furniture"
        tokens_a = tokenizer.tokenize(a)
        tokens_b = tokenizer.tokenize(b)
        tokens_a_new, tokens_b_new = \
        word_peice_connected(tokens_a, [1] * len(tokens_a))[0], \
        word_peice_connected(tokens_b, [1] * len(tokens_b))[0]  # connecting the word_peices together
        pairs = find_hypernymy_pairs(a, b, tokens_a_new, tokens_b_new, False, filter_repeat_word=True)

    elif debug == 'location-country':
        # a = "if you're coming from Brittany, you should check your GPS coordinates."
        # b = "I recommended you to check your GPS coordinates because of what city in America you were coming from."
        ## can't located Porto with Spact NER:
        # a = "although Porto has its share of pubs, bars, discos, and even a well attended casino with revues, most visitors don't come to the island for evening entertainment."
        # b = "Portugal has many pubs, bars, and discos."
        # a = "Although Trussville has its share of pubs, bars, discos, and even a well attended casino with revues, most visitors don't come to the island for evening entertainment."
        # b = "Alabama has many pubs, bars, and discos."
        a = "Although Trussville has its share of pubs, bars, discos, and even a well attended casino with revues, most visitors don't come to the island for evening entertainment."
        b = "Alabama has many pubs, bars, and discos."
        tokens_a = tokenizer.tokenize(a)
        tokens_b = tokenizer.tokenize(b)
        tokens_a_new, tokens_b_new = \
        word_peice_connected(tokens_a, [1] * len(tokens_a))[0], \
        word_peice_connected(tokens_b, [1] * len(tokens_b))[0]  # connecting the word_peices together
        pairs = find_location_country_pairs(a, b, tokens_a_new, tokens_b_new, local_wiki, include_ORG=True)

    elif debug == 'features':  # COLORS
        a = "and um they have little bar there so we sit there and um sipping on some appletinis before we get to eat"
        b = "I sipped on a bright green cocktail"
        tokens_a = tokenizer.tokenize(a)
        tokens_b = tokenizer.tokenize(b)
        tokens_a_new, tokens_b_new = \
        word_peice_connected(tokens_a, [1] * len(tokens_a))[0], \
        word_peice_connected(tokens_b, [1] * len(tokens_b))[0]  # connecting the word_peices together
        pairs = find_color_pairs(a, b, tokens_a_new, tokens_b_new, local_wiki_features, is_print=True)

    elif debug == 'trademark-country':
        a = "Cotral leather goods remain unequalled ."
        b = "There is an Italian company that produces leather goods."
        tokens_a = tokenizer.tokenize(a)
        tokens_b = tokenizer.tokenize(b)
        tokens_a_new, tokens_b_new = \
        word_peice_connected(tokens_a, [1] * len(tokens_a))[0],\
        word_peice_connected(tokens_b, [1] * len(tokens_b))[0]  # connecting the word_peices together
        pairs = find_trademark_country_pairs(a, b, tokens_a_new, tokens_b_new, local_wiki)
