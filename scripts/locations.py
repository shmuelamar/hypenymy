import csv
import json
import os
import random
import re
import time
from collections import Counter

import cbox
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

import datautils

COUNTRIES_FNAME = os.path.join(datautils.SRCDIR, 'countries.json')

KNOWN_LOCATIONS = {
    c.lower()
    for cs in datautils.json_load(COUNTRIES_FNAME).values()
    for c in cs
} | set(datautils.json_load(COUNTRIES_FNAME))

COUNTRIES = [
    "Afghanistan",
    "Albania",
    "Algeria",
    "Andorra",
    "Angola",
    "Antigua and Barbuda",
    "Argentina",
    "Armenia",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahamas",
    "Bahrain",
    "Bangladesh",
    "Barbados",
    "Belarus",
    "Belgium",
    "Belize",
    "Benin",
    "Bhutan",
    "Bolivia",
    "Bosnia and Herzegovina",
    "Botswana",
    "Brazil",
    "Brunei",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    "Côte d'Ivoire",
    "Cabo Verde",
    "Cambodia",
    "Cameroon",
    "Canada",
    "Central African Republic",
    "Chad",
    "Chile",
    "China",
    "Colombia",
    "Comoros",
    "Congo (Congo-Brazzaville)",
    "Costa Rica",
    "Croatia",
    "Cuba",
    "Cyprus",
    "Czechia (Czech Republic)",
    "Democratic Republic of the Congo",
    "Denmark",
    "Djibouti",
    "Dominica",
    "Dominican Republic",
    "Ecuador",
    "Egypt",
    "El Salvador",
    "Equatorial Guinea",
    "Eritrea",
    "Estonia",
    "Eswatini ",
    "Ethiopia",
    "Fiji",
    "Finland",
    "France",
    "Gabon",
    "Gambia",
    "Georgia",
    "Germany",
    "Ghana",
    "Greece",
    "Grenada",
    "Guatemala",
    "Guinea",
    "Guinea-Bissau",
    "Guyana",
    "Haiti",
    "Holy See",
    "Honduras",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Jamaica",
    "Japan",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kiribati",
    "Kuwait",
    "Kyrgyzstan",
    "Laos",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Liberia",
    "Libya",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Maldives",
    "Mali",
    "Malta",
    "Marshall Islands",
    "Mauritania",
    "Mauritius",
    "Mexico",
    "Micronesia",
    "Moldova",
    "Monaco",
    "Mongolia",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Myanmar (formerly Burma)",
    "Namibia",
    "Nauru",
    "Nepal",
    "Netherlands",
    "New Zealand",
    "Nicaragua",
    "Niger",
    "Nigeria",
    "North Korea",
    "North Macedonia",
    "Norway",
    "Oman",
    "Pakistan",
    "Palau",
    "Palestine State",
    "Panama",
    "Papua New Guinea",
    "Paraguay",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "Russia",
    "Rwanda",
    "Saint Kitts and Nevis",
    "Saint Lucia",
    "Saint Vincent and the Grenadines",
    "Samoa",
    "San Marino",
    "Sao Tome and Principe",
    "Saudi Arabia",
    "Senegal",
    "Serbia",
    "Seychelles",
    "Sierra Leone",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "Solomon Islands",
    "Somalia",
    "South Africa",
    "South Korea",
    "South Sudan",
    "Spain",
    "Sri Lanka",
    "Sudan",
    "Suriname",
    "Sweden",
    "Switzerland",
    "Syria",
    "Tajikistan",
    "Tanzania",
    "Thailand",
    "Timor-Leste",
    "Togo",
    "Tonga",
    "Trinidad and Tobago",
    "Tunisia",
    "Turkey",
    "Turkmenistan",
    "Tuvalu",
    "Uganda",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States of America",
    "Uruguay",
    "Uzbekistan",
    "Vanuatu",
    "Venezuela",
    "Vietnam",
    "Yemen",
    "Zambia",
    "Zimbabwe",
]


def retry_forever(f):
    def wrapped(*args, **kwargs):
        for _ in range(10):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                print(f'retrying {f.__name__} after {e!r}')
                time.sleep(15)
        print('failed retrieve - returning None')
        return None

    return wrapped


@retry_forever
def get_wikidata_country(location, sparql):
    # getting the country of 'location'

    query_s = """
    PREFIX bd: <http://www.bigdata.com/rdf#> 
    PREFIX mwapi: <https://www.mediawiki.org/ontology#API/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
    PREFIX wikibase: <http://wikiba.se/ontology#> 
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 

    SELECT ?item ?typeLabel
    WHERE { 
      SERVICE wikibase:mwapi {
        bd:serviceParam wikibase:api "EntitySearch" ;
                        wikibase:endpoint "www.wikidata.org" ;
                        mwapi:search "%s";
                        mwapi:language "en" . 
        ?item wikibase:apiOutputItem mwapi:item . 
        ?num wikibase:apiOrdinal true . 
      } 
      #?item (wdt:P279|wdt:P31) ?type 
      ?item wdt:P17 ?type 
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    } 
    ORDER BY ASC(?num)
    LIMIT 1""" % (
        location
    )

    print(f'querying location - {location}')
    sparql.setQuery(query_s)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"]) > 0:
        return results["results"]["bindings"][0]['typeLabel']['value']
    return None


def extract_locations(sents, countries):
    nlp = datautils.get_nlp()
    locs_cnt = Counter()
    for sent in tqdm(sents):
        premise = sent['sentence1']
        doc = nlp(premise)
        for e in doc.ents:
            if e.label_ in ['LOC', 'GPE']:
                locs_cnt[e.text.lower()] += 1
    return locs_cnt


def load_mnli(fname):
    df = pd.read_json(fname, lines=True)
    return df.to_dict(orient='records')


@cbox.cmd
def extract_mnli_locations(mnli_fname, outfile):
    print('started')
    mnli = load_mnli(mnli_fname)

    print('extracting locations from wikidata')
    countries = [c.lower() for c in COUNTRIES]
    locs_cnt = extract_locations(mnli, countries)

    with open(outfile, 'w') as fp:
        json.dump(dict(locs_cnt), fp, indent=4)


@cbox.cmd
def download_locations(locs_cnt_fname, outfile):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    countries = {c.lower() for c in COUNTRIES}

    with open(locs_cnt_fname) as fp:
        locs_cnt = json.load(fp)

    with open(outfile) as fp:
        found_locs = [json.loads(l)['location'] for l in fp]
    with open(outfile, 'a') as fp:
        for loc, cnt in locs_cnt.items():
            if loc in found_locs:
                print('already exists')
                continue
            elif '"' in loc or '?' in loc:
                print(f'invalid char at loc {loc}')
                continue
            res = get_wikidata_country(loc, sparql)
            is_country = loc.lower() in countries
            fp.write(
                json.dumps(
                    {
                        'location': loc,
                        'cnt': cnt,
                        'result': res,
                        'is_country': is_country,
                    }
                )
            )
            fp.write('\n')
            fp.flush()


def filter_locations_sents(sentence, doc, loc_entities_types=('LOC', 'GPE')):
    loc_entities = [e for e in doc.ents if e.label_ in loc_entities_types]

    # not interesting - no locations
    if len(loc_entities) == 0:
        return False, None

    return (
        True,
        {
            'num_locations': len(loc_entities),
            'locations': [str(e).lower() for e in loc_entities],
            'start_char': [e.start_char for e in loc_entities],
            'end_char': [e.end_char for e in loc_entities],
            # known to us - found on the db
            'known_location': [
                str(e).lower() in KNOWN_LOCATIONS for e in loc_entities
            ],
            # take either spacy number of span tokens or word regex - the higher
            'word_count': [
                max(e.end - e.start, count_words(str(e))) for e in loc_entities
            ],
        },
    )


def count_words(s):
    return len(re.findall(r'\w+', s, flags=re.I | re.U))


def extract_sentences_from_mnli(mnli_samples, filter_func):
    sentence2name = {'sentence1': 'permise', 'sentence2': 'hypothesis'}
    nlp = datautils.get_nlp()

    for sample_id, sample in enumerate(tqdm(mnli_samples)):
        for sent_id in sentence2name:
            sent = sample[sent_id]

            # parse it with spacy
            doc = nlp(sent)
            is_match, metadata = filter_func(sent, doc)

            if is_match:
                yield {
                    'sentence': sent,
                    'type': sentence2name[sent_id],
                    'genre': sample['genre'],
                    'sample_id': sample_id,
                    **metadata,
                }


@cbox.cmd
def extract_mnli_sentences_with_locations(mnli_fname, outfile):
    mnli_samples = load_mnli(mnli_fname)
    matches = extract_sentences_from_mnli(mnli_samples, filter_locations_sents)
    datautils.jsonl_dump(matches, outfile)


@cbox.cmd
def create_sentences_pool_for_filtering(infile, outfile):
    country2cities = get_country2cities_deduped()
    known_deduped_locations = {c for c in country2cities} | {
        c for cs in country2cities.values() for c in cs
    }

    # because we dedup strings each city has exactly one country
    city2country = {
        city: country
        for country, cities in country2cities.items()
        for city in cities
    }

    def _filter_sents(x):
        return (
            x['num_locations'] == 1
            and all(x['known_location'])
            and x['word_count'][0] == 1
            and x['genre'] != 'telephone'
            and x['locations'][0].lower() in known_deduped_locations
        )

    sents = [x for x in datautils.jsonl_load(infile) if _filter_sents(x)]

    random.seed('location location location')
    random.shuffle(sents)

    with open(outfile, 'w') as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                'original_sentence',
                'highlighted_sentence',
                'country',
                'location',
                'is_country',
                'start_char',
                'end_char',
                'id',
            ],
        )
        writer.writeheader()

        for s in sents:
            loc = s['locations'][0].lower()
            is_country = loc in country2cities

            if is_country:
                country = loc
                location = random.choice(country2cities[loc])
            else:
                country = city2country[loc]
                location = loc

            schar = s['start_char'][0]
            echar = s['end_char'][0]

            row = {
                'original_sentence': s['sentence'],
                'highlighted_sentence': s['sentence'][:schar]
                + '**'
                + s['sentence'][schar:echar]
                + '**'
                + s['sentence'][echar:],
                'country': country,
                'location': location,
                'is_country': is_country,
                'start_char': schar,
                'end_char': echar,
                'id': f'{s["sample_id"]}_{s["type"][0]}',
            }

            writer.writerow(row)


@cbox.cmd
def finalize_sentences_for_tagging(infile, outfile):
    df = pd.read_csv(infile)

    df = df[df['can_be_entailed'].isin({0, 1})]
    df = df.sample(n=len(df), random_state=42)
    df = df[
        [
            'id',
            'original_sentence',
            'can_be_entailed',
            'country',
            'start_char',
            'end_char',
        ]
    ]

    df = df.rename(columns={'original_sentence': 'sentence'})
    country2city = {
        'Albania': 'Durres',
        'Argentina': 'Salta',
        'Australia': 'Sydney',
        'Dominican': 'Republic	Santiago',
        'England': 'Liverpool',
        'Finland': 'Turku',
        'France': 'Lyon',
        'Germany': 'Munich',
        'Greece': 'Thessaloniki',
        'Guadeloupe': 'Lamentin',
        'Indonesia': 'Makassar',
        'India': 'Mumbai',
        'Iraq': 'Erbil',
        'Israel': 'Haifa',
        'Japan': 'Osaka',
        'Malaysia': 'Malacca',
        'Martinique': 'Ducos',
        'Nepal': 'Pokhara',
        'Netherlands': 'Rotterdam',
        'Nicaragua': 'Leon',
        'Philippines': 'Makati',
        'Portugal': 'Porto',
        'Romania': 'Oradea',
        'South Africa': 'Durban',
        'Sweden': 'Malmo',
        'United Kingdom': 'Bristol',
        'United States': 'Chicago',
        'Vietnam': 'Dalat',
    }
    df['country'] = df['country'].apply(str.title)
    df['location'] = df['country'].apply(country2city.__getitem__)

    random.seed('locations!!')
    df['other_location'] = df['country'].apply(
        lambda x: random.choice(
            sorted(set(country2city.values()) - {country2city[x]})
        )
    )

    df['sentence'] = df.apply(
        lambda x: x.sentence[: x.start_char]
        + '{{{location}}}'
        + x.sentence[x.end_char :],
        axis=1,
    )

    sec2_labels = ['Neutral', 'Contradiction']
    sec2_word_types = ['Country', 'Location']
    df['section2_label'] = [sec2_labels[i % 2] for i in range(len(df))]
    df['section2_word_type'] = [
        sec2_word_types[(i // 2) % 2] for i in range(len(df))
    ]

    df.to_csv(outfile, index=False)


def get_country2cities_deduped():
    country2city = datautils.json_load(COUNTRIES_FNAME)

    locations_cnt = Counter()
    for country, cities in country2city.items():
        locations_cnt.update([c.lower() for c in cities + [country.lower()]])

    deduped_country2cities = {}
    for country, cities in country2city.items():
        if locations_cnt[country.lower()] == 1:
            deduped_country2cities[country.lower()] = [
                c.lower() for c in cities if locations_cnt[c.lower()] == 1
            ]
    return deduped_country2cities


if __name__ == '__main__':
    cbox.main(
        [
            extract_mnli_locations,
            download_locations,
            extract_mnli_sentences_with_locations,
            create_sentences_pool_for_filtering,
            finalize_sentences_for_tagging,
        ]
    )
