import numpy as np
import copy

neighbours = {
    'a': ['q', 'w', 's','z'], 
    'b': ['v',' ','g','h','n'],
    'c': ['x','d','f','v', ' '],
    'd': ['e', 'r', 'f','c','x','s'],
    'e': ['w','r','d','s'],
    'f': ['d','r','t','g','v','c'],
    'g': ['f','t','y','h','b','v'],
    'h': ['g','y','u','j','n','b'],
    'i': ['u','8','9','o','k','j'],
    'j': ['h','u','i','k','m','n'],
    'k': ['j','i','o','l',',','m'],
    'l': ['k','o','p'],
    'm': ['n','j','k',',',' '],
    'n': ['b','h','j','m',' '],
    'o': ['i','p','l','k'],
    'p': ['o','l'],
    'q': ['w','a'],
    'r': ['e','t','f','d'],
    's': ['a','w','e','d','x','z'],
    't': ['r','y','g','f'],
    'u': ['y','i','j','h'],
    'v': ['c','f','g','b',' '],
    'w': ['q','e','s','a'],
    'x': ['z','s','d','c',' '],
    'y': ['t', 'u','h','g'],
    'z': ['a','s','x'],
}

scenarios = ["orthographic", "lexicalphrasal", "register", "L2", "LazyUser"]

def is_vowel(char: str) -> bool:
        '''
        Check if a character is a vowel.
        '''
        if char in ['a', 'e', 'i', 'o', 'u']:
            return True
        return False
    
def is_word_initial(sentence: str, char_idx: int) -> bool:
    '''
    Check if a character is word initial.
    '''
    if char_idx == 0:
        return True
    if sentence[char_idx-1] == ' ':
        return True
    return False

def is_word_final(sentence: str, char_idx: int) -> bool:
    '''
    Check if a character is word final.
    '''
    if char_idx == len(sentence)-1:
        return True
    if sentence[char_idx+1] == ' ':
        return True
    return False

def count_vowels(sentence: str) -> int:
    '''
    Count the number of vowels in a sentence. Also return the number of vowels in {"a", "e", "i}.
    '''
    return sum([1 for char in sentence if is_vowel(char)]), sum([1 for char in sentence if char in ["a", "e", "i"]])

def define_noise_schemas():
    '''
    Define noise profile schemas, with the following structure:
    {<scenario_key>: {
        <noise_class>: { # like orthographic, lexical, etc.
            'subtype_distribution': { # like natural_typos, insertion, etc.
                <subtype>: <probability of applying subtype> # distribution over subtypes, should sum to 1
            }
        }
    }}
    Note that this is missing the level of noise for the noiser (parameter p), which may lie in some natural range given a scenario.
    '''
    
    noise_profiles = {
        # Individual noise class profiles
            'orthographic': {
                'orthographic': {
                    'subtype_distribution': {
                        'natural_typos': 0.03,
                        'insertion': 0.17,
                        'omission': 0.37,
                        'transposition': 0.05,
                        'substitution': 0.38
                    }
                }
            },
            'typos_synthetic': {
                'orthographic': {
                    'subtype_distribution': {
                        'natural_typos': 1,
                        'insertion': 0,
                        'omission': 0,
                        'transposition': 0,
                        'substitution': 0
                    }
                }
            },
            'lexicalphrasal': {
                'lexicalphrasal': {
                    'subtype_distribution': None
                }
            },
            'register': {
                'register': {
                    'subtype_distribution': None
                }
            },
            # Composite noise class profiles, corresponding to *scenarios*
            'L2': { 
                'orthographic': {
                    'subtype_distribution': {
                        'natural_typos': 0.03,
                        'insertion': 0.17,
                        'omission': 0.37,
                        'transposition': 0.05,
                        'substitution': 0.38
                    }
                },
                'lexicalphrasal': {
                    'subtype_distribution': None
                }
            },
            'LazyUser': { 
                'orthographic': {
                    'subtype_distribution': {
                        'natural_typos': 0.95,
                        'insertion': 0.01,
                        'omission': 0.01,
                        'transposition': 0.01,
                        'substitution': 0.02
                    }
                },
                'register': {
                    'subtype_distribution': None
                }
            }
    }
    return noise_profiles


def get_noise_profile_key(noise_profile: dict) -> str:
    '''
    Get a key for a noise profile.
    '''
    key = [f"{noiser_comp}_{noiser_comp_profile['p']:.2f}" for noiser_comp, noiser_comp_profile in noise_profile.items()]
    return "_".join(key)
        

def define_noise_profiles(scenario: str):
    '''
    Given a noise profile schema, we generate a list of noise profiles for each scenario, varying over natural ranges of levels of noise per noise class.
    '''
    noise_schemas = define_noise_schemas()
    noise_profiles = []
    noise_schema = noise_schemas[scenario]
    if scenario == "typos_synthetic":
        for p in np.linspace(0.01, 1, 10):
            noise_schema_copy = copy.deepcopy(noise_schema)
            noise_schema_copy['orthographic']['p'] = p
            noise_profiles.append(noise_schema_copy)
    if scenario == "orthographic":
        for p in np.linspace(0.04, 0.4, 10):
            noise_schema_copy = copy.deepcopy(noise_schema)
            noise_schema_copy['orthographic']['p'] = p
            noise_profiles.append(noise_schema_copy)
    elif scenario == "lexicalphrasal":
        for level in [1, 2]:
            noise_schema_copy = copy.deepcopy(noise_schema)
            noise_schema_copy['lexicalphrasal']['p'] = level
            noise_profiles.append(noise_schema_copy)
    elif scenario == "register":
        for level in [1, 2]:
            noise_schema_copy = copy.deepcopy(noise_schema)
            noise_schema_copy['register']['p'] = level
            noise_profiles.append(noise_schema_copy)
    elif scenario == "L2":
        for p in np.linspace(0.0, 0.4, 11):
            for level in [0, 1, 2]:
                noise_schema_copy = copy.deepcopy(noise_schema)
                noise_schema_copy['orthographic']['p'] = p            
                noise_schema_copy['lexicalphrasal']['p'] = level
                noise_profiles.append(noise_schema_copy)
    elif scenario == "LazyUser":
        for p in np.linspace(0.0, 0.4, 11):
            for level in [0, 1, 2]:
                noise_schema_copy = copy.deepcopy(noise_schema)
                noise_schema_copy['orthographic']['p'] = p            
                noise_schema_copy['register']['p'] = level
                noise_profiles.append(noise_schema_copy)

    return noise_profiles


def define_lexicalphrasal_database():
    '''
    This is our curated list of noised prompts per prompt. Level 1 is less noised, Level 2 is more noised.
    '''

    database_level1 = {
        "Translate the following text from {source_lang} to {target_lang}.\\n{source_text}": [
            "Translate this text from {source_lang} to {target_lang}.\\n{source_text}",
            "Change the following text from {source_lang} into {target_lang}.\\n{source_text}",
            "Please translate this text from {source_lang} to {target_lang}.\\n{source_text}",
            "Rewrite this text from {source_lang} in {target_lang}.\\n{source_text}",
            #"Convert this text from {source_lang} to {target_lang}.\\n{source_text}",
            "Make this text in {source_lang} into {target_lang}.\\n{source_text}",
            "Put this text in {target_lang} instead of {source_lang}.\\n{source_text}",
            "Turn this text into {target_lang} from {source_lang}.\\n{source_text}",
            "Write this text again but in {target_lang}.\\n{source_text}",
            "Translate the given text from {source_lang} into {target_lang}.\\n{source_text}"
        ],
        
        "Translate this from {source_lang} to {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:": [
            "Change this from {source_lang} into {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Translate this part from {source_lang} to {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Rewrite this from {source_lang} in {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            #"Convert this sentence from {source_lang} to {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Make this text in {target_lang} from {source_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            #"Switch this from {source_lang} into {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Write this again in {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Put this text into {target_lang} from {source_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Change this text into {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Translate this sentence from {source_lang} to {target_lang} exactly:\\n{source_lang}: {source_text}\\n{target_lang}:"
        ],

        "### Instruction:\\nTranslate Input from {source_lang} to {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n": [
            "### Instruction:\\nChange Input from {source_lang} into {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nRewrite Input from {source_lang} in {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            #"### Instruction:\\nConvert Input from {source_lang} to {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nMake Input in {target_lang} from {source_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            #"### Instruction:\\nSwitch Input from {source_lang} into {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nWrite Input again in {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nPut Input into {target_lang} from {source_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nTranslate the Input into {target_lang} from {source_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nChange the Input text from {source_lang} to {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nRewrite this Input in {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n"
        ],

        "Translate the following line from\\n{source_lang} to {target_lang}.\\nBe very literal, and only translate the content of the line, do not add any explanations: {source_text}": [
            "Change the following line from\\n{source_lang} to {target_lang}.\\nBe exact, do not add extra words: {source_text}",
            "Translate this line from\\n{source_lang} to {target_lang}.\\nDo not explain, just change words: {source_text}",
            #"Convert the following line into\\n{target_lang} from {source_lang}.\\nKeep it the same, do not add anything: {source_text}",
            "Rewrite the next line in\\n{target_lang} from {source_lang}.\\nBe exact, no extra explanation: {source_text}",
            "Put the next line into\\n{target_lang} from {source_lang}.\\nDo not change meaning, only words: {source_text}",
            "Make this sentence in\\n{target_lang} from {source_lang}.\\nBe very exact, no other words: {source_text}",
            #"Switch this line to\\n{target_lang} from {source_lang}.\\nKeep it the same, just change language: {source_text}",
            "Write this again in\\n{target_lang} from {source_lang}.\\nDo not explain, only change words: {source_text}",
            "Change this line into\\n{target_lang} from {source_lang}.\\nNo extra information, be literal: {source_text}",
            "Translate exactly this line from\\n{source_lang} to {target_lang}.\\nNo explanations, just change words: {source_text}"
        ],

        # GEMBA prompts
        "Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100, where a score of zero means 'no meaning preserved' and score of one hundred means 'perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ": [
            "Score the following translation from {source_lang} to {target_lang} on continuous scale from 0 to 100, where score of zero mean 'no meaning is preserved' and score of one hundred mean 'perfect meaning with grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Rate the following translation from {source_lang} to {target_lang} using a continuous scale from 0-100, where score 0 means 'no meaning kept' and 100 means 'perfect grammar and meaning'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Give score for translation from {source_lang} to {target_lang} in a scale from 0 to 100, where zero mean 'no meaning left' and 100 mean 'perfectly correct meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Evaluate the following translation from {source_lang} to {target_lang} on a continuous scale 0-100, where 0 is 'no meaning preserved' and 100 is 'perfect meaning with correct grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Translation must be rated from {source_lang} to {target_lang} using a continuous scale from 0-100, where 0 means 'no meaning saved' and 100 means 'perfect grammar and meaning'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Rate the translation from {source_lang} to {target_lang} with a score between 0-100, where 0 means 'meaning not kept' and 100 means 'fully correct grammar and meaning'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Give score to the translation from {source_lang} into {target_lang} on a scale from 0-100, where zero is 'no meaning saved' and one hundred is 'perfect sentence'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "This translation from {source_lang} to {target_lang} must be given a score on a scale of 0-100, where 0 represents 'no meaning at all' and 100 represents 'perfect grammar and correct meaning'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Evaluate this translation from {source_lang} to {target_lang} using scale 0-100, where score 0 mean 'nothing is same' and score 100 mean 'grammar and meaning are perfect'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Rate the following translation from {source_lang} to {target_lang} in a continuous scale between 0 to 100, where 0 mean 'no meaning is kept' and 100 mean 'perfectly correct in grammar and meaning'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: "
        ],

        "Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100 that starts on 'No meaning preserved', goes through 'Some meaning preserved', then 'Most meaning preserved and few grammar mistakes', up to 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ": [
            "Score the following translation from {source_lang} to {target_lang} on a continuous scale 0-100 that start at 'No meaning preserved', go to 'Some meaning preserved', then 'Most meaning with few grammar mistake', and end at 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Rate the translation from {source_lang} to {target_lang} using a continuous scale from 0 to 100 that begin at 'No meaning preserved', continue to 'Some meaning preserved', then 'Most meaning and few grammar mistake', finishing at 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Give score to the following translation from {source_lang} to {target_lang} in scale from 0-100, starting in 'No meaning preserved', then moving to 'Some meaning is preserved', then 'Most meaning but some grammar mistake', ending at 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Evaluate the translation from {source_lang} into {target_lang} in continuous scale from 0-100, where 0 is 'No meaning preserved', next is 'Some meaning preserved', then 'Most meaning is keep but grammar mistake', up to 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Translation from {source_lang} to {target_lang} must be rated in a continuous scale from 0-100, first level is 'No meaning preserved', then go through 'Some meaning is keep', then 'Most meaning with grammar mistake', and last is 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Rate this translation from {source_lang} into {target_lang} on a scale from 0-100, starting with 'No meaning preserved', after 'Some meaning kept', then 'Most meaning is there but with few grammar mistake', finally 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Score translation from {source_lang} to {target_lang} using a scale of 0-100, where 0 means 'No meaning is kept', then 'Some meaning is stay', then 'Most meaning but some grammar mistake', up to 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Give rating for translation from {source_lang} to {target_lang} in a scale of 0-100, first step is 'No meaning preserved', after that 'Some meaning is keep', then 'Most meaning but some mistake', and last 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Evaluate the translation quality from {source_lang} to {target_lang} in a scale of 0-100. Start from 'No meaning preserved', then 'Some meaning', after 'Most meaning but few grammar mistake', finally 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): "
        ],


        "Please analyze the given source and translated sentences and output a translation quality score on a continuous scale ranging from 0 to 100. Translation quality should be evaluated based on both fluency and adequacy. A score close to 0 indicates a low quality translation, while a score close to 100 indicates a high quality translation. Do not provide any explanations or text apart from the score.\n{source_lang} Sentence: {source_seg}\n{target_lang} Sentence: {target_seg}\nScore: ": [
            "Please analyze the given source and translated sentences and give a translation quality score on a continuous scale from 0 to 100. Translation quality should evaluate fluency and adequacy. A score close to 0 means low quality, while a score close to 100 means high quality. Do not give explanations or other text, only the score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Analyze the given source and translated sentences and provide a translation quality score using a continuous scale from 0 to 100. The score should be based on both fluency and adequacy. A score near 0 means low quality translation, while a score near 100 means high quality. No explanations, only the score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Check the given source and translated sentence and rate translation quality using a continuous scale from 0 to 100. Quality should be judged based on fluency and adequacy. A score near 0 means poor quality, while a score near 100 means excellent quality. Do not provide explanations, only score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Please evaluate the source and translated sentence and assign a translation quality score from 0 to 100. Translation quality should be assessed according to fluency and adequacy. A score close to 0 represents a bad translation, while a score close to 100 represents a very good one. No need for explanations, only the score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Analyze the source and translated sentence, then rate translation quality on a scale from 0 to 100. Translation quality must be measured in terms of fluency and adequacy. A score near 0 means poor translation, while a score near 100 means a perfect one. Do not include explanations, only give the score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Please check the source and translated sentence, then output a quality score between 0 and 100. Score should consider fluency and adequacy. A score close to 0 indicates low quality, while a score close to 100 indicates excellent quality. No explanation should be given, only the score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Evaluate the given source and translated sentence and provide a score from 0 to 100, where 0 means very poor translation and 100 means perfect translation. Fluency and adequacy should be used to determine the score. No extra text, just the score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Read the source and translated sentence and assign a translation quality score from 0 to 100, taking into account fluency and adequacy. A lower score means worse translation, while a higher score means better translation. Do not write any explanations, just the score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Analyze the translation quality of the given source and translated sentences and assign a score between 0 and 100. Score should be based on both fluency and adequacy. 0 means very poor quality, while 100 means excellent quality. No explanations, only the score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Please assess the quality of the translation by assigning a score from 0 to 100, using fluency and adequacy as criteria. A score close to 0 means a bad translation, while a score close to 100 means a high-quality translation. Only provide the score, no explanations.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: "
        ]
        
    }

    # Level 2 is more noised 
    database_level2 = {
        "Translate the following text from {source_lang} to {target_lang}.\\n{source_text}": [
            "Translate this following text from {source_lang} into {target_lang}.\\n{source_text}",
            "Make translate this text from {source_lang} to {target_lang}.\\n{source_text}",
            "You translate this text from {source_lang} into {target_lang}.\\n{source_text}",
            "Change this text to {target_lang} from {source_lang}.\\n{source_text}",
            "Translate text from {source_lang} for {target_lang}.\\n{source_text}",
            "Rewrite this text in {target_lang} from {source_lang}.\\n{source_text}",
            "Put this text in {target_lang} instead {source_lang}.\\n{source_text}",
            "Make this text into {target_lang} from {source_lang}.\\n{source_text}",
            "Translate now this text {source_lang} to {target_lang}.\\n{source_text}",
            "This text you change from {source_lang} into {target_lang}.\\n{source_text}"
        ],

        "Translate this from {source_lang} to {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:": [
            "Change this for {target_lang} from {source_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Translate now this {source_lang} into {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Make this translation {source_lang} for {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "You translate this text to {target_lang} from {source_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Rewrite text into {target_lang} from {source_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Put this words in {target_lang}, from {source_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Make this text be {target_lang} and not {source_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            #"Switch the text into {target_lang}, take from {source_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "You make words in {target_lang}, not in {source_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "Turn this words in {target_lang} from {source_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:"
        ],

        "### Instruction:\\nTranslate Input from {source_lang} to {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n": [
            "### Instruction:\\nYou change Input from {source_lang} into {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            #"### Instruction:\\nMake Input translate from {source_lang} to {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nNow translate Input into {target_lang} from {source_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nRewrite Input from {source_lang} to {target_lang} only words\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nTranslate now this Input into {target_lang} from {source_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nPut Input in {target_lang} from {source_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nMake this Input in {target_lang}, not {source_lang}\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nChange Input into {target_lang} now\\n### Input:\\n{source_text}\\n### Response:\\n",
            #"### Instruction:\\nSwitch this Input to {target_lang}, no more\\n### Input:\\n{source_text}\\n### Response:\\n",
            "### Instruction:\\nYou take Input and put it in {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n"
        ],

        "Translate the following line from\\n{source_lang} to {target_lang}.\\nBe very literal, and only translate the content of the line, do not add any explanations: {source_text}": [
            "Change the next line from\\n{source_lang} into {target_lang}.\\nBe exact, no more: {source_text}",
            "Translate now this line from\\n{source_lang} to {target_lang}.\\nDo not put more words: {source_text}",
            "Make the next line into\\n{target_lang} from {source_lang}.\\nOnly words, not explain: {source_text}",
            "Rewrite this line in\\n{target_lang} from {source_lang}.\\nBe very exact, no change: {source_text}",
            "Turn the sentence be in\\n{target_lang} not {source_lang}.\\nKeep same, not explain: {source_text}",
            "Put the words in\\n{target_lang} from {source_lang}.\\nDo not put more things: {source_text}",
            "Make sentence be in\\n{target_lang} from {source_lang}.\\nOnly words, nothing extra: {source_text}",
            #"Switch this line into\\n{target_lang} from {source_lang}.\\nDo not add, only change: {source_text}",
            "Now change this line into\\n{target_lang} from {source_lang}.\\nKeep the same words: {source_text}",
            "You must rewrite this line in\\n{target_lang} from {source_lang}.\\nNo extra, only same words: {source_text}"
        ],

        # GEMBA PROMPTS

        "Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100, where a score of zero means 'no meaning preserved' and score of one hundred means 'perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ": [
            "Score the following translation from {source_lang} into {target_lang} on continuous scale from 0 to 100, where score zero mean 'no meaning is preserved' and score hundred mean 'perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Give score for the following translation from {source_lang} to {target_lang} on a continuous scale of 0-100, where score 0 means 'no meaning is keep' and score 100 means 'perfect grammar and meaning'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Rate the following translation from {source_lang} to {target_lang} on a scale from 0 until 100, where 0 mean 'no meaning is left' and 100 mean 'perfect meaning with grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Score this translation from {source_lang} to {target_lang} in a scale of 0-100, where a score 0 means 'no meaning stay' and a 100 means 'perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Translation should be scored from {source_lang} into {target_lang} in a continuous scale from 0-100, where 0 is 'meaning lost' and 100 is 'perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Score the following translation from {source_lang} to {target_lang} using scale 0 to 100, where score of 0 mean 'no meaning preserved' and 100 mean 'perfect grammar and meaning'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Give a score for the following translation from {source_lang} into {target_lang}, where 0 is 'meaning not keep' and 100 is 'perfect grammar with meaning'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            # "This translation from {source_lang} to {target_lang} should get a score between 0-100. 0 is 'meaning not preserved', 100 is 'perfect sentence'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Rate the following translation from {source_lang} to {target_lang} with a score of 0-100, where zero mean 'no meaning' and hundred mean 'perfect grammar and meaning'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: ",
            "Score this translation from {source_lang} to {target_lang} on a continuous scale between 0-100, where 0 is 'nothing preserved' and 100 is 'perfectly correct meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore: "
        ],

        "Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100 that starts on 'No meaning preserved', goes through 'Some meaning preserved', then 'Most meaning preserved and few grammar mistakes', up to 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ": [
            "Score the following translation from {source_lang} into {target_lang} on continuous scale from 0 to 100, start at 'No meaning preserved', then 'Some meaning preserved', then 'Most meaning with few grammar mistake', and 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Rate the following translation from {source_lang} to {target_lang} in a scale 0-100, beginning from 'No meaning preserved', after that 'Some meaning preserved', next 'Most meaning and few grammar mistake', then 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Give score to translation from {source_lang} to {target_lang} in scale of 0 until 100, first is 'No meaning preserved', then 'Some meaning is preserved', after 'Most meaning preserved but few grammar mistake', up to 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Evaluate the translation from {source_lang} into {target_lang} in a continuous scale 0-100, where 0 is 'No meaning preserved', next 'Some meaning is keep', then 'Most meaning preserved with few grammar mistakes', last 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Score translation from {source_lang} to {target_lang} on scale of 0-100, first step 'No meaning preserved', after that 'Some meaning preserved', then 'Most meaning preserved and some grammar mistake', up until 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Rate this translation from {source_lang} into {target_lang} with scale of 0-100, where first level is 'No meaning preserved', after 'Some meaning is there', next 'Most meaning but with few grammar mistake', and last 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            # "Translation from {source_lang} to {target_lang} must be rated in scale from 0-100, first 'No meaning preserved', then 'Some meaning is keep', then 'Most meaning but have grammar mistake', and last 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Score the following translation from {source_lang} to {target_lang} using scale 0-100, where 0 mean 'No meaning is preserved', next 'Some meaning is there', after 'Most meaning but few grammar mistakes', and last is 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Give score for translation from {source_lang} into {target_lang} in range 0-100, beginning in 'No meaning preserved', then 'Some meaning preserved', then 'Most meaning but some grammar mistakes', until 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): ",
            "Rate translation from {source_lang} to {target_lang} with a score between 0-100, start in 'No meaning preserved', next 'Some meaning preserved', then 'Most meaning but have grammar mistake', up until 'Perfect meaning and grammar'.\\n{source_lang} source: '{source_seg}'\\n{target_lang} translation: '{target_seg}'\\nScore (0-100): "
        ],
        

        "Please analyze the given source and translated sentences and output a translation quality score on a continuous scale ranging from 0 to 100. Translation quality should be evaluated based on both fluency and adequacy. A score close to 0 indicates a low quality translation, while a score close to 100 indicates a high quality translation. Do not provide any explanations or text apart from the score.\n{source_lang} Sentence: {source_seg}\n{target_lang} Sentence: {target_seg}\nScore: ": [
            "Please analyze given source and translated sentence, then give translation quality score from 0 to 100. Quality must be check on fluency and adequacy. Score close to 0 mean translation is bad, score near 100 mean translation is good. No explanations, only number.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Analyze source and translated sentence, output score between 0-100 for quality. Check for fluency and adequacy. Score near 0 is bad translation, near 100 is very good. No extra text, only number.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Look at source and translated sentence, then score translation from 0-100. Score 0 mean low quality, score 100 mean very good quality. Use fluency and adequacy for rating. No explanations, just score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Check source and translated text, then give translation score 0-100. 0 is very bad, 100 is perfect. Score must base on fluency and adequacy. No text, only number.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Please look at source and translation, give quality score between 0 to 100. Low score mean bad, high score mean good. Fluency and adequacy must be checked. No explanations, only score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Analyze both sentence and rate translation between 0-100. 0 is poor quality, 100 is best quality. Score must consider fluency and adequacy. No need to explain, just write score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Read source and translated sentence, then give score 0 to 100 for quality. 0 means very bad, 100 means perfect. Score depend on fluency and adequacy. No extra words, only score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Analyze translation and give number score from 0-100. Score low if bad, score high if very good. Fluency and adequacy must be used. No explanation, only score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Look at both sentences, then give score between 0-100. Lower score is worse, higher is better. Fluency and adequacy must be checked. No extra text, just score.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: ",
            "Check quality of translation, give score 0-100. Score must depend on fluency and adequacy. 0 is very poor, 100 is perfect. No need to write anything else, just number.\\n{source_lang} Sentence: {source_seg}\\n{target_lang} Sentence: {target_seg}\\nScore: "
        ]
    }
    database = {1: database_level1, 2: database_level2}


    return database


def define_register_database():

    database_level1 = {
        "Translate the following text from {source_lang} to {target_lang}.\\n{source_text}": [
            "translate this text {source_lang} to {target_lang}.\\n{source_text}",
            "change text from {source_lang} to {target_lang}.\\n{source_text}",
            #"convert {source_lang} to {target_lang}.\\n{source_text}",
            "rewrite in {target_lang}, from {source_lang}.\\n{source_text}",
            "make this {target_lang} instead.\\n{source_text}",
            "turn into {target_lang} from {source_lang}.\\n{source_text}",
            "need this in {target_lang}.\\n{source_text}",
            #"switch lang to {target_lang}.\\n{source_text}",
            #"fix this in {target_lang}.\\n{source_text}",
            "translate {source_text} to {target_lang}"
        ],

        "Translate this from {source_lang} to {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:": [
            "change lang {source_lang} -> {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "need {target_lang} for this: \\n{source_lang}: {source_text}\\n{target_lang}:",
            #"convert to {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            #"fix text in {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "rewrite {source_text} in {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            #"swap {source_lang} for {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "make this {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "turn this into {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "quick translate {source_lang} to {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "{target_lang} version of this pls: \\n{source_lang}: {source_text}\\n{target_lang}:"
        ],

        "### Instruction:\\nTranslate Input from {source_lang} to {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n": [
            "### instruction:\\nchange input {source_lang} -> {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\nmake input {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            #"### instruction:\\nconvert text {source_lang} to {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            #"### instruction:\\nfix this in {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\ntranslate input to {target_lang}, no extra\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\nput this in {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\nrewrite in {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            #"### instruction:\\nquick swap to {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\nneed {target_lang} version\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\nturn input into {target_lang}\\n### input:\\n{source_text}\\n### response:\\n"
        ],

        "Translate the following line from\\n{source_lang} to {target_lang}.\\nBe very literal, and only translate the content of the line, do not add any explanations: {source_text}": [
            "change next line to {target_lang}, literal only:\\n{source_text}",
            "translate this, no extras:\\n{source_text}",
            "make this {target_lang}, exact:\\n{source_text}",
            #"fix text in {target_lang}, nothing else:\\n{source_text}",
            #"swap to {target_lang}, just words:\\n{source_text}",
            "put this in {target_lang}, no more:\\n{source_text}",
            "rewrite {source_text} in {target_lang}, keep same",
            "quick translate, no explain:\\n{source_text}",
            "need {target_lang} version, literal:\\n{source_text}",
            "turn this into {target_lang}, no changes:\\n{source_text}"
        ]
    }
    database_level2 = {
        "Translate the following text from {source_lang} to {target_lang}.\\n{source_text}": [
            "translate this text {source_lang} to {target_lang}.\\n{source_text}",
            "change text from {source_lang} to {target_lang}.\\n{source_text}",
            #"convert {source_lang} to {target_lang}.\\n{source_text}",
            "rewrite in {target_lang}, from {source_lang}.\\n{source_text}",
            "make this {target_lang} instead.\\n{source_text}",
            "turn into {target_lang} from {source_lang}.\\n{source_text}",
            "need this in {target_lang}.\\n{source_text}",
            #"switch lang to {target_lang}.\\n{source_text}",
            #"fix this in {target_lang}.\\n{source_text}",
            "translate {source_text} to {target_lang}"
        ],

        "Translate this from {source_lang} to {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:": [
            "change lang {source_lang} -> {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "need {target_lang} for this: \\n{source_lang}: {source_text}\\n{target_lang}:",
            #"convert to {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            #"fix text in {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "rewrite {source_text} in {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            #"swap {source_lang} for {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "make this {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "turn this into {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "quick translate {source_lang} to {target_lang}:\\n{source_lang}: {source_text}\\n{target_lang}:",
            "{target_lang} version of this pls: \\n{source_lang}: {source_text}\\n{target_lang}:"
        ],

        "### Instruction:\\nTranslate Input from {source_lang} to {target_lang}\\n### Input:\\n{source_text}\\n### Response:\\n": [
            "### instruction:\\nchange input {source_lang} -> {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\nmake input {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            #"### instruction:\\nconvert text {source_lang} to {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            #"### instruction:\\nfix this in {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\ntranslate input to {target_lang}, no extra\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\nput this in {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\nrewrite in {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            #"### instruction:\\nquick swap to {target_lang}\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\nneed {target_lang} version\\n### input:\\n{source_text}\\n### response:\\n",
            "### instruction:\\nturn input into {target_lang}\\n### input:\\n{source_text}\\n### response:\\n"
        ],

        "Translate the following line from\\n{source_lang} to {target_lang}.\\nBe very literal, and only translate the content of the line, do not add any explanations: {source_text}": [
            "change next line to {target_lang}, literal only:\\n{source_text}",
            "translate this, no extras:\\n{source_text}",
            "make this {target_lang}, exact:\\n{source_text}",
            #"fix text in {target_lang}, nothing else:\\n{source_text}",
            #"swap to {target_lang}, just words:\\n{source_text}",
            "put this in {target_lang}, no more:\\n{source_text}",
            "rewrite {source_text} in {target_lang}, keep same",
            "quick translate, no explain:\\n{source_text}",
            "need {target_lang} version, literal:\\n{source_text}",
            "turn this into {target_lang}, no changes:\\n{source_text}"
        ]
    }
    database = {1: database_level1, 2: database_level2}

    return database

