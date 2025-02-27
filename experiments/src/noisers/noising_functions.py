from typing import Dict, List
from collections import defaultdict
import numpy as np
from utils import neighbours as typo_neighbours
from utils import is_vowel, is_word_initial, is_word_final, count_vowels, define_lexicalphrasal_database, define_register_database
np.random.seed(42)

class Orthographic:
    '''
    Class for generating orthographic errors.
    We define different **classes** of errors: {natural_typos, insertion, omission, transposition, substitution}.
    Each class has a set of **subclasses** that generate specific errors.
    Subclasses are sampled uniformly given the class.
    '''


    def __init__(self, subtype_distribution: Dict[str, float], p: float):
        '''
        subtype_distribution (Dict[str, float]): Defines a distribution over the different classes of orthographic errors.
        '''
        self.subtype_distribution = subtype_distribution
        self.p = p


    def natural_typos_substitute(self, sentence: str, char_idx: int) -> str:
        '''Substitute a character with a nearby character on the keyboard.'''
        char = sentence[char_idx].lower()
        if char in typo_neighbours:
            new_char = np.random.choice(typo_neighbours[char])
            if sentence[char_idx].isupper():
                new_char = new_char.upper()
            sentence = sentence[:char_idx] + new_char + sentence[char_idx+1:]
        return sentence, char_idx
    
    def natural_typos_transposition(self, sentence: str, char_idx: int) -> str:
        '''
        Swap a character with the previous character.
        '''
        sentence = sentence[:char_idx-1] + sentence[char_idx] + sentence[char_idx-1] + sentence[char_idx+1:]
        return sentence, char_idx
        

    def natural_typos_omission(self, sentence: str, char_idx: int) -> str:
        '''
        Omit a character.
        '''
        sentence = sentence[:char_idx] + sentence[char_idx+1:]
        return sentence, char_idx-1 # Move back one character after omission
    
    def natural_typos_doubling(self, sentence: str, char_idx: int) -> str:
        '''
        Double a character.
        '''
        sentence = sentence[:char_idx] + sentence[char_idx] + sentence[char_idx:]
        return sentence, char_idx+1 # Move forward one character after doubling
    
    def insertion_consonant_doubling(self, sentence: str, char_idx: int) -> str:
        '''
        Doubles a non-word-initial consonant.
        '''
        sentence = sentence[:char_idx] + sentence[char_idx] + sentence[char_idx:] 
        return sentence, char_idx+1 # Move forward one character after doubling
    
    def omission_consonant_pair(self, sentence: str, char_idx: int) -> str:
        '''
        Omitting one of a consonant pair: ct, ck, cq, ch, gh
        '''
        sentence = sentence[:char_idx] + sentence[char_idx+1:]
        return sentence, char_idx-1 # Move back one character after omission
    
    def omission_consonant_is_valid(self, sentence: str, char_idx: int) -> bool:
        '''
        Check if consonant pair is vulnerable.
        '''
        affected_pairs = {"ct", "ck", "cq", "ch", "gh", "kt", "mn"}
        if not is_word_initial(sentence, char_idx) and sentence[char_idx-1:char_idx+1] in affected_pairs:
            return True
        return False
    
    def omission_r_dropping(self, sentence: str, char_idx: int) -> str:
        '''
        Omitting an r before a consonant.
        '''
        sentence = sentence[:char_idx] + sentence[char_idx+1:]
        return sentence, char_idx-1 # Move back one character after omission
    
    def omission_e_dropping(self, sentence: str, char_idx: int) -> str:
        '''
        Omitting e if 1) it is word final, 2) it's after for 3) it's before ly
        '''
        sentence = sentence[:char_idx] + sentence[char_idx+1:]
        return sentence, char_idx-1 # Move back one character after omission
    
    def omission_e_dropping_is_valid(self, sentence: str, char_idx: int) -> bool:
        '''
        Check if e dropping is valid.
        '''
        if sentence[char_idx] == 'e':
            if is_word_final(sentence, char_idx):
                return True
            if char_idx >= 3 and sentence[char_idx-3:char_idx] == 'for':
                return True
            if sentence[char_idx-1:char_idx+2] == 'ly':
                return True
        return False
    
    def transposition_vowel_pairs(self, sentence: str, char_idx: int) -> str:
        '''
        Transposing consecutive vowels.
        '''
        sentence = sentence[:char_idx] + sentence[char_idx+1] + sentence[char_idx] + sentence[char_idx+2:]
        return sentence, char_idx+1
    
    def transposition_common_pairs(self, sentence: str, char_idx: int) -> str:
        '''
        Transposing consecutive consonants.
        '''
        sentence = sentence[:char_idx] + sentence[char_idx+1] + sentence[char_idx] + sentence[char_idx+2:]
        return sentence, char_idx+1

    def transposition_common_pairs_is_valid(self, sentence: str, char_idx: int) -> bool:
        # ur, er, re, ng, gn, ph, gh, th, sp
        if not is_word_initial(sentence, char_idx) and sentence[char_idx-1:char_idx+1] in {"ur", "er", "re", "ng", "gn", "ph", "gh", "th", "sp"}:
            return True
        return False
    
    def substitution_vowels(self, sentence: str, char_idx: int) -> str:
        '''
        Substituting a vowel for another. We do this in a weighed manner, based on observed errors.
        TODO: Incorporate weights for {a, e, i}, etc.
        '''
        vowels = {'a', 'e', 'i', 'o', 'u'}
        vowels.remove(sentence[char_idx])
        if sentence[char_idx] == "i":
            vowels.add("y")
        new_vowel = np.random.choice(list(vowels))
        # The following simulates the observed error distribution for vowel substitution,
        # where 66% of errors are confusions of {"a", "e", "i"} and 34% are other pairs.
        if sentence[char_idx] in {"a", "e", "i"} and new_vowel in {"a", "e", "i"}:
            change_ratio = 0.66
        else:
            change_ratio = 0.34
        if np.random.rand() < change_ratio:
            sentence = sentence[:char_idx] + new_vowel + sentence[char_idx+1:]

        return sentence, char_idx

    
    def substitution_consonants(self, sentence: str, char_idx: int) -> str:
        '''
        Substituting a consonant for another.
        '''
        sibilants = {"s", "c", "z"}
        d_t = {"d", "t"}
        y_i = {"y", "i"}
        relevant_set = [char_set for char_set in [sibilants, d_t, y_i] if sentence[char_idx] in char_set][0]
        relevant_set.remove(sentence[char_idx])
        new_char = np.random.choice(list(relevant_set))
        sentence = sentence[:char_idx] + new_char + sentence[char_idx+1:]
        return sentence, char_idx
    
    def substitution_consonants_is_valid(self, sentence: str, char_idx: int) -> bool:
        '''
        Check if consonant substitution is valid.
        '''
        sibilants = {"s", "c", "z"}
        d_t = {"d", "t"}
        y_i = {"y", "i"}
        relevant_set = [char_set for char_set in [sibilants, d_t, y_i] if sentence[char_idx] in char_set]
        if len(relevant_set) > 0:
            return True
        return False

    
    def find_relevant_subclasses(self, sentence: str, error_class: str, char_idx: int) -> List:
        '''
        Given a character, find relevant subclasses of orthographic errors per class.
        '''
        # Ignore punctuation, newlines, and spaces
        if sentence[char_idx] in {".", ",", "!", "?", " ", "'", "\\"}:
            return []
        if sentence[char_idx] == "n" and char_idx > 0 and sentence[char_idx-1] == "\\":
            return []
        

        subclasses = []
        if error_class == 'natural_typos':
            # Natural typos
            subclasses.append(self.natural_typos_substitute)
            if not is_word_initial(sentence, char_idx) and not sentence[char_idx-1].isupper(): # Do not transpose if word initial or previous character is uppercase
                subclasses.append(self.natural_typos_transposition)
            if not is_word_initial(sentence, char_idx) and not sentence[char_idx-1].isupper(): # Do not transpose if word initial or previous character is uppercase
                subclasses.append(self.natural_typos_omission)
            if not sentence[char_idx].isupper():
                subclasses.append(self.natural_typos_doubling)
        elif error_class == 'omission':
            # Omission
            if self.omission_consonant_is_valid(sentence, char_idx):
                subclasses.append(self.omission_consonant_pair)
            if sentence[char_idx] == 'r' and not is_word_initial(sentence, char_idx) and not is_word_final(sentence, char_idx) and not is_vowel(sentence[char_idx+1]):
                subclasses.append(self.omission_r_dropping)
            if self.omission_e_dropping_is_valid(sentence, char_idx):
                subclasses.append(self.omission_e_dropping)
        elif error_class == 'transposition':
            # Transposition
            if not is_word_initial(sentence, char_idx) and is_vowel(sentence[char_idx]) and is_vowel(sentence[char_idx-1]):
                subclasses.append(self.transposition_vowel_pairs)
            if self.transposition_common_pairs_is_valid(sentence, char_idx):
                subclasses.append(self.transposition_common_pairs)
        elif error_class == 'substitution':
            # Substitution
            if is_vowel(sentence[char_idx]) and not is_word_initial(sentence, char_idx) and not is_word_final(sentence, char_idx):
                subclasses.append(self.substitution_vowels)
            if self.substitution_consonants_is_valid(sentence, char_idx):
                subclasses.append(self.substitution_consonants)
        elif error_class == 'insertion':
            # Insert
            if not is_vowel(sentence[char_idx]) and not is_word_initial(sentence, char_idx):
                subclasses.append(self.insertion_consonant_doubling)


        return subclasses

    
    def noise(self, sentence: str) -> str:
        '''
        Given a sentence, apply orthographic errors on characters with probability p.
        For each character / unit:
        1. Sample a class of error, 
        2. Find relevant subclasses for the character. If none exist, skip the character.
        3. Sample a subclass
        4. Noise the character according to the subclass
        '''
        
        char_idx = 0
        while char_idx < len(sentence):
            # Do not replace anything in placeholders - there is still the source sentence placeholder
            if sentence[char_idx] == '{':
                close_i = sentence.find('}', char_idx)
                char_idx = close_i + 1
                continue
            if np.random.rand() < self.p:
                # Sample a class of error
                classes = list(self.subtype_distribution.keys())
                error_class = np.random.choice(classes, p=[self.subtype_distribution[c] for c in classes])
                subclasses = self.find_relevant_subclasses(sentence = sentence, error_class = error_class, char_idx = char_idx)
                if len(subclasses) > 0:
                    # Sample a subclass
                    subclass = np.random.choice(subclasses)
                    sentence, char_idx = subclass(sentence, char_idx)
            char_idx += 1

        return sentence

                
class LexicalPhrasal:

    def __init__(self, subtype_distribution: Dict[str, float], p: int):
        '''
        p: Level of noise, should be in {0,1,2}
        subtype_distribution (Dict[str, float]): Defines a distribution over the different classes of lexical errors.
        '''
        self.subtype_distribution = subtype_distribution
        self.p = p
        self.database = define_lexicalphrasal_database()

    def noise(self, sentence: str) -> str:
        '''
        We have a pre-curated list of noised prompts per prompt that we want to experiment with, for each level of noise.
        Given a sentence, we simply look it up in our database and sample uniformly from the noised prompts.
        '''
        if self.p == 0:
            return sentence
        noised_prompts = self.database[self.p][sentence]
        return np.random.choice(noised_prompts)
    

class Register:

    def __init__(self, subtype_distribution: Dict[str, float], p: int):
        '''
        p: Level of noise, should be in {0,1,2}
        subtype_distribution (Dict[str, float]): Defines a distribution over the different classes of lexical errors.
        '''
        self.subtype_distribution = subtype_distribution
        self.p = p
        self.database = define_register_database()

    def noise(self, sentence: str) -> str:
        '''
        We have a pre-curated list of noised prompts per prompt that we want to experiment with, for each level of noise.
        Given a sentence, we simply look it up in our database and sample uniformly from the noised prompts.
        '''
        if self.p == 0:
            return sentence
        noised_prompts = self.database[self.p][sentence]
        return np.random.choice(noised_prompts)



class ComposeNoise:

    '''This class accepts a *noise profile* and applies noise according to the profile.'''
    def __init__(self, profile: Dict[str, Dict]):
        '''
        profile: A dictionary that defines the noise profile. e.g. : 
        {'orthographic': {'p': 0.5, 'subtype_distribution': {'natural_typos': 0.03, 'insertion': 0.17, 'omission': 0.37, 'transposition': 0.05, 'substitution': 0.4}}}
        '''
        self.profile = profile
        self.noisers = {}
        for noiser_type, noiser_profile in profile.items():
            if noiser_type == 'orthographic':
                self.noisers[noiser_type] = Orthographic(noiser_profile['subtype_distribution'], noiser_profile['p'])
            elif noiser_type == 'lexicalphrasal':
                self.noisers[noiser_type] = LexicalPhrasal(noiser_profile['subtype_distribution'], noiser_profile['p'])
            elif noiser_type == 'register':
                self.noisers[noiser_type] = Register(noiser_profile['subtype_distribution'], noiser_profile['p'])
    
    def noise(self, sentence: str) -> str:
        '''
        Apply noise in the following order: lexicalphrasal / register -> orthographic.
        Note that we cannot currently compose lexicalphrasal and register errors. 
        '''
        order = ['lexicalphrasal', 'register', 'orthographic']
        for noiser_type in order:
            if noiser_type in self.noisers:
                sentence = self.noisers[noiser_type].noise(sentence)
        return sentence
    

