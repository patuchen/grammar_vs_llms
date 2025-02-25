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

def define_noise_profiles():
    '''
    Define noise profiles, with the following structure:
    {<scenario_key>: {
        <noise_class>: { # like orthographic, lexical, etc.
            'p': <probability of applying noise type>,
            'subtype_distribution': { # like natural_typos, insertion, etc.
                <subtype>: <probability of applying subtype> # distribution over subtypes, should sum to 1
            }
        }
    }}
    '''
    
    noise_profiles = {'orthographic': {
                'orthographic': {
                    'p': 0.5,
                    'subtype_distribution': {
                        'natural_typos': 0.03,
                        'insertion': 0.17,
                        'omission': 0.37,
                        'transposition': 0.05,
                        'substitution': 0.38
                    }
                }
            },
            'L2': { # THIS IS JUST AN EXAMPLE!
                'orthographic': {
                    'p': 0.5,
                    'subtype_distribution': {
                        'natural_typos': 0.03,
                        'insertion': 0.17,
                        'omission': 0.37,
                        'transposition': 0.05,
                        'substitution': 0.38
                    }
                },
                'lexical': {
                    'p': 0.5,
                    'subtype_distribution': {
                        'synonym': 0.5,
                        'antonym': 0.5
                    }
                }
            },
            'TeenUser': { # THIS IS JUST AN EXAMPLE! 
                'orthographic': {
                    'p': 0.03,
                    'subtype_distribution': {
                        'natural_typos': 0.95,
                        'insertion': 0.01,
                        'omission': 0.01,
                        'transposition': 0.01,
                        'substitution': 0.02
                    }
                },
                'register': {
                    'p': 1,
                    'subtype_distribution': {
                        'formal': 0.5,
                        'informal': 0.5
                    }
                }
            }
    }
    return noise_profiles

