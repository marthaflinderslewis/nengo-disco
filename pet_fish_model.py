import nengo
import nengo_spa as spa
import numpy as np

################################################################################################
# There are two functions that build networks: one that collapses the adjective features
# and one that expands them out.  Both functions take the same parameters. If you want to
# run the model against multiple adjectives, you stick them in a list and the model will
# run the nouns through the first adjective, then move on to the second one.
#
# You can auto-build your adjectives.  See the auto_build_adjective function for more info.
# 
# You can hand-build your adjectives as well.  Just put them in a dict and pass it into either
# of the build network functions.
################################################################################################


def get_features(noun_dict):
    """
    Input: a DICTIONARY of nouns where the key is the noun as a string, and the value a
        string containing its spa equation
    
    Output: a STRING of features formatted for spa
    """

    features = [value for value in noun_dict.values()]
    
    stripped = []
    
    for feature in features:
        bad_chars = '+-*.0123456789()'
        test = "".join(char for char in feature if char not in bad_chars)
        stripped.append(test.split(' '))
    
    all_features = []
    
    for index, value in enumerate(stripped):
        for inner_index, inner_value in enumerate(value):
            all_features.append(inner_value)
    
    all_features = list(set(all_features))
    all_features = [feature for feature in all_features if feature != '']
    
    answer = ';'.join(feature for feature in all_features)

    print('\n get_features function: \n\n', answer, '\n\n')
    return answer
    

def auto_build_adjective(spa_statement, only_self, do_all, omit):
    """
    Input: (i) a STRING of features formated for spa.  Please use _ instead of spaces (e.g., LIVES_HOUSE)
          (ii) a list of features to only bind with themselves
         (iii) a list of features to bind with all other features
          (iv) a list of features to omit from equations
    
    Output: a DICTIONARY of convolutions where each key is a feature and each value a string 
        for the convolution of that feature
    """
    
    features = spa_statement.replace(' ', '')
    features = features.split(';')
    
    all_pairs = [feature for feature in features if (feature not in only_self) and (feature not in omit)]
    
    row_pairs = []
    for index, value in enumerate(do_all):
        sub_list = []
        for inner_index, inner_value in enumerate(features):
            sub_list.append(value + '*' + inner_value)
        row_pairs.append(sub_list)
    
    for value in only_self:
        row_pairs.append([value + '*' + value])
    
    equations = []
    for row in row_pairs:
        equations.append(' + '.join(row))
        
    answer = {}
    for feature in features:
        for equation in equations:
            if equation.startswith(feature):
                answer.update({feature: equation})
        
    print('\n build_adjective funcion:\n\n', answer, '\n\n')
    return answer


def build_network_collapsed(noun, inverse_noun, all_adj_dicts, features, D):
    """
    Input: (i) A noun dictionary where keys are the nouns and values are their equations
          (ii) A dictionary whose keys are the noun dictionaries values (and visa-versa)
         (iii) A list of adjective dictionaries.  Each dict is for a new adjective.
               Each key in each dict is feature (e.g., CARED_FOR) and each value is the
               equation for that key.
          (iv) A string of features
           (v) the dimensionality of pointers
    
    Output: A spa network where the adjective is collapsed into a single transcode and
            bound to the noun
    """
    
    
    def set_noun(t, noun):
        all_nouns = [key for key in noun.keys()]
        index = int(t/0.3) % len(noun)
        
        return all_nouns[index]
        
    noun_vocab = spa.Vocabulary(D)
    noun_vocab.populate(";".join(noun.keys()))
    features_vocab = spa.Vocabulary(D)
    features_vocab.populate(features)
    
    current_noun = spa.Transcode(lambda t: set_noun(t, noun), output_vocab=noun_vocab, label='NOUN')
    
    noun_attributes = spa.ThresholdingAssocMem(
        threshold=0.2,
        input_vocab=noun_vocab,
        output_vocab=features_vocab,
        mapping = noun,
        function=lambda x: x > 0,
        label='NOUN FEATURES'
        )
    
    current_noun >> noun_attributes
    
    for adjective in all_adj_dicts: # Builds for each adjecive in adjective list
        adjective_definition = "+".join(adjective.values())
        adjective = spa.Transcode(adjective_definition, output_vocab=features_vocab, label='ADJECTIVE')
        wta_cleanup = spa.ThresholdingAssocMem(0.35, input_vocab = features_vocab, mapping=features_vocab.keys(), function=lambda x: x > 0, label='CLEAN UP')
        output = spa.State(vocab=features_vocab, label='A-N FEATURES')
        
        adjective * ~noun_attributes >> wta_cleanup
        wta_cleanup >> output
        
        match_noun = spa.WTAAssocMem(
            threshold=0.7,
            input_vocab=features_vocab,
            output_vocab = noun_vocab,
            mapping=inverse_noun,
            function=lambda x: x > 0,
            label='OUTPUT NOUN'
            )
            
        output >> match_noun
    
    return True


def build_network_expanded(noun, inverse_noun, all_adj_dicts, features, D):
    def set_noun(t, noun):
        all_nouns = [key for key in noun.keys()]
        index = int(t/0.2) % len(noun)
        
        return all_nouns[index]
      
        
    def set_adjective(adj_dict):
        for key, value in adj_dict.items():
            adj_row = spa.Transcode(value, output_vocab=vocab, label=key)
            wta = spa.WTAAssocMem(0.3, input_vocab=vocab, mapping=vocab.keys(), function=lambda x: x > 0, label=key+' CLEANED')
            adj_row * ~noun_attributes >> wta
            wta >> end
            
        return True
    
    vocab = spa.Vocabulary(D, max_similarity=0.1)
    vocab.populate(features)
    
    noun_vocab = spa.Vocabulary(D)
    noun_vocab.populate(";".join(noun.keys()))
    
    current_noun = spa.Transcode(lambda t: set_noun(t, noun), output_vocab=noun_vocab, label='NOUN')
    end = spa.State(vocab=vocab, label='A-N FEATURES')
    
    noun_attributes = spa.ThresholdingAssocMem(
        threshold=0.2,
        input_vocab=noun_vocab,
        output_vocab=vocab,
        mapping = noun,
        function=lambda x: x > 0,
        label='NOUN FEATURES'
        )
    
    current_noun >> noun_attributes
    
    for adj in all_adj_dicts:
        set_adjective(adj)
        
    match_noun = spa.WTAAssocMem(
        threshold=0.7,
        input_vocab=vocab,
        output_vocab=noun_vocab,
        mapping=inverse_noun,
        function=lambda x: x > 0.,
        label='NOUN'
        )
    
    end >> match_noun
    
    return True

D = 128

# Equations for our nouns
FISH     = '0.13*CARED_FOR + 0.51*VICIOUS + 0.00*FLUFFY + 0.63*SCALY + 0.51*LIVES_SEA + 0.19*LIVES_HOUSE + 0.19*LIVES_ZOO'
GOLDFISH = '0.44*CARED_FOR + 0.00*VICIOUS + 0.00*FLUFFY + 0.62*SCALY + 0.00*LIVES_SEA + 0.62*LIVES_HOUSE + 0.19*LIVES_ZOO'
CAT      = '0.57*CARED_FOR + 0.13*VICIOUS + 0.57*FLUFFY + 0.00*SCALY + 0.00*LIVES_SEA + 0.57*LIVES_HOUSE + 0.00*LIVES_ZOO'
DOG      = '0.67*CARED_FOR + 0.37*VICIOUS + 0.37*FLUFFY + 0.00*SCALY + 0.00*LIVES_SEA + 0.52*LIVES_HOUSE + 0.00*LIVES_ZOO'
SHARK    = '0.00*CARED_FOR + 0.57*VICIOUS + 0.00*FLUFFY + 0.57*SCALY + 0.57*LIVES_SEA + 0.00*LIVES_HOUSE + 0.11*LIVES_ZOO'
LION     = '0.19*CARED_FOR + 0.62*VICIOUS + 0.44*FLUFFY + 0.00*SCALY + 0.00*LIVES_SEA + 0.00*LIVES_HOUSE + 0.62*LIVES_ZOO'

animals = {'FISH': FISH, 'CAT': CAT, 'GOLDFISH': GOLDFISH,
        'DOG': DOG, 'SHARK': SHARK, 'LION': LION}
            
inverse_animals = {value: key for key, value in animals.items()}


# Handcoded adjective
PET_ADJ_C = '1.0*CARED_FOR*CARED_FOR + 1.0*CARED_FOR*VICIOUS +1.0*CARED_FOR*FLUFFY +1.0*CARED_FOR*SCALY + 1.0*CARED_FOR*LIVES_SEA + 1.0*CARED_FOR*LIVES_HOUSE + 1.0*CARED_FOR*LIVES_ZOO'
PET_ADJ_V = '1.0*VICIOUS*VICIOUS'
PET_ADJ_F = '0.2*FLUFFY*VICIOUS +1.0*FLUFFY*FLUFFY'
PET_ADJ_S = '1.0*SCALY*SCALY'
PET_ADJ_E = '0.0*LIVES_SEA*LIVES_SEA'
PET_ADJ_H = '1.0*LIVES_HOUSE*LIVES_SEA +1.0*LIVES_HOUSE*LIVES_HOUSE + 1.0*LIVES_HOUSE*LIVES_ZOO'
PET_ADJ_Z = '0.0*LIVES_ZOO*LIVES_ZOO'
  
  
          
pet = {'CARED_FOR': PET_ADJ_C, 'VICIOUS': PET_ADJ_V, 'FLUFFY': PET_ADJ_F, 
            'SCALY': PET_ADJ_S, 'LIVES_SEA':PET_ADJ_E, 'LIVES_HOUSE': PET_ADJ_H, 'LIVES_ZOO':PET_ADJ_Z}

features = get_features(animals)
adjectives = [pet] # See auto_build_adjective for options re: auto-building pet

with spa.Network(seed=5) as model:
    build_network_expanded(noun=animals, inverse_noun=inverse_animals, 
        all_adj_dicts=adjectives, features=features, D=D)

    
    
    
    
    

    
    
    
