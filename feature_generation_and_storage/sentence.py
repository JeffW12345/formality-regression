class Sentence:
    def __init__(self, sentence_id=None, formality_score_from_raters=None, informativeness=None,
                 implicature_score=None, length_in_words=None, length_in_characters=None,
                 f_score=None, i_score=None, lexical_density=None, sentence_content=None):
        self.sentence_id = sentence_id
        self.formality_score_from_raters = formality_score_from_raters
        self.informativeness = informativeness
        self.implicature_score = implicature_score
        self.length_in_words = length_in_words
        self.length_in_characters = length_in_characters
        self.f_score = f_score
        self.i_score = i_score
        self.lexical_density = lexical_density
        self.sentence_content = sentence_content

        self.total_syllables_stop_words_excluded = None
        self.total_syllables_per_word_stop_words_excluded = None

        self.total_syllables_stop_words_included = None
        self.total_syllables_per_word_stop_words_included = None

        self.number_of_nouns = None
        self.number_of_adjectives = None
        self.number_of_adverbs = None
        self.number_of_verbs = None
        self.number_of_pronouns = None

        self.number_of_stop_words = None
        self.proportion_of_stop_words = None

        self.number_of_spelling_and_grammatical_errors = None
        self.spelling_and_grammar_errors_per_character = None

        self.positive_vader_score = None
        self.negative_vader_score = None
        self.neutral_vader_score = None
        self.compound_vader_score = None


