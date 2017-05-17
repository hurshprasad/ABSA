import configparser
from read import constants as cs

def get_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config


def get(config, section, option):
    value = config.get(section, option)
    return value


def constant(f):
    def fset(self, value):
        raise TypeError

    def fget(self):
        return f(self)

    return property(fget, fset)


class _Const(object):

    config = None

    def parse_argument(self, parser):

        parser.add_argument("-dev_set")
        parser.add_argument("-conf_dir")
        parser.add_argument("-conf_name")
        parser.add_argument("-data_dir")
        parser.add_argument("-out_dir")
        parser.add_argument("-w2v_file")
        parser.add_argument("-learning_rate")
        parser.add_argument("-hidden_size")
        parser.add_argument("-dropout")
        parser.add_argument("-batch_size")
        parser.add_argument("-project_size")
        parser.add_argument("-aspects")
        parser.add_argument("-name")
        parser.add_argument("-attention")
        parser.add_argument("-model_path")
        parser.add_argument("-save_model")
        parser.add_argument("-train")
        parser.add_argument("-model")  # GRU, BasicLSTM, BiRNN
        parser.add_argument("-epoch")
        parser.add_argument("-reload_train")
        parser.add_argument("-map")

        args = parser.parse_args()

        if args.conf_dir is None:
            args.conf_dir = 'conf/'

        if args.conf_name is None:
            args.conf_name = 'semeval_base.conf'

        self.config = get_config(args.conf_dir + args.conf_name)

        if args.model is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_MODEL, args.model)

        if args.learning_rate is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_LEARNING_RATE, args.learning_rate)

        if args.hidden_size is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_HIDDEN_SIZE, args.hidden_size)

        if args.dropout is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_DROPOUT, args.dropout)

        if args.batch_size is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_BATCH_SIZE, args.batch_size)

        if args.project_size is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_PROJECT_SIZE, args.project_size)

        if args.aspects is not None:
            self.config.set(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_ASPECTS, args.aspects)

        if args.dev_set is not None:
            self.config.set(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_DEV_SET, args.dev_set)

        if args.attention is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_ATTENTION, args.attention)

        if args.model_path is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_MODEL_PATH, args.model_path)

        if args.save_model is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_SAVE_MODEL, args.save_model)

        if args.out_dir is not None:
            self.config.set(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_OUT_DIR, args.out_dir)

        if args.train is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_TRAIN, args.train)

        if args.reload_train is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_RELOAD_TRAIN, args.reload_train)

        if args.epoch is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_MAX_EPOCH, args.epoch)

        if args.map is not None:
            self.config.set(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_HEATMAP, args.map)

    @constant
    def DATA_DIR(self):
        return self.config.get(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_DATA_DIR)

    @constant
    def TEST_DIR(self):
        return self.config.get(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_TEST_DIR)

    @constant
    def LEARNING_RATES(self):
        learning_rates = self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_LEARNING_RATE)
        return [float(l) for l in learning_rates.split(",")]

    @constant
    def HIDDEN_SIZE(self):
        hidden_size = self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_HIDDEN_SIZE)
        return [int(s) for s in hidden_size.split(",")]

    @constant
    def DROPOUTS(self):
        dropout = self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_DROPOUT)
        return [float(s) for s in dropout.split(",")]

    @constant
    def BATCH_SIZES(self):
        batch_size = self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_BATCH_SIZE)
        return [int(s) for s in batch_size.split(",")]

    @constant
    def PROJECT_SIZE(self):
        project_size = self.config.getint(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_PROJECT_SIZE)
        return [int(s) for s in project_size.split(",")]

    @constant
    def ASPECTS(self):
        return self.config.get(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_ASPECTS).split(",")

    @constant
    def DEV_SET(self):
        return self.config.get(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_DEV_SET).replace(" ", "").split(",")

    @constant
    def ATTENTION(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_ATTENTION)

    @constant
    def MODEL_PATH(self):
        return None

    @constant
    def SAVE_MODEL(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_LOAD_MODEL)

    @constant
    def MODEL(self):
        return self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_MODEL)

    @constant
    def SAVE_DATA(self):
        return self.config.getboolean(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_SAVE_DATA)

    @constant
    def DATA_FILE(self):
        return self.config.get(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_SAVE_DATA_FILE)

    @constant
    def MOCK(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_MOCK)

    @constant
    def MOCK_DEV(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_MOCK_DEV)

    @constant
    def TRAIN_FILES(self):
        return self.config.get(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_TRAIN_SET).replace(" ", "").split(",")

    @constant
    def DEV_FILES(self):
        return self.config.get(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_DEV_SET).replace(" ", "").split(",")

    @constant
    def TEST_FILES(self):
        return self.config.get(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_TEST_SET).replace(" ", "").split(",")

    @constant
    def UNIGRAM(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_UNIGRAM)

    @constant
    def BIGRAM(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_BIGRAM)

    @constant
    def TRIGRAM(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_TRIGRAM)

    @constant
    def LEMMA(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_LEMMA)

    @constant
    def THRESHOLD(self):
        return self.config.getfloat(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_THRESHOLD)

    @constant
    def W2VEC_FILE(self):
        return self.config.get(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_W2V_FILE)

    @constant
    def OUT_DIR(self):
        return self.config.get(cs.CONFIG_SECTION_GENERAL, cs.CONFIG_OPTION_OUT_DIR)

    @constant
    def REMOVE_STOP_WORDS(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_REMOVE_STOP_WORDS)

    @constant
    def BUCKET(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_INCLUDE_BUCKETS)

    @constant
    def TARGET_MASK(self):
        return "TARGET_LOC"

    @constant
    def SLOT1_MODEL_PATH(self):
        return self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_SAVE_SLOT1)

    @constant
    def SLOT3_MODEL_PATH(self):
        return self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_SAVE_SLOT3)

    @constant
    def SLOT3_TARGET_MODEL_PATH(self):
        return self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_SAVE_SLOT3_TARGET)

    @constant
    def MULTITASK_CHECKPOINT_PATH(self):
        return self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_SAVE_MULTITASK)

    @constant
    def CNN_MODEL_PATH(self):
        return self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_SAVE_SLOT3_CNN)

    @constant
    def MULTITASK_CHECKPOINT_PATH_HURSH(self):
        return self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_SAVE_MULTITASK_HURSH)

    @constant
    def SENTENCE_PATH(self):
        return self.config.get(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_SENTENCE)

    @constant
    def TRAIN(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_TRAIN)

    @constant
    def RELOAD_TRAIN(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_RELOAD_TRAIN)

    @constant
    def HEATMAP(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_HEATMAP)

    @constant
    def MAX_EPOCH(self):
        return self.config.getint(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_MAX_EPOCH)

    @constant
    def UNIGRAM(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_UNIGRAM)

    @constant
    def BIGRAM(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_BIGRAM)

    @constant
    def TRIGRAM(self):
        return self.config.getboolean(cs.CONFIG_SECTION_MODEL, cs.CONFIG_OPTION_TRIGRAM)



CONST = _Const()
