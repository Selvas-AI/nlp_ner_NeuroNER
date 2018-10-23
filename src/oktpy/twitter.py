import jpype

from .util import load_okt, java_bool, attachThread


class KoreanToken(object):
    def __init__(self, kt_java):
        self.text = kt_java.getText()
        self.pos = kt_java.getPos().name()
        self.offset = kt_java.getOffset()
        self.length = kt_java.getLength()
        self.unknown = kt_java.isUnknown()
        self.stem = kt_java.getStem()

    def __str__(self):
        return self.pos

    def __repr__(self):
        return '{}({}: {}, {})'.format(self.text, self.pos, self.offset, self.length)


class KoreanPhrase(object):
    def __init__(self, kp_scala):
        self.offset = kp_scala.offset()
        self.text = kp_scala.text()
        self.length = kp_scala.length()
        self.pos = kp_scala.pos()

    def __str__(self):
        return self.text

    def __repr__(self):
        return '{}({}: {}, {})'.format(self.text, self.pos, self.offset, self.length)


class Twitter():
    def __init__(self):
        self._okt = load_okt()

    @attachThread
    def pos(self, phrase, norm=False, stem=False, join=False):
        if join == True:
            print("Twitter pos function : 'join' argument not used")

        tokens = self._okt.tokenize(phrase, java_bool(norm), java_bool(stem))
        return [KoreanToken(t) for t in tokens]

    @attachThread
    def morphs(self, phrase, norm=False, stem=False):
        return [t.text for t in self.pos(phrase, norm=norm, stem=stem)]

    @attachThread
    def nouns(self, phrase):
        tokens = self.pos(phrase)
        return [t.text for t in tokens if t.pos == 'Noun']

    @attachThread
    def phrases(self, phrases):
        filterSpam = False
        addHashtags = False
        phrs = self._okt.extractPhrases(phrases, java_bool(filterSpam), java_bool(addHashtags))
        return [KoreanPhrase(p) for p in phrs]

    @attachThread
    def splitSentence(self, phrases):
        sentences = self._okt.splitSentences(phrases)
        return [s for s in sentences]

    @attachThread
    def addNouns(self, nouns):
        self._okt.addNounsToDictionary(nouns)

    def __exit(self, exc_type, exc_value, traceback):
        jpype.shutdownJVM()


class TwitterMorphManager:
    class __TwitterMorphManager:
        def __init__(self):
            self.morph_analyzer = Twitter()

    Instance = None

    def __init__(self):
        if not TwitterMorphManager.Instance:
            TwitterMorphManager.Instance = TwitterMorphManager.__TwitterMorphManager()

    def __getattr__(self, item):
        return getattr(self.Instance, item)
