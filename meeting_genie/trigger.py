INTERROGATIVES = {"what", "why", "when", "where", "who", "how"}
AUXILIARIES = {"is", "are", "do", "does", "can", "could", "would", "should"}
GREETINGS = {"hi", "hello", "hey", "anyway", "so", "okay", "one"}

def strip_greeting(sentence):
    words = sentence.strip().split()
    while words and words[0].strip(",.").lower() in GREETINGS:
        words = words[1:]
    return " ".join(words)

def score_sentence(sentence, cfg):
    words = sentence.strip().split()
    cleaned = strip_greeting(sentence)
    clean_words = cleaned.split()
    score = 0.0

    if sentence.strip().endswith("?"):
        score += cfg["trigger"]["signals"]["ends_with_question_mark"]

    if any(w.lower() in INTERROGATIVES for w in clean_words[:5]):
        score += cfg["trigger"]["signals"]["starts_with_interrogative"]

    elif any(w.lower() in AUXILIARIES for w in clean_words[:5]):
        score += cfg["trigger"]["signals"]["starts_with_auxiliary"]

    if 3 <= len(words) <= 40:
        score += cfg["trigger"]["signals"]["length_in_range"]

    return score

class Trigger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pending = []

    def feed(self, utterance):
        if utterance.speaker == "them":
            score = score_sentence(utterance.text, self.cfg)
            if score >= self.cfg["trigger"]["threshold"]:
                self.pending.append(utterance.text)

        elif utterance.speaker == "me":
            self.pending = []

        if self.pending and utterance.silence_after_ms >= self.cfg["trigger"]["silence_ms"]:
            fired = self.pending
            self.pending = []
            return fired

        return None