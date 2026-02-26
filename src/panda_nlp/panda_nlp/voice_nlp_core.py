import whisper
import spacy
from spacy.matcher import Matcher
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class SpatialDescriptionClause:
    verb: Optional[str] = None
    verb_lemma: Optional[str] = None
    color: Optional[str] = None
    shape: Optional[str] = None
    object: Optional[str] = None
    raw_text: str = ""

    def to_dict(self):
        return {
            "verb": self.verb,
            "verb_lemma": self.verb_lemma,
            "color": self.color,
            "shape": self.shape,
            "object": self.object,
            "raw_text": self.raw_text
        }


class VoiceControlledRoboticsNLP:

    def __init__(self, sample_rate=16000, duration=5):
        self.sample_rate = sample_rate
        self.duration = duration

        print("Loading Whisper base model (more accurate than tiny)...")
        self.whisper_model = whisper.load_model("base")

        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.colors = {
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'orange',
            'purple', 'pink', 'brown'
        }

        self.shapes = {
            'sphere', 'cube', 'cylinder', 'cone', 'pyramid', 'box', 'ball'
        }

        self.action_verbs = {
            'pick', 'grab', 'take', 'lift', 'get',
            'bring', 'carry', 'fetch', 'grasp', 'hold'
        }

        # Phonetic near-misses that Whisper commonly hallucinates
        # Maps mishearing -> correct color
        self.color_aliases = {
            'reed': 'red', 'rad': 'red', 'read': 'red', 'rid': 'red',
            'blu': 'blue', 'blew': 'blue',
            'grean': 'green', 'greene': 'green', 'grain': 'green',
        }

        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()

        print("Initialization complete!")

    def _setup_patterns(self):
        pattern1 = [
            {"POS": "VERB"},
            {"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "?"},
            {"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "?"},
            {"POS": "NOUN", "OP": "?"}
        ]
        self.matcher.add("ACTION_PATTERN", [pattern1])

    def record_audio(self) -> str:
        print(f"Recording for {self.duration} seconds... speak now!")

        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            device=6,  # pulse device
        )
        sd.wait()

        # Check if audio is too quiet (mic not picking up)
        amplitude = np.max(np.abs(recording))
        if amplitude < 0.001:
            print("WARNING: Very quiet audio detected. Check microphone.")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            audio_int16 = (recording * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        print("Recording complete!")
        return temp_file.name

    def speech_to_text(self, audio_path: str) -> str:
        print("Transcribing...")
        result = self.whisper_model.transcribe(
            audio_path,
            language="en",
            condition_on_previous_text=False,  # prevents hallucination loops
            no_speech_threshold=0.6,           # reject silence
            logprob_threshold=-1.0,            # reject low confidence
            temperature=0.0,                   # deterministic, less creative
        )

        # Reject if all segments have high no_speech probability
        segments = result.get("segments", [])
        if segments and all(seg.get("no_speech_prob", 0) > 0.5 for seg in segments):
            print("Transcription rejected: detected as silence/noise")
            return ""

        text = result["text"].strip()
        print(f"Transcription: '{text}'")
        return text

    def extract_sdc(self, text: str) -> SpatialDescriptionClause:
        doc = self.nlp(text.lower())
        sdc = SpatialDescriptionClause(raw_text=text)
        text_lower = text.lower()

        # Extract verb
        for token in doc:
            if token.lemma_ in self.action_verbs:
                sdc.verb = token.text
                sdc.verb_lemma = token.lemma_
                break
        if not sdc.verb:
            for token in doc:
                if token.pos_ == "VERB":
                    sdc.verb = token.text
                    sdc.verb_lemma = token.lemma_
                    break

        # Extract color — strict whole-word match only
        for color in self.colors:
            if re.search(r'\b' + color + r'\b', text_lower):
                sdc.color = color
                break

        # If no direct match, try phonetic aliases
        if not sdc.color:
            for word in text_lower.split():
                clean = re.sub(r'[^a-z]', '', word)
                if clean in self.color_aliases:
                    sdc.color = self.color_aliases[clean]
                    print(f"Color alias matched: '{clean}' -> '{sdc.color}'")
                    break

        # Extract shape
        for token in doc:
            if token.text in self.shapes or token.lemma_ in self.shapes:
                sdc.shape = token.text
                break

        # Extract object
        for token in doc:
            if token.pos_ == "NOUN" and token.dep_ in ["dobj", "pobj", "nsubj", "ROOT"]:
                if token.text not in self.colors and token.text not in self.shapes:
                    sdc.object = token.text
                    break
        if not sdc.object:
            for token in reversed(doc):
                if token.pos_ == "NOUN":
                    sdc.object = token.text
                    break

        return sdc

    def perform_nlp_tasks(self, text: str) -> Dict:
        doc = self.nlp(text)
        return {
            "tokens": [token.text for token in doc],
            "lemmas": [token.lemma_ for token in doc],
            "pos_tags": [(token.text, token.pos_) for token in doc],
            "dependencies": [(token.text, token.dep_, token.head.text) for token in doc],
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "noun_chunks": [chunk.text for chunk in doc.noun_chunks]
        }

    def process_voice_command(self, audio_path: Optional[str] = None) -> Tuple[str, Dict, SpatialDescriptionClause]:
        if audio_path is None:
            audio_path = self.record_audio()

        text = self.speech_to_text(audio_path)
        nlp_analysis = self.perform_nlp_tasks(text)
        sdc = self.extract_sdc(text)

        if audio_path.startswith(tempfile.gettempdir()):
            os.unlink(audio_path)

        return text, nlp_analysis, sdc

    def process_text_command(self, text: str) -> Tuple[Dict, SpatialDescriptionClause]:
        nlp_analysis = self.perform_nlp_tasks(text)
        sdc = self.extract_sdc(text)
        return nlp_analysis, sdc

    def display_results(self, text: str, nlp_analysis: Dict, sdc: SpatialDescriptionClause):
        print(f"\nTRANSCRIBED TEXT: {text}")
        print(f"\nSPATIAL DESCRIPTION CLAUSE:")
        print(f"   Action: {sdc.verb} ({sdc.verb_lemma})")
        print(f"   Color:  {sdc.color}")
        print(f"   Shape:  {sdc.shape}")
        print(f"   Object: {sdc.object}")
        print(f"\nJSON: {json.dumps(sdc.to_dict(), indent=2)}")


def main():
    system = VoiceControlledRoboticsNLP(duration=5)
    test_commands = [
        "pick up the red sphere",
        "take the blue ball",
        "grab the green sphere",
    ]
    for cmd in test_commands:
        print(f"\nProcessing: '{cmd}'")
        _, sdc = system.process_text_command(cmd)
        system.display_results(cmd, {}, sdc)


if __name__ == "__main__":
    main()
