"""
Clone Hero Content Manager - Lyrics Generator Service

Generates procedural lyrics timed to beats and inserts them into the
[Events] section of a Clone Hero notes.chart file.

Clone Hero lyrics format (inside [Events]):
    {tick} = E "phrase_start"
    {tick} = E "lyric {word}"
    ...
    {tick} = E "phrase_end"
    {tick} = E "phrase_start"
    {tick} = E "lyric {word}"
    ...

Lyric words are displayed in the in-game lyric track.  Each phrase is a
line of text that scrolls across the screen.  phrase_start / phrase_end
delimit the boundaries of each displayed line.

This module provides:
    - Themed word banks for procedural lyric generation
    - Syllable-aware word selection for natural rhythm
    - Phrase construction timed to musical sections and beats
    - Integration with the chart generation pipeline
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Resolution constant (must match song_generator.py)
# ---------------------------------------------------------------------------
RESOLUTION = 192  # ticks per quarter note


# ---------------------------------------------------------------------------
# Word banks — organised by theme / mood / part-of-speech
# ---------------------------------------------------------------------------

THEMES = {
    "rock": {
        "nouns": [
            "fire",
            "road",
            "night",
            "thunder",
            "storm",
            "steel",
            "heart",
            "flame",
            "rebel",
            "highway",
            "shadow",
            "wire",
            "riot",
            "anthem",
            "edge",
            "engine",
            "guitar",
            "stage",
            "crowd",
            "smoke",
            "lightning",
            "chain",
            "dream",
            "battle",
        ],
        "verbs": [
            "burn",
            "ride",
            "fight",
            "scream",
            "break",
            "shake",
            "rock",
            "roll",
            "blaze",
            "run",
            "crash",
            "strike",
            "roar",
            "shatter",
            "ignite",
            "rise",
            "fall",
            "howl",
        ],
        "adjectives": [
            "wild",
            "loud",
            "fast",
            "heavy",
            "dark",
            "bright",
            "fierce",
            "raw",
            "free",
            "electric",
            "savage",
            "bold",
            "relentless",
            "burning",
            "raging",
            "untamed",
        ],
        "prepositions": [
            "through",
            "into",
            "over",
            "under",
            "beyond",
            "across",
            "beneath",
            "within",
            "against",
            "among",
        ],
        "exclamations": [
            "yeah",
            "oh",
            "whoa",
            "hey",
            "now",
            "come on",
            "alright",
            "let's go",
        ],
    },
    "pop": {
        "nouns": [
            "love",
            "heart",
            "star",
            "light",
            "sky",
            "dream",
            "kiss",
            "dance",
            "moon",
            "summer",
            "smile",
            "rain",
            "music",
            "baby",
            "world",
            "color",
            "rhythm",
            "feeling",
            "melody",
            "magic",
            "sparkle",
            "glow",
            "moment",
            "night",
        ],
        "verbs": [
            "shine",
            "dance",
            "feel",
            "sing",
            "love",
            "glow",
            "fly",
            "spin",
            "hold",
            "touch",
            "move",
            "sway",
            "believe",
            "breathe",
            "whisper",
            "sparkle",
            "dream",
            "float",
        ],
        "adjectives": [
            "beautiful",
            "sweet",
            "golden",
            "bright",
            "young",
            "endless",
            "perfect",
            "shining",
            "lovely",
            "warm",
            "crystal",
            "electric",
            "wonderful",
            "infinite",
        ],
        "prepositions": [
            "with",
            "into",
            "through",
            "above",
            "around",
            "beside",
            "between",
            "toward",
            "across",
            "beneath",
        ],
        "exclamations": [
            "ooh",
            "ah",
            "baby",
            "yeah",
            "oh",
            "la la",
            "na na",
            "hey",
        ],
    },
    "electronic": {
        "nouns": [
            "signal",
            "pulse",
            "wave",
            "circuit",
            "grid",
            "code",
            "system",
            "data",
            "void",
            "neon",
            "laser",
            "synth",
            "byte",
            "frequency",
            "matrix",
            "digital",
            "echo",
            "current",
            "voltage",
            "spectrum",
            "pixel",
            "phase",
        ],
        "verbs": [
            "transmit",
            "pulse",
            "sync",
            "scan",
            "decode",
            "connect",
            "override",
            "reboot",
            "upload",
            "process",
            "surge",
            "flow",
            "compute",
            "resonate",
            "oscillate",
            "initialize",
        ],
        "adjectives": [
            "digital",
            "electric",
            "cyber",
            "virtual",
            "neon",
            "binary",
            "synthetic",
            "quantum",
            "chromatic",
            "sonic",
            "automated",
            "magnetic",
            "luminous",
            "modular",
        ],
        "prepositions": [
            "through",
            "into",
            "beyond",
            "within",
            "across",
            "inside",
            "outside",
            "between",
            "above",
            "below",
        ],
        "exclamations": [
            "go",
            "now",
            "activate",
            "engage",
            "run",
            "loading",
            "online",
        ],
    },
    "chill": {
        "nouns": [
            "breeze",
            "ocean",
            "wave",
            "sunset",
            "cloud",
            "shore",
            "river",
            "garden",
            "mountain",
            "meadow",
            "horizon",
            "silence",
            "peace",
            "dawn",
            "tide",
            "forest",
            "sky",
            "island",
            "valley",
            "whisper",
            "feather",
            "sand",
        ],
        "verbs": [
            "drift",
            "flow",
            "breathe",
            "wander",
            "rest",
            "float",
            "settle",
            "bloom",
            "fade",
            "glide",
            "linger",
            "ease",
            "unwind",
            "melt",
            "ripple",
            "soothe",
        ],
        "adjectives": [
            "gentle",
            "soft",
            "calm",
            "quiet",
            "warm",
            "slow",
            "still",
            "deep",
            "mellow",
            "peaceful",
            "serene",
            "tender",
            "smooth",
            "hazy",
            "tranquil",
        ],
        "prepositions": [
            "along",
            "through",
            "over",
            "beside",
            "beneath",
            "among",
            "upon",
            "toward",
            "within",
            "around",
        ],
        "exclamations": [
            "mmm",
            "ooh",
            "ah",
            "shhh",
            "hmm",
            "oh",
            "la",
        ],
    },
    "metal": {
        "nouns": [
            "darkness",
            "fury",
            "chaos",
            "demon",
            "abyss",
            "sword",
            "throne",
            "blood",
            "skull",
            "dragon",
            "inferno",
            "doom",
            "wrath",
            "vengeance",
            "oblivion",
            "legion",
            "iron",
            "tomb",
            "eclipse",
            "serpent",
            "phantom",
            "carnage",
        ],
        "verbs": [
            "destroy",
            "conquer",
            "crush",
            "shatter",
            "unleash",
            "devour",
            "annihilate",
            "dominate",
            "ravage",
            "summon",
            "reign",
            "forge",
            "descend",
            "consume",
            "obliterate",
        ],
        "adjectives": [
            "eternal",
            "brutal",
            "savage",
            "unholy",
            "merciless",
            "relentless",
            "immortal",
            "ruthless",
            "ancient",
            "forsaken",
            "infernal",
            "devastating",
            "colossal",
            "malevolent",
        ],
        "prepositions": [
            "through",
            "into",
            "beyond",
            "beneath",
            "from",
            "upon",
            "within",
            "across",
            "against",
            "among",
        ],
        "exclamations": [
            "rise",
            "fall",
            "war",
            "death",
            "now",
            "behold",
            "arise",
        ],
    },
    "default": {
        "nouns": [
            "time",
            "world",
            "life",
            "day",
            "night",
            "way",
            "mind",
            "soul",
            "voice",
            "story",
            "moment",
            "place",
            "feeling",
            "change",
            "truth",
            "dream",
            "sound",
            "path",
            "journey",
            "wonder",
            "memory",
            "spirit",
            "song",
        ],
        "verbs": [
            "find",
            "know",
            "feel",
            "see",
            "move",
            "run",
            "take",
            "give",
            "hold",
            "stand",
            "walk",
            "turn",
            "reach",
            "begin",
            "carry",
            "follow",
            "search",
            "wander",
        ],
        "adjectives": [
            "new",
            "old",
            "true",
            "real",
            "bright",
            "long",
            "great",
            "high",
            "deep",
            "strong",
            "clear",
            "open",
            "far",
            "wide",
            "free",
            "endless",
            "rising",
        ],
        "prepositions": [
            "through",
            "into",
            "over",
            "with",
            "from",
            "beyond",
            "around",
            "along",
            "beside",
            "within",
        ],
        "exclamations": [
            "oh",
            "yeah",
            "hey",
            "come on",
            "now",
            "whoa",
            "alright",
        ],
    },
}


# ---------------------------------------------------------------------------
# Phrase templates — each is a list of part-of-speech tags
# ---------------------------------------------------------------------------
# Tags: N=noun, V=verb, A=adjective, P=preposition, E=exclamation,
#        D=determiner, PR=pronoun
DETERMINERS = ["the", "a", "this", "that", "my", "your", "our", "every", "no"]
PRONOUNS = ["I", "we", "you", "they", "it", "she", "he"]
CONJUNCTIONS = ["and", "but", "or", "so", "yet", "while", "as"]

PHRASE_TEMPLATES = [
    # Short punchy lines
    ["E"],
    ["V", "P", "D", "N"],
    ["A", "N"],
    ["V", "D", "N"],
    ["PR", "V"],
    ["E", "E"],
    # Medium lines
    ["PR", "V", "P", "D", "N"],
    ["D", "A", "N", "V"],
    ["V", "D", "A", "N"],
    ["PR", "V", "D", "N"],
    ["P", "D", "N", "PR", "V"],
    ["A", "N", "V", "P", "D", "N"],
    # Longer lines
    ["PR", "V", "P", "D", "A", "N"],
    ["D", "N", "V", "P", "D", "N"],
    ["E", "PR", "V", "D", "A", "N"],
    ["V", "D", "N", "C", "V", "D", "N"],
    ["PR", "V", "A", "C", "A"],
    # Call/response & repetition
    ["E", "V", "D", "N", "E"],
    ["N", "N", "N"],
    ["V", "V", "V"],
]


# ---------------------------------------------------------------------------
# Chorus templates — repeated structures that feel like a hook
# ---------------------------------------------------------------------------
CHORUS_TEMPLATES = [
    ["E", "PR", "V", "D", "N"],
    ["V", "D", "A", "N"],
    ["PR", "V", "E"],
    ["E", "E", "V"],
    ["D", "N", "V", "P", "PR"],
    ["A", "N", "A", "N"],
]


# ---------------------------------------------------------------------------
# Section type detection
# ---------------------------------------------------------------------------
VERSE_LABELS = {"verse", "verse 1", "verse 2", "verse 3", "pre-chorus", "pre-chorus 2"}
CHORUS_LABELS = {"chorus", "chorus 1", "chorus 2", "final chorus", "main"}
BRIDGE_LABELS = {"bridge", "breakdown", "interlude"}
INTRO_LABELS = {"intro"}
OUTRO_LABELS = {"outro"}
INSTRUMENTAL_LABELS = {"solo", "guitar solo", "instrumental"}


def _classify_section(label: str) -> str:
    """Classify a section label into a broad category."""
    lower = label.lower().strip()
    if lower in VERSE_LABELS:
        return "verse"
    if lower in CHORUS_LABELS:
        return "chorus"
    if lower in BRIDGE_LABELS:
        return "bridge"
    if lower in INTRO_LABELS:
        return "intro"
    if lower in OUTRO_LABELS:
        return "outro"
    if lower in INSTRUMENTAL_LABELS:
        return "instrumental"
    # Default: treat numbered sections as verses
    if "verse" in lower:
        return "verse"
    if "chorus" in lower:
        return "chorus"
    return "verse"


# ---------------------------------------------------------------------------
# Syllable estimation
# ---------------------------------------------------------------------------
def _estimate_syllables(word: str) -> int:
    """
    Rough syllable count for an English word.

    This is a heuristic — not perfect, but good enough for rhythm alignment.
    """
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1

    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Adjust for silent e
    if word.endswith("e") and count > 1:
        count -= 1
    # Adjust for -le endings
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1

    return max(1, count)


# ---------------------------------------------------------------------------
# Theme selection
# ---------------------------------------------------------------------------
def _select_theme(genre: str, song_name: str) -> str:
    """Pick the best theme based on genre and song name."""
    genre_lower = (genre or "").lower()
    name_lower = (song_name or "").lower()

    # Direct genre matches
    genre_map = {
        "rock": "rock",
        "metal": "metal",
        "hard rock": "rock",
        "punk": "rock",
        "alternative": "rock",
        "grunge": "rock",
        "pop": "pop",
        "dance": "pop",
        "r&b": "pop",
        "electronic": "electronic",
        "edm": "electronic",
        "techno": "electronic",
        "house": "electronic",
        "trance": "electronic",
        "dubstep": "electronic",
        "drum and bass": "electronic",
        "dnb": "electronic",
        "ambient": "chill",
        "chill": "chill",
        "lofi": "chill",
        "lo-fi": "chill",
        "jazz": "chill",
        "classical": "chill",
        "folk": "chill",
        "acoustic": "chill",
        "death metal": "metal",
        "black metal": "metal",
        "thrash": "metal",
        "doom": "metal",
        "heavy metal": "metal",
        "metalcore": "metal",
    }

    for keyword, theme in genre_map.items():
        if keyword in genre_lower or keyword in name_lower:
            return theme

    return "default"


# ---------------------------------------------------------------------------
# Phrase generation
# ---------------------------------------------------------------------------
def _generate_word(
    tag: str, theme_words: Dict[str, List[str]], rng: random.Random
) -> str:
    """Generate a single word based on its part-of-speech tag."""
    if tag == "N":
        return rng.choice(theme_words["nouns"])
    elif tag == "V":
        return rng.choice(theme_words["verbs"])
    elif tag == "A":
        return rng.choice(theme_words["adjectives"])
    elif tag == "P":
        return rng.choice(theme_words["prepositions"])
    elif tag == "E":
        return rng.choice(theme_words["exclamations"])
    elif tag == "D":
        return rng.choice(DETERMINERS)
    elif tag == "PR":
        return rng.choice(PRONOUNS)
    elif tag == "C":
        return rng.choice(CONJUNCTIONS)
    else:
        return rng.choice(theme_words["nouns"])


def _generate_phrase(
    template: List[str],
    theme_words: Dict[str, List[str]],
    rng: random.Random,
    avoid_words: Optional[set] = None,  # type: ignore[type-arg]
) -> List[str]:
    """
    Generate a phrase (list of words) from a template.

    Tries to avoid recently used words for variety.
    """
    if avoid_words is None:
        avoid_words = set()

    words = []
    for tag in template:
        # Try up to 5 times to get a non-repeated word
        for attempt in range(5):
            w = _generate_word(tag, theme_words, rng)
            if w.lower() not in avoid_words or attempt == 4:
                words.append(w)
                avoid_words.add(w.lower())
                break

    return words


def _generate_chorus_phrase(
    theme_words: Dict[str, List[str]],
    rng: random.Random,
    chorus_cache: List[List[str]],
    phrase_index: int,
) -> List[str]:
    """
    Generate or recall a chorus phrase.

    Choruses repeat the same lyrics each time they appear, so we cache
    the generated phrases and replay them on subsequent choruses.
    """
    if phrase_index < len(chorus_cache):
        return chorus_cache[phrase_index]

    template = rng.choice(CHORUS_TEMPLATES)
    phrase = _generate_phrase(template, theme_words, rng)
    chorus_cache.append(phrase)
    return phrase


# ---------------------------------------------------------------------------
# Tick conversion (mirrors song_generator.py)
# ---------------------------------------------------------------------------
def _seconds_to_ticks(time_s: float, tempo: float, resolution: int = RESOLUTION) -> int:
    """Convert a time in seconds to chart ticks."""
    beats = time_s * (tempo / 60.0)
    return int(round(beats * resolution))


# ---------------------------------------------------------------------------
# Main lyric generation
# ---------------------------------------------------------------------------
def generate_lyrics(
    tempo: float,
    beat_times: List[float],
    onset_times: List[float],
    onset_strengths: List[float],
    duration: float,
    segments: List[Dict[str, Any]],
    song_name: str = "",
    genre: str = "",
    seed: Optional[int] = None,
) -> List[str]:
    """
    Generate lyric events for a Clone Hero chart.

    Returns a list of event strings ready to be inserted into the [Events]
    section of a notes.chart file.  Each string is in the format:

        {tick} = E "lyric {word}"
        {tick} = E "phrase_start"
        {tick} = E "phrase_end"

    The lyrics are timed to beats and structured around the detected
    song segments (verse, chorus, bridge, etc.).

    Parameters
    ----------
    tempo : float
        Song tempo in BPM.
    beat_times : list of float
        Times (in seconds) of each detected beat.
    onset_times : list of float
        Times (in seconds) of each detected onset.
    onset_strengths : list of float
        Strength of each onset (0.0–1.0+).
    duration : float
        Total song duration in seconds.
    segments : list of dict
        Detected song segments with 'time', 'label', and 'energy' keys.
    song_name : str
        Name of the song (used for theme selection and seeding).
    genre : str
        Genre hint for theme selection.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of str
        Event lines (without leading whitespace) for the [Events] section.
    """
    rng = random.Random(seed or hash(song_name + genre))

    theme_name = _select_theme(genre, song_name)
    theme_words = THEMES.get(theme_name, THEMES["default"])

    if len(beat_times) < 4:
        logger.warning("⚠️ Too few beats for lyrics generation ({})", len(beat_times))
        return []

    events: List[str] = []

    # Build a beat index for quick lookup
    beat_set = set()
    for bt in beat_times:
        beat_set.add(round(bt, 3))

    # Build segment ranges: [(start_time, end_time, label), ...]
    seg_ranges: List[Tuple[float, float, str]] = []
    for i, seg in enumerate(segments):
        start = seg["time"]
        end = segments[i + 1]["time"] if i + 1 < len(segments) else duration
        seg_ranges.append((start, end, seg["label"]))

    # Chorus cache for repeating lyrics
    chorus_cache: List[List[str]] = []

    # Track recently used words to promote variety
    recent_words: set = set()  # type: ignore[type-arg]

    # Process each segment
    for seg_start, seg_end, seg_label in seg_ranges:
        section_type = _classify_section(seg_label)
        seg_duration = seg_end - seg_start

        # Skip very short segments
        if seg_duration < 3.0:
            continue

        # Instrumental / intro / outro: fewer or no lyrics
        if section_type == "instrumental":
            continue

        if section_type == "intro":
            # Maybe just one exclamation line in the intro
            if seg_duration > 6.0 and rng.random() < 0.6:
                mid_time = seg_start + seg_duration * 0.5
                tick = _seconds_to_ticks(mid_time, tempo)
                excl = rng.choice(theme_words["exclamations"])
                events.append(f'{tick} = E "phrase_start"')
                events.append(f'{tick} = E "lyric {excl}"')
                end_tick = _seconds_to_ticks(mid_time + 2.0, tempo)
                events.append(f'{end_tick} = E "phrase_end"')
            continue

        if section_type == "outro":
            # Possibly repeat last chorus line or a short phrase
            if seg_duration > 6.0 and rng.random() < 0.5:
                mid_time = seg_start + seg_duration * 0.3
                tick = _seconds_to_ticks(mid_time, tempo)
                if chorus_cache:
                    phrase = chorus_cache[0]
                else:
                    phrase = _generate_phrase(
                        rng.choice(PHRASE_TEMPLATES[:5]),
                        theme_words,
                        rng,
                        recent_words,
                    )
                events.append(f'{tick} = E "phrase_start"')
                for wi, word in enumerate(phrase):
                    word_tick = tick + wi * (RESOLUTION // 2)
                    events.append(f'{word_tick} = E "lyric {word}"')
                end_tick = tick + len(phrase) * (RESOLUTION // 2) + RESOLUTION
                events.append(f'{end_tick} = E "phrase_end"')
            continue

        # ── Verse / Chorus / Bridge ──
        # Decide how many phrases to put in this segment
        # Roughly 1 phrase per 4 beats, with a minimum of 2
        beats_in_segment = [bt for bt in beat_times if seg_start <= bt < seg_end]

        if len(beats_in_segment) < 2:
            continue

        # Phrases per segment: depends on segment length
        phrases_per_segment = max(2, min(8, len(beats_in_segment) // 4))

        # Distribute phrases evenly across the segment's beats
        beat_stride = max(1, len(beats_in_segment) // phrases_per_segment)

        chorus_phrase_idx = 0

        for p_idx in range(phrases_per_segment):
            beat_index = p_idx * beat_stride
            if beat_index >= len(beats_in_segment):
                break

            phrase_start_time = beats_in_segment[beat_index]

            # Make sure we don't go past the segment
            if phrase_start_time >= seg_end - 1.0:
                break

            # Generate the phrase words
            if section_type == "chorus":
                phrase = _generate_chorus_phrase(
                    theme_words, rng, chorus_cache, chorus_phrase_idx
                )
                chorus_phrase_idx += 1
            elif section_type == "bridge":
                # Bridge: use slightly different templates
                template = rng.choice(PHRASE_TEMPLATES[6:])
                phrase = _generate_phrase(template, theme_words, rng, recent_words)
            else:
                # Verse: standard templates
                template = rng.choice(PHRASE_TEMPLATES)
                phrase = _generate_phrase(template, theme_words, rng, recent_words)

            if not phrase:
                continue

            # Time each word to subsequent beats
            start_tick = _seconds_to_ticks(phrase_start_time, tempo)

            events.append(f'{start_tick} = E "phrase_start"')

            # Spread words across beats (or half-beats for short phrases)
            word_spacing_ticks = RESOLUTION  # 1 beat per word by default
            if len(phrase) > 4:
                word_spacing_ticks = RESOLUTION // 2  # half-beat for longer phrases

            for wi, word in enumerate(phrase):
                word_tick = start_tick + wi * word_spacing_ticks
                # Hyphenate multi-syllable words for display
                syllables = _estimate_syllables(word)
                if syllables > 2 and len(word) > 5:
                    # Split into two display parts with hyphen
                    mid = len(word) // 2
                    events.append(f'{word_tick} = E "lyric {word[:mid]}-"')
                    events.append(
                        f'{word_tick + word_spacing_ticks // 2} = E "lyric {word[mid:]}"'
                    )
                else:
                    events.append(f'{word_tick} = E "lyric {word}"')

            # Phrase end: after the last word plus a small gap
            phrase_end_tick = (
                start_tick + len(phrase) * word_spacing_ticks + RESOLUTION // 2
            )
            events.append(f'{phrase_end_tick} = E "phrase_end"')

        # Every few verses, clear recent words to avoid running out
        if len(recent_words) > 30:
            # Keep only the most thematic words
            recent_words.clear()

    logger.info(
        "🎤 Generated {} lyric events across {} segments (theme: {})",
        len(events),
        len(seg_ranges),
        theme_name,
    )

    return events


def generate_lyrics_for_chart(
    tempo: float,
    beat_times: List[float],
    onset_times: List[float],
    onset_strengths: List[float],
    duration: float,
    segments: List[Dict[str, Any]],
    song_name: str = "",
    genre: str = "",
    seed: Optional[int] = None,
) -> List[str]:
    """
    Convenience wrapper: generates lyric events formatted as chart lines.

    Returns lines ready to be appended inside the [Events] { ... } block,
    each prefixed with two spaces for consistent chart formatting.
    """
    raw_events = generate_lyrics(
        tempo=tempo,
        beat_times=beat_times,
        onset_times=onset_times,
        onset_strengths=onset_strengths,
        duration=duration,
        segments=segments,
        song_name=song_name,
        genre=genre,
        seed=seed,
    )

    return [f"  {line}" for line in raw_events]


# ---------------------------------------------------------------------------
# Utility: extract plain-text lyrics from generated events
# ---------------------------------------------------------------------------
def extract_plain_lyrics(events: List[str]) -> str:
    """
    Extract plain-text lyrics from chart event lines.

    Useful for displaying lyrics in the web UI or saving to a lyrics file.
    Returns a multi-line string with one phrase per line.
    """
    lines: List[str] = []
    current_phrase: List[str] = []

    for event in events:
        stripped = event.strip()

        if '"phrase_start"' in stripped:
            current_phrase = []
        elif '"phrase_end"' in stripped:
            if current_phrase:
                # Join words, handling hyphenated syllables
                line = ""
                for word in current_phrase:
                    if word.endswith("-"):
                        line += word[:-1]
                    elif line and not line.endswith("-"):
                        line += " " + word
                    else:
                        line += word
                lines.append(line)
                current_phrase = []
        elif '"lyric ' in stripped:
            # Extract the word from: {tick} = E "lyric {word}"
            try:
                lyric_part = stripped.split('"lyric ', 1)[1]
                word = lyric_part.rstrip('"').strip()
                if word:
                    current_phrase.append(word)
            except (IndexError, ValueError):
                pass

    return "\n".join(lines)
