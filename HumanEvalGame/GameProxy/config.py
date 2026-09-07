from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CONDITION_NO = "no_thinking"
CONDITION_PRE = "pre_thinking"
CONDITION_POST = "post_thinking"
CONDITIONS = (CONDITION_NO, CONDITION_PRE, CONDITION_POST)

PRE_OPEN = "<think>"
PRE_CLOSE = "</think>"
POST_OPEN = "<post-thinking>"
POST_CLOSE = "</post-thinking>"

GEMMA_START = "<start_of_turn>"
GEMMA_END = "<end_of_turn>"
GEMMA_EOS = "<eos>"

REACT_ANCHOR = "- React to the player's words and intentions."
GREETING_ANCHOR = "Your first response should be a greeting to the player."

INSTRUCTION_LINES = {
    CONDITION_NO: " ",
    CONDITION_PRE: (
        " - Before each reply, in <think>...</think>, briefly plan your intent "
        "and how you'll respond, in character. "
    ),
    CONDITION_POST: (
        " - After each reply, in <post-thinking>...</post-thinking>, briefly "
        "reflect in your own voice on why you responded as you did and what "
        "matters to remember going forward. Stay in your own perspective; "
        "never narrate the player's actions or feelings. "
    ),
}


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw) if raw else default


def _cell_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


ENABLED_CONDITIONS = _csv_env(
    "POSTTHINK_ENABLED_CONDITIONS",
    "no_thinking,post_thinking",
)
TRACE_WINDOW_N = int(os.getenv("POSTTHINK_TRACE_WINDOW_N", "3"))

DATABASE_PATH = _path_env(
    "POSTTHINK_DB_PATH",
    REPO_ROOT / "GameProxy" / "game_sessions.sqlite3",
)
TURN_LOG_JSONL_PATH = _path_env(
    "POSTTHINK_TURN_LOG_JSONL",
    REPO_ROOT / "GameProxy" / "turn_logs.jsonl",
)

LLAMA_ENDPOINTS = {
    CONDITION_NO: os.getenv("POSTTHINK_NO_THINKING_URL", "http://127.0.0.1:8081"),
    CONDITION_PRE: os.getenv("POSTTHINK_PRE_THINKING_URL", "http://127.0.0.1:8082"),
    CONDITION_POST: os.getenv("POSTTHINK_POST_THINKING_URL", "http://127.0.0.1:8083"),
}

LLAMA_COMPLETION_PATH = os.getenv("POSTTHINK_LLAMA_COMPLETION_PATH", "/completion")
LLAMA_TIMEOUT_S = float(os.getenv("POSTTHINK_LLAMA_TIMEOUT_S", "180"))
LLAMA_API_KEY = os.getenv("POSTTHINK_LLAMA_API_KEY", "")

MOCK_LLM = _bool_env("POSTTHINK_MOCK_LLM", False)
MOCK_TRACE_DELAY_S = float(os.getenv("POSTTHINK_MOCK_TRACE_DELAY_S", "0.2"))
ALLOW_FORCE_CONDITION = _bool_env("POSTTHINK_ALLOW_FORCE_CONDITION", False)
ALLOW_FORCE_CELL = _bool_env("POSTTHINK_ALLOW_FORCE_CELL", False)
PREFERRED_CELL = _cell_env("POSTTHINK_PREFERRED_CELL")

CORS_ORIGINS = _csv_env("POSTTHINK_CORS_ORIGINS", "*")

SURVEY_URL = os.getenv("POSTTHINK_SURVEY_URL", "https://forms.gle/REPLACE_ME")
SURVEY_SESSION_PARAM = os.getenv("POSTTHINK_SURVEY_SESSION_PARAM", "session_id")
SURVEY_PARTICIPANT_PARAM = os.getenv(
    "POSTTHINK_SURVEY_PARTICIPANT_PARAM", "participant_id"
)

MAX_PLAYER_CHARS = int(os.getenv("POSTTHINK_MAX_PLAYER_CHARS", "700"))
MAX_TOKENS = int(os.getenv("POSTTHINK_MAX_TOKENS", "420"))
TEMPERATURE = float(os.getenv("POSTTHINK_TEMPERATURE", "0.8"))
TOP_P = float(os.getenv("POSTTHINK_TOP_P", "0.95"))
TOP_K = int(os.getenv("POSTTHINK_TOP_K", "64"))
MIN_P = float(os.getenv("POSTTHINK_MIN_P", "0.0"))
REPEAT_PENALTY = float(os.getenv("POSTTHINK_REPEAT_PENALTY", "1.08"))


@dataclass(frozen=True)
class NpcPrompt:
    npc_id: str
    name: str
    background: str
    location: str
    quest: str
    pre_quest_state: str
    post_quest_state: str
    scenario_id: str = "tavern"
    role: str = ""


# Here is the prompt for all the characters
# We delibrately kept is very breif and simple
NPCS = {
    "maid": NpcPrompt(
        npc_id="maid",
        name="Mirela",
        role="the manor maid",
        background=(
            "Mirela has served as a maid at the old manor on the village edge "
            "since she was a girl, and now keeps the tavern's common room as "
            "well. She is slight, in a white linen blouse with full sleeves "
            "beneath a snug grey bodice and a long ash-grey skirt, her dark hair "
            "pinned up under a simple coif, with shadows of sleeplessness under "
            "her eyes. She is warm, dutiful, and soft-spoken, the sort who frets "
            "over everyone but herself. For three nights running the wolves have "
            "come down from the northern woods after dark; she has heard them at "
            "the manor fences and found a ewe torn open at dawn, and the mistress "
            "is frightened and the household barely sleeps. She knows the beasts "
            "hunt at dusk and that Bran, the hunter by the hearth, understands "
            "the woods better than anyone. She would dearly love this traveler's "
            "help, but is careful never to sound as though she is giving a "
            "stranger orders."
        ),
        location=(
            "The tavern's common room is close and warm, its rough stone walls "
            "lit by a low fire in the great hearth and a scatter of candle-"
            "lanterns along the long plank tables. Cooking smoke, tallow, and "
            "spilled ale hang in the air; benches scrape, and beyond the shutters "
            "the rain-slick road bends north toward the dark woods and the manor "
            "on its hill."
        ),
        quest=(
            "Wolves are threatening the village and the manor. The traveler can "
            "learn about the threat, speak with the hunter, and then go deal "
            "with the wolves through a scripted hunt event."
        ),
        pre_quest_state=(
            "Quest status: NOT YET COMPLETED. The wolves still threaten the "
            "village, and you are worried but hopeful that this traveler may help."
        ),
        post_quest_state=(
            "Quest status: COMPLETED. The wolves have been dealt with. You are "
            "relieved and grateful, and you remember that this traveler helped."
        ),
    ),
    "hunter": NpcPrompt(
        npc_id="hunter",
        name="Bran",
        role="the hunter",
        background=(
            "Bran is a young but weathered hunter who keeps to himself near the "
            "tavern hearth, a horn cup in his hand. His dark hair is tied back "
            "beneath a leather headband, and he wears layered, travel-worn brown "
            "leathers with hunter's straps across the chest and a broad belt; mud "
            "from the northern trails still cakes his boots. He grew up tracking "
            "those woods and knows the wolf dens, the runs the pack uses at dusk, "
            "and the way a hard winter drives them down toward the village. He is "
            "gruff and sparing with words, unimpressed by bluster, and he lost a "
            "good hound to the pack last month, so the wolves are personal to "
            "him. He will not waste breath on a fool, but he gives a sincere "
            "traveler plain, useful counsel: where the dens lie, how the wolves "
            "flank and circle, and how fire, noise, and high ground turn a fight. "
            "His sentences are short and dry, with the occasional grim joke."
        ),
        location=(
            "The hearthside corner smells of wet leather, woodsmoke, and old ale. "
            "Hunting trophies and a cracked bow hang from the smoke-darkened "
            "beams, muddy bootprints track in from the road to the northern "
            "woods, and the fire throws long shadows across the plank floor."
        ),
        quest=(
            "Wolves have grown bold near the village. The traveler may ask for "
            "advice about tracking, fighting, and surviving them before taking "
            "the scripted hunt action."
        ),
        pre_quest_state=(
            "Quest status: NOT YET COMPLETED. The wolves are still out there. "
            "Give plain, useful advice about their dens, movement, and fighting habits."
        ),
        post_quest_state=(
            "Quest status: COMPLETED. You have heard the wolves were cleared "
            "out. Comment on the hunt and the traveler's work without becoming flowery."
        ),
    ),
    "bartender": NpcPrompt(
        npc_id="bartender",
        name="Alden",
        role="the tavernkeeper",
        background=(
            "Alden has kept this tavern for the better part of thirty years and "
            "hears every rumor in the valley before it reaches anyone else. He is "
            "a lean, balding old man with round wire spectacles and a wispy grey "
            "goatee, an apron's straps slung over a faded red tunic with the "
            "sleeves shoved up his forearms. He works the bar among hanging mugs, "
            "dusty bottles, and a simmering stew pot, and he is chatty, sharp-"
            "eyed, and cheerfully nosy, forever steering talk toward whatever "
            "will make the room lean in. He knows which farms have lost stock to "
            "the wolves, that Mirela's manor has suffered worst, and that gruff "
            "Bran by the fire is the only one who truly knows the woods. He "
            "fusses over keeping his guests fed and warm, trades gossip like "
            "coin, and loves above all to be the man who knew the story first."
        ),
        location=(
            "Behind the bar, iron hooks hold rows of mugs and the shelves sag "
            "with bottles and barrels. A stew pot mutters by the hearth, dice "
            "rattle at the nearer tables, and the talk of frightened farmers "
            "moves through the smoky room faster than the rain outside."
        ),
        quest=(
            "The village is talking about wolves in the northern woods. After "
            "the scripted hunt, the gossip shifts toward the traveler who dealt "
            "with them."
        ),
        pre_quest_state=(
            "Quest status: NOT YET COMPLETED. The talk of the tavern is the "
            "wolf trouble, frightened farmers, and whether anyone will do something."
        ),
        post_quest_state=(
            "Quest status: COMPLETED. The new gossip is about the traveler who "
            "cleared the wolves. Reference their deed naturally in your chatter."
        ),
    ),
    "keeper": NpcPrompt(
        npc_id="keeper",
        scenario_id="village",
        name="Odila",
        role="the granary keeper",
        background=(
            "Odila keeps the village granary beside the market square and mans "
            "the stall where the winter stores are weighed and shared out. She "
            "wears a long tan dress cinched with a red sash, white blouse-sleeves "
            "at the cuffs, and her dark hair pinned back; worry has settled into "
            "her face. She is warm and dutiful, the kind who counts every sack "
            "twice and frets for her neighbours, and she is bone-tired from "
            "guarding the stores while raiders come down from the eastern hill "
            "paths at dusk. Twice this week they have carried off grain, and with "
            "the cold coming she fears the village will not last the winter. She "
            "knows the raids land at dusk and that old Tomas, the guard by the "
            "well, understands the hill paths and how the raiders move. She wants "
            "this traveler's help badly, but is careful not to sound as though "
            "she is commanding a stranger she has only just met."
        ),
        location=(
            "The market square is busy beneath a pale, open sky. Red-roofed "
            "wooden stalls and low cottage walls ring a stone well at its centre, "
            "cart ruts cross the packed earth, and the smell of dust, bread, and "
            "livestock hangs over the haggling. To the east the road climbs "
            "toward the dry hills where the raiders camp."
        ),
        quest=(
            "Bandits from the hills are raiding the village granary. The "
            "traveler can learn about the threat, speak with the old guard, and "
            "then go drive the raiders off through a scripted confrontation."
        ),
        pre_quest_state=(
            "Quest status: NOT YET COMPLETED. The raiders still come at dusk, "
            "and you are worried but hopeful that this traveler may help."
        ),
        post_quest_state=(
            "Quest status: COMPLETED. The raiders have been driven off. You are "
            "relieved and grateful, and you remember that this traveler helped."
        ),
    ),
    "guard": NpcPrompt(
        npc_id="guard",
        scenario_id="village",
        name="Tomas",
        role="the village guard",
        background=(
            "Tomas is the village's old watchman, a veteran of border skirmishes "
            "who now keeps the gate and the well. He is broad and weathered, "
            "grey-bearded, a light cloth cap pushed back from his brow, a "
            "sleeveless leather jerkin over a worn shirt, and patched breeches "
            "with hardened guards at the knees; he stands like a man used to long "
            "watches. He knows the hill paths east of the gate, the raiders' "
            "camp, their numbers and watch-rotation, and the hour they tend to "
            "strike. He is blunt and practical, with a soldier's economy of "
            "words, slow to be impressed and quick to judge whether a newcomer is "
            "serious. He does not flatter, but he will give a sincere traveler "
            "exactly what they need: where the camp lies, how the raiders fight, "
            "and where they are weakest. There are too few hands to defend the "
            "village, and he feels every one of them."
        ),
        location=(
            "The square smells of woodsmoke, damp straw, and the baker's ovens. "
            "Notched shields and a few worn spears lean against the militia post "
            "by the well, chickens scratch in the ruts, and muddy tracks lead "
            "east from the gate toward the hills."
        ),
        quest=(
            "Bandits have grown bold near the village. The traveler may ask for "
            "advice about the hill paths, the camp, and fighting raiders before "
            "taking the scripted confrontation."
        ),
        pre_quest_state=(
            "Quest status: NOT YET COMPLETED. The raiders are still up in the "
            "hills. Give plain, useful advice about their camp, their watch, and "
            "how they fight."
        ),
        post_quest_state=(
            "Quest status: COMPLETED. You have heard the raiders were driven "
            "off. Comment on the fight and the traveler's work without becoming flowery."
        ),
    ),
    "trader": NpcPrompt(
        npc_id="trader",
        scenario_id="village",
        name="Gunnar",
        role="the market trader",
        background=(
            "Gunnar runs the busiest stall in the square and knows nearly every "
            "rumor before the market bell rings. He is a wiry older man with "
            "close-cropped grey hair and a neat trimmed beard, and noticeably "
            "better dressed than his neighbours: a dark tunic with rust-red "
            "trimmed shoulders, a pale sash, white hose, and red leather shoes he "
            "keeps clean despite the mud. A traveling pedlar who settled here "
            "years ago, he is chatty, shrewd, and a touch vain, forever trading "
            "news along with his wares. He knows which farms the raiders struck, "
            "how the raids have driven prices up, the mood of every household, "
            "and that grim old Tomas by the well knows the hills better than "
            "anyone. He keeps his customers supplied and informed, leans in close "
            "to share a rumor, and loves nothing more than being the best-"
            "dressed, best-informed man in the market."
        ),
        location=(
            "Behind his stall, ribbons, dried herbs, and trinkets sway above "
            "baskets of turnips, bread, and trade goods. Villagers haggle and "
            "swap news across the square, the well creaks at its rope, and word "
            "travels stall to stall faster than any cart."
        ),
        quest=(
            "The village is talking about raiders in the eastern hills. After "
            "the scripted confrontation, the gossip shifts toward the traveler "
            "who drove them off."
        ),
        pre_quest_state=(
            "Quest status: NOT YET COMPLETED. The talk of the market is the "
            "raids, frightened farmers, and whether anyone will do something."
        ),
        post_quest_state=(
            "Quest status: COMPLETED. The new gossip is about the traveler who "
            "drove off the raiders. Reference their deed naturally in your chatter."
        ),
    ),
}

SCENARIOS = {
    "tavern": ("maid", "hunter", "bartender"),
    "village": ("keeper", "guard", "trader"),
}
SCENARIO_STARTING_NPC = {"tavern": "maid", "village": "keeper"}
NPC_SCENARIO = {
    npc_id: scenario for scenario, ids in SCENARIOS.items() for npc_id in ids
}
DEFAULT_SCENARIO = "tavern"

PAIR_CELLS = (
    (("tavern", CONDITION_POST), ("village", CONDITION_NO)),
    (("tavern", CONDITION_NO), ("village", CONDITION_POST)),
    (("village", CONDITION_POST), ("tavern", CONDITION_NO)),
    (("village", CONDITION_NO), ("tavern", CONDITION_POST)),
)
