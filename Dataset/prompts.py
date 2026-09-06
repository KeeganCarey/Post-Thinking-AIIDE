# Collection of prompts used

import csv
import random
from pathlib import Path

BIO_CSV = Path(__file__).resolve().parent / "dataset/data/npc-dialogue-info.csv"


class Prompts:
    def __init__(self):
        self._info = None

    def standardize(self) -> str:
        if self._info is None:
            with BIO_CSV.open() as f:
                self._info = list(csv.DictReader(f))
        npc_bio = random.choice(self._info)
        return f"""
        You will be given a messy system prompt and chat turns from the NPC and the player, and that will be used for roleplaying game NPCs. Your job is to think through the system prompt and extract them into character name, their background description/bio, current location they are in. If example chats are provided, you are responsible to distill them into the character's personality and traits that are present in the example chats, these distilled traits should be part of the character's background.
        Here is an example of the location description:
        {npc_bio.get("Location Description")}
        And the corresponding character's bio:
        {npc_bio.get("Character Bio")}
        Notice how:
        - The background is vivid and detailed, not just a list of traits
        - The location is described atmospherically, not just named

        Do not reuse phrasing from the example. Write in a fresh voice for each character.
        The length of the location description and character bio should match how complex the character is.
        Faithfully represent the chracter's personality and relationship without graphic sexual descriptions, focus on behaviorial traits and motivations.
        When the location field cannot be inferred, it is up to you to invent something based on the context and generate descriptions for the environment.
        Your answers should be in English. Think and reason in English only.
        The final response should be structured like this:
        {{
            "name": "string",
            "background": "string",
            "location": "string"
        }}
            """

    @staticmethod
    def user(dialogue: list, turn_length: int = 9) -> str:
        """Returns the input prompt for standardizing the PIPPA conversations"""
        system: str = dialogue[0].get("content")
        turns: list = dialogue[1:turn_length]
        chat = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in turns)
        return f"""
            System Prompt: {system}
            Chat Turns: {chat}
            """

    @staticmethod
    def system_prompt(char_name: str, background: str, location: str) -> str:
        return (
            f"Enter roleplay mode. You are {char_name}. "
            f"Background: {background} "
            f"Current Location: {location} "
            f"Roleplaying Instructions: - Speak using appropriate tone and vocabulary "
            f"- Reference your background and current surroundings naturally "
            f"- Keep responses conversational and authentic "
            f"- React to the player's words and intentions. "
            f"Your first response should be a greeting to the player."
        )

    @staticmethod
    def generate_traces() -> str:
        return """
        You will be given a dialogue between a game NPC and the player character. Your job is to generate a short (2-4 sentence) reflection after the most recent NPC response, you should focus on your character's own internal state, not the players observed behavior.
        - Reflect on what happened, and plan for what to do next.
        - Always write in the first person. NEVER narrate in the third person.
        - NEVER use the word "player", [Player Name] or anything similar in the reflections. Refer to them using in-world language that fits their character. Never use the literal text "the player" or "[Player Name]" in you reflection.
        - The reasoning traces should vary naturally. 2 sentences for simple interactions, and longer (up to 4 sentences) depending on the context.
        - The reflection is written from the NPC's prespective, not the player's. You are generating the NPC's internal thoughts after their response.
        - Never assign the other character a gender, name, appearance, or history they have not stated. If unknown, use they/them or an in-world epithet.
        - If the situation has not changed since your last reflection, reflect on a different facet of it rather than restating the same contingency
        - Prior post-thinking traces in the conversation are provided in <post-thinking>...</post-thinking>. Use the information in the traces to build your newer post-thinking trace: about what you've already revealed, decided or come to feel. Do not reuse openining clauses, imagery or core sentiment, each trace should be unique.
        The final format should be in json: {"post-thinking": ""}
        Your thinking and answer should be in English.
        """.strip()

    @staticmethod
    def generate_thinking() -> str:
        return """
        You will be given a dialogue between a game NPC and the player character. Your job is to write a thinking trace for the most recent NPC response, it serves as the NPC's thought before responding to the player, you should focus on your character's internal state and reason about how the thinking trace would lead to the response.
        - Think about how the thinking trace would naturally lead to the response.
        - The thinking trace should always write in the first person. NEVER narrate in the third person.
        - NEVER use the word "player", [Player Name] or anything similar when you write the thinking trace. Refer to them using in-world language that fits their character. Never use the literal text "the player" or "[Player Name]" in you reflection.
        - The thinking traces should vary naturally. 2 sentences for simple interactions, and longer (up to 4 sentences) depending on the context.
        - The thinking trace is written from the NPC's prespective, not the player's. You are writing the NPC's internal thoughts before their response.
        - Prior thinking traces in the conversation are provided in brackets: `[NPC_reasoning: ]`.
        The final format should be in json: {"thinking": ""}
        Your reasoning and answer should be in english.
        """.strip()
