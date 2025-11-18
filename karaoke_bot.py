import os
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands

from openai import OpenAI

# ------------- CONFIG -------------

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN env var not set")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set. Bot will use a tiny built-in fallback list.")
client_oa: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

ANSWER_TIME_LIMIT = 60  # seconds

intents = discord.Intents.default()
intents.message_content = True  # needed to read answers
bot = commands.Bot(command_prefix="!", intents=intents)


# ------------- STATE -------------

@dataclass
class KaraokeRound:
    channel_id: int
    message_id: int
    song_title: str
    artist: str
    acceptable_answers: List[str] = field(default_factory=list)
    is_active: bool = True
    ends_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(seconds=ANSWER_TIME_LIMIT))
    winner_id: Optional[int] = None
    genre: Optional[str] = None


# one active round per channel
active_rounds: Dict[int, KaraokeRound] = {}


# ------------- OPENAI HELPERS -------------

async def generate_song_round(genre: Optional[str]) -> Dict:
    """
    Uses OpenAI to generate a song round:
    - song_title
    - artist
    - clue (no long lyrics, just hints)
    - acceptable_answers (variations that should count as correct)
    Returns a dict or raises on fatal failure.
    """

    if not client_oa:
        # Minimal safe fallback if OpenAI isn't configured
        # (You can expand this list with public-domain style / generic hints.)
        return {
            "song_title": "Imagine",
            "artist": "John Lennon",
            "clue": "Classic peace anthem about imagining a world with no borders or possessions.",
            "acceptable_answers": ["imagine", "imagine - john lennon", "imagine john lennon"]
        }

    genre_text = f" in the {genre} genre" if genre else ""

    prompt = f"""
You are powering a Discord karaoke guessing game.

Pick a well-known, globally recognisable song{genre_text}.
Return ONLY a compact JSON object with this exact structure:

{{
  "song_title": "...",
  "artist": "...",
  "clue": "...",
  "acceptable_answers": ["...", "..."]
}}

Rules:
- "clue" must NOT contain full copyrighted lyrics. You may:
  - Describe the song's theme, mood, era, or scenario.
  - Mention the rhythm or style.
  - If you quote any lyrics, keep any single quoted fragment under 8 words and keep the total quoted content safely under 90 characters.
- Make the "clue" fun and guessable but not blatantly obvious.
- In "acceptable_answers", include reasonable variations:
  - song title alone
  - song title + artist
  - common shorter variants.
- Do not add any explanation outside JSON.
"""

    # Use Responses API for structured JSON
    resp = client_oa.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_output_tokens=250
    )

    text = ""
    for item in resp.output:
        if item.type == "message":
            for content_part in item.message.content:
                if content_part.type == "text":
                    text += content_part.text

    # Try to extract JSON
    text = text.strip()
    # Sometimes models wrap JSON in ```json ... ```
    if text.startswith("```"):
        text = text.strip("`")
        # remove optional 'json' header
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # ultra basic fallback
        return {
            "song_title": "Imagine",
            "artist": "John Lennon",
            "clue": "Classic peace anthem about imagining a world with no borders or possessions.",
            "acceptable_answers": ["imagine", "imagine - john lennon", "imagine john lennon"]
        }

    # normalise structure
    song_title = str(data.get("song_title", "")).strip()
    artist = str(data.get("artist", "")).strip()
    clue = str(data.get("clue", "")).strip()
    acceptable = data.get("acceptable_answers") or []

    if not isinstance(acceptable, list):
        acceptable = [str(acceptable)]

    # ensure basic sanity
    if not song_title or not clue:
        return {
            "song_title": "Imagine",
            "artist": "John Lennon",
            "clue": "Classic peace anthem about imagining a world with no borders or possessions.",
            "acceptable_answers": ["imagine", "imagine - john lennon", "imagine john lennon"]
        }

    return {
        "song_title": song_title,
        "artist": artist or "Unknown",
        "clue": clue,
        "acceptable_answers": [str(a).lower().strip() for a in acceptable if str(a).strip()]
    }


# ------------- GAME LOGIC -------------

def _normalise_answer(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _is_correct_guess(round_obj: KaraokeRound, guess: str) -> bool:
    g = _normalise_answer(guess)
    if not g:
        return False

    # Direct match against acceptable answers
    for ans in round_obj.acceptable_answers:
        if g == _normalise_answer(ans):
            return True

    # Also allow title-only match
    title = _normalise_answer(round_obj.song_title)
    if title and title in g:
        return True

    return False


async def run_karaoke_round(round_obj: KaraokeRound, channel: discord.TextChannel, msg: discord.Message):
    """
    Waits for answers up to ANSWER_TIME_LIMIT seconds.
    Freezes when someone gets it right, or ends when time is up.
    Also keeps the embed updated and occasionally 'bumps' it.
    """
    round_obj.message_id = msg.id
    active_rounds[channel.id] = round_obj

    # Start countdown updater
    bot.loop.create_task(update_embed_timer(round_obj, channel))

    # We'll also do a gentle bump halfway so it's not buried under heavy chat
    bot.loop.create_task(bump_reminder(round_obj, channel))

    try:
        while round_obj.is_active:
            remaining = (round_obj.ends_at - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                break

            try:
                guess_msg: discord.Message = await bot.wait_for(
                    "message",
                    timeout=remaining,
                    check=lambda m: (
                        m.channel.id == channel.id
                        and not m.author.bot
                        and round_obj.is_active
                    )
                )
            except asyncio.TimeoutError:
                # time up
                break

            if not round_obj.is_active:
                break

            if _is_correct_guess(round_obj, guess_msg.content):
                round_obj.is_active = False
                round_obj.winner_id = guess_msg.author.id

                # Edit embed to show winner
                try:
                    current_msg = await channel.fetch_message(round_obj.message_id)
                except discord.NotFound:
                    current_msg = None

                if current_msg and current_msg.embeds:
                    emb = current_msg.embeds[0]
                    emb.color = discord.Color.green()
                    emb.title = "🎤 Karaoke Round – We have a winner!"
                    emb.description = (
                        f"**Song:** {round_obj.song_title} – {round_obj.artist}\n"
                        f"**Winner:** {guess_msg.author.mention}\n\n"
                        "Thanks for playing! Start another `/karaoke` round anytime."
                    )
                    emb.set_footer(text="Round complete.")
                    await current_msg.edit(embed=emb)

                await channel.send(
                    f"✅ {guess_msg.author.mention} got it right: **{round_obj.song_title}** by **{round_obj.artist}**!"
                )
                break

        # If still active after loop, time’s up
        if round_obj.is_active:
            round_obj.is_active = False
            try:
                current_msg = await channel.fetch_message(round_obj.message_id)
            except discord.NotFound:
                current_msg = None

            if current_msg and current_msg.embeds:
                emb = current_msg.embeds[0]
                emb.color = discord.Color.red()
                emb.title = "⏰ Time's up!"
                emb.description = (
                    f"No one guessed it in time.\n\n"
                    f"**Song:** {round_obj.song_title} – {round_obj.artist}"
                )
                emb.set_footer(text="Round complete.")
                await current_msg.edit(embed=emb)

            await channel.send(
                f"⏰ Time's up! The song was **{round_obj.song_title}** by **{round_obj.artist}**."
            )

    finally:
        # clean up
        active_rounds.pop(channel.id, None)


async def update_embed_timer(round_obj: KaraokeRound, channel: discord.TextChannel):
    """
    Periodically edits the embed footer with the remaining time.
    Stops as soon as the round is inactive.
    """
    await asyncio.sleep(1)  # tiny delay so original embed is created

    while round_obj.is_active:
        try:
            msg = await channel.fetch_message(round_obj.message_id)
        except discord.NotFound:
            break

        if not msg.embeds:
            break

        remaining = int((round_obj.ends_at - datetime.now(timezone.utc)).total_seconds())
        if remaining < 0:
            remaining = 0

        emb = msg.embeds[0]
        emb.set_footer(text=f"⏱ Time left: {remaining}s")
        try:
            await msg.edit(embed=emb)
        except discord.HTTPException:
            pass

        if remaining <= 0:
            break

        # update every 5 seconds
        await asyncio.sleep(5)


async def bump_reminder(round_obj: KaraokeRound, channel: discord.TextChannel):
    """
    Gentle reminder so the question doesn't get completely buried.
    Sends a single bump around halfway, linking back to the main embed.
    """
    await asyncio.sleep(ANSWER_TIME_LIMIT / 2)
    if not round_obj.is_active:
        return

    try:
        msg = await channel.fetch_message(round_obj.message_id)
    except discord.NotFound:
        return

    await channel.send(
        f"🎵 Karaoke round still live! You’ve got about **{int((round_obj.ends_at - datetime.now(timezone.utc)).total_seconds())}s** left.\n"
        f"Jump to the question: {msg.jump_url}"
    )


# ------------- SLASH COMMANDS -------------

@bot.event
async def on_ready():
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s).")
    except Exception as e:
        print(f"Failed to sync commands: {e}")
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")


@bot.tree.command(name="karaoke", description="Start a 60-second karaoke guessing round in this channel.")
@app_commands.describe(
    genre="Optional: pick a genre like pop, rock, kpop, rnb, etc."
)
async def karaoke_command(interaction: discord.Interaction, genre: Optional[str] = None):
    channel = interaction.channel
    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message(
            "Please use this command in a normal text channel.", ephemeral=True
        )
        return

    if channel.id in active_rounds and active_rounds[channel.id].is_active:
        await interaction.response.send_message(
            "There is already a karaoke round running in this channel. Wait for it to finish first.",
            ephemeral=True
        )
        return

    await interaction.response.defer(thinking=True)

    try:
        song_data = await generate_song_round(genre)
    except Exception as e:
        print(f"Error generating song round: {e}")
        await interaction.followup.send(
            "I couldn't generate a new song round just now. Please try again in a moment.",
            ephemeral=True
        )
        return

    song_title = song_data["song_title"]
    artist = song_data["artist"]
    clue = song_data["clue"]
    acceptable_answers = song_data.get("acceptable_answers") or []

    ends_at = datetime.now(timezone.utc) + timedelta(seconds=ANSWER_TIME_LIMIT)

    emb = discord.Embed(
        title="🎤 Karaoke Guessing Game",
        description=(
            f"**Clue:** {clue}\n\n"
            "Guess the **song title** in chat!\n"
            f"You have **{ANSWER_TIME_LIMIT} seconds**.\n"
            "_(Song + artist will be revealed at the end.)_"
        ),
        color=discord.Color.blurple()
    )
    if genre:
        emb.add_field(name="Genre", value=genre, inline=True)
    emb.add_field(name="Mode", value="Open chat – first correct answer wins", inline=True)
    emb.set_footer(text=f"⏱ Time left: {ANSWER_TIME_LIMIT}s")

    msg = await interaction.followup.send(embed=emb, wait=True)

    round_obj = KaraokeRound(
        channel_id=channel.id,
        message_id=msg.id,
        song_title=song_title,
        artist=artist,
        acceptable_answers=acceptable_answers,
        is_active=True,
        ends_at=ends_at,
        winner_id=None,
        genre=genre,
    )

    # fire-and-forget the game loop
    bot.loop.create_task(run_karaoke_round(round_obj, channel, msg))


# ------------- RUN -------------

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
