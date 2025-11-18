import os
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import discord
from discord.ext import commands
from discord import app_commands

from openai import OpenAI

# ------------- CONFIG -------------

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN env var not set")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var not set")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client_oa = OpenAI(api_key=OPENAI_API_KEY)

ROUND_TIME = 60   # seconds to guess
BREAK_TIME = 15   # seconds between rounds
DEFAULT_ROUNDS = 10  # levels per /mic session

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix=commands.when_mentioned_or("!", "m."),
    intents=intents
)


# ------------- STATE -------------

@dataclass
class SongRound:
    song_title: str
    artist: str
    lyric_lines: List[str]
    acceptable_titles: List[str]
    acceptable_artists: List[str]


# per-channel flag to avoid multiple games in same channel
active_games: Dict[int, bool] = {}


# ------------- OPENAI HELPERS -------------

async def generate_song_round() -> SongRound:
    """
    Ask OpenAI for:
      - song_title
      - artist
      - lyric_lines (1-3 very short lines)
      - acceptable_title_answers
      - acceptable_artist_answers

    If anything goes wrong, we fall back to a static song
    so the game never breaks.
    """
    fallback = SongRound(
        song_title="Imagine",
        artist="John Lennon",
        lyric_lines=[
            "You may say I'm a dreamer",
            "But I'm not the only one",
        ],
        acceptable_titles=["imagine", "imagine - john lennon", "imagine john lennon"],
        acceptable_artists=["john lennon", "lennon"],
    )

    prompt = """
You are powering a Discord “guess the song” game using lyrics.

Pick a well-known, globally recognisable song.

Return ONLY a compact JSON object with this exact structure:

{
  "song_title": "...",
  "artist": "...",
  "lyric_lines": ["...", "...", "..."],
  "acceptable_title_answers": ["...", "..."],
  "acceptable_artist_answers": ["...", "..."]
}

Rules:
- "lyric_lines" must contain 1 to 3 very short lyric-style lines:
  - Each line MUST be 8 words or fewer.
  - The TOTAL characters across all lines MUST stay safely under 90 characters.
  - Do NOT output full verses or long passages.
- It is OK if lines resemble real lyrics, as long as they stay under the above limits.
- In "acceptable_title_answers":
  - include sensible variations of the song title (title alone, title + artist, common short forms).
- In "acceptable_artist_answers":
  - include reasonable variations of the artist name (full name, common short name).
- Do not include any explanation or text outside the JSON object.
"""

    try:
        resp = client_oa.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=400,
        )

        text = (resp.choices[0].message.content or "").strip()

        # Strip ```json ... ``` wrapper if present
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        data = json.loads(text)

        song_title = str(data.get("song_title", "")).strip()
        artist = str(data.get("artist", "")).strip()
        lyric_lines = data.get("lyric_lines") or []
        acc_title = data.get("acceptable_title_answers") or []
        acc_artist = data.get("acceptable_artist_answers") or []

        if not isinstance(lyric_lines, list):
            lyric_lines = [str(lyric_lines)]
        if not isinstance(acc_title, list):
            acc_title = [str(acc_title)]
        if not isinstance(acc_artist, list):
            acc_artist = [str(acc_artist)]

        def norm_list(values: List) -> List[str]:
            out: List[str] = []
            for v in values:
                s = str(v).strip()
                if s:
                    out.append(s)
            return out

        lyric_lines = norm_list(lyric_lines)[:3]
        safe_lines: List[str] = []
        total_chars = 0
        for line in lyric_lines:
            words = line.split()
            line_short = " ".join(words[:8])
            if len(line_short) > 90:
                line_short = line_short[:90]
            total_chars += len(line_short)
            if total_chars > 90:
                break
            safe_lines.append(line_short)

        acc_title = [s.lower().strip() for s in norm_list(acc_title)]
        acc_artist = [s.lower().strip() for s in norm_list(acc_artist)]

        if not song_title or not safe_lines:
            print("[MicMate] Missing title or lyric_lines, using fallback.")
            return fallback

        return SongRound(
            song_title=song_title,
            artist=artist or "Unknown",
            lyric_lines=safe_lines,
            acceptable_titles=acc_title,
            acceptable_artists=acc_artist,
        )

    except Exception as e:
        print("[MicMate] Error from OpenAI, using fallback:", repr(e))
        return fallback


# ------------- GAME HELPERS -------------

def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def is_correct_guess(song: SongRound, guess: str) -> bool:
    g = _norm(guess)
    if not g:
        return False

    title_norm = _norm(song.song_title)
    artist_norm = _norm(song.artist)

    # Check title answers
    for ans in song.acceptable_titles:
        if g == _norm(ans):
            return True
    if title_norm and title_norm in g:
        return True

    # Check artist answers
    for ans in song.acceptable_artists:
        if g == _norm(ans):
            return True
    if artist_norm and artist_norm in g:
        return True

    return False


async def play_single_level(
    channel: discord.TextChannel,
    level: int,
    total_levels: int,
    scores: Dict[int, int],
    last_song: Optional[SongRound] = None,
) -> Tuple[Optional[int], SongRound]:
    """
    Runs one level:
    - sends lyrics embed
    - avoids repeating the previous song if possible
    - waits up to ROUND_TIME for correct guess
    """
    # Try a few times to get a different song than last round
    attempts = 0
    while True:
        song = await generate_song_round()
        attempts += 1

        if last_song is None:
            break

        same_title = _norm(song.song_title) == _norm(last_song.song_title)
        same_artist = _norm(song.artist) == _norm(last_song.artist)

        if not (same_title and same_artist):
            break  # different song, good

        if attempts >= 5:
            # Give up after 5 tries, just use whatever we got
            print("[MicMate] Got the same song multiple times from OpenAI, using it anyway.")
            break

    # Build lyrics block
    lyrics_block = "\n".join(f"• “{line}”" for line in song.lyric_lines)

    desc = (
        f"**Lyrics:**\n{lyrics_block}\n\n"
        f"Mode: Guess the **TITLE** or **ARTIST** in chat.\n"
        f"You have **{ROUND_TIME} seconds**."
    )

    embed = discord.Embed(
        title=f"🎶 Mic – Level {level}",
        description=desc,
        color=discord.Color.blurple(),
    )
    embed.set_footer(text=f"Time left: {ROUND_TIME} seconds")

    question_msg = await channel.send(embed=embed)

    # Wait for guesses
    winner_id: Optional[int] = None
    winner_msg: Optional[discord.Message] = None
    ends_at = datetime.now(timezone.utc) + timedelta(seconds=ROUND_TIME)

    while True:
        remaining = (ends_at - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            break

        try:
            msg: discord.Message = await bot.wait_for(
                "message",
                timeout=remaining,
                check=lambda m: (
                    m.channel.id == channel.id
                    and not m.author.bot
                ),
            )
        except asyncio.TimeoutError:
            break

        if is_correct_guess(song, msg.content):
            winner_id = msg.author.id
            winner_msg = msg
            break

    # Winner reaction + answer embed
    if winner_id is not None and winner_msg is not None:
        try:
            await winner_msg.add_reaction("🎤")
        except discord.HTTPException:
            pass

        answer_embed = discord.Embed(
            title="✅ Answer guessed!",
            description=(
                f"**Song:** {song.song_title} – {song.artist}\n"
                f"**Winner:** <@{winner_id}>\n\n"
                f"Next song in **{BREAK_TIME} seconds**..."
            ),
            color=discord.Color.green(),
        )
        answer_embed.set_footer(text="Answer locked. Get ready for the next level.")
        await channel.send(embed=answer_embed)

        # Update scores
        scores[winner_id] = scores.get(winner_id, 0) + 1

    else:
        answer_embed = discord.Embed(
            title="⏰ Time's up!",
            description=(
                "No one guessed it in time.\n\n"
                f"**Song:** {song.song_title} – {song.artist}\n\n"
                "Game over for this Mic session.\n"
                "Use `/mic` or `m.mic` to start a new game."
            ),
            color=discord.Color.red(),
        )
        answer_embed.set_footer(text="Round failed. Mic session ended.")
        await channel.send(embed=answer_embed)


    return winner_id, song


async def show_ranking(
    channel: discord.TextChannel,
    scores: Dict[int, int],
    next_level: int,
    total_levels: int,
    final: bool = False,
):
    """Show Team Ranking embed."""
    embed = discord.Embed(
        title="🏆 Team Ranking",
        color=discord.Color.gold(),
    )

    if not scores:
        embed.description = "No points yet. Everyone is still on 0."
    else:
        sorted_scores = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        lines = []
        for i, (user_id, score) in enumerate(sorted_scores, start=1):
            lines.append(f"**{i}.** <@{user_id}> — `{score}` point(s)")
        embed.description = "\n".join(lines)

    if final:
        embed.add_field(
            name="Game Over",
            value="That’s the last level. Use `/mic` or `m.mic` to start a new game.",
            inline=False,
        )
    else:
        embed.add_field(
            name="Next Step",
            value=f"Level **{next_level}** will start soon.",
            inline=False,
        )

    await channel.send(embed=embed)


async def run_mic_game(
    channel: discord.TextChannel,
    total_levels: int,
):
    active_games[channel.id] = True
    scores: Dict[int, int] = {}
    last_song: Optional[SongRound] = None

    await channel.send(
        f"🎮 **Mic game starting!**\n"
        f"Levels: `{total_levels}`. Try to guess the **title or artist** each round.\n"
        f"Type `/mic` or `m.mic` again later to start a fresh game."
    )

    for level in range(1, total_levels + 1):
        if not active_games.get(channel.id, False):
            break

        winner_id, this_song = await play_single_level(
            channel,
            level,
            total_levels,
            scores,
            last_song=last_song,
        )
        last_song = this_song  # remember for next round
        
        # If no one got it right, game stops here
        if winner_id is None:
            await show_ranking(
                channel,
                scores,
                next_level=level,      # not used when final=True
                total_levels=total_levels,
                final=True,
            )
            break       
       
        # Otherwise continue as normal
        await asyncio.sleep(BREAK_TIME)

        final_round = (level == total_levels)
        await show_ranking(
            channel,
            scores,
            next_level=level + 1,
            total_levels=total_levels,
            final=final_round,
        )

        if not final_round:
            await asyncio.sleep(3)


        # break again so the ranking can be read before next level
        if not final_round:
            await asyncio.sleep(3)

    # game finished
    active_games.pop(channel.id, None)


# ------------- EVENTS -------------

@bot.event
async def on_ready():
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s).")
    except Exception as e:
        print(f"Failed to sync commands: {e}")
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")


# ------------- SLASH COMMANDS -------------

@bot.tree.command(
    name="mic",
    description="Start a Mic lyrics guessing game (like Gartic, multiple levels)."
)
@app_commands.describe(
    rounds=f"How many levels to play (default {DEFAULT_ROUNDS})"
)
async def mic_slash(
    interaction: discord.Interaction,
    rounds: Optional[int] = None,
):
    channel = interaction.channel
    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message(
            "Please use this in a normal text channel.",
            ephemeral=True,
        )
        return

    if active_games.get(channel.id, False):
        await interaction.response.send_message(
            "There is already a Mic game running in this channel.",
            ephemeral=True,
        )
        return

    total_levels = rounds or DEFAULT_ROUNDS
    if total_levels < 1:
        total_levels = 1
    if total_levels > 50:
        total_levels = 50

    await interaction.response.send_message(
        f"Starting Mic game with **{total_levels}** levels...", ephemeral=True
    )

    bot.loop.create_task(run_mic_game(channel, total_levels))


# ------------- PREFIX COMMANDS -------------

@bot.command(name="mic")
async def mic_prefix(ctx: commands.Context, rounds: Optional[int] = None):
    """Prefix version: m.mic [rounds]"""
    channel = ctx.channel
    if not isinstance(channel, discord.TextChannel):
        await ctx.reply(
            "Please use this in a normal text channel.",
            mention_author=False,
        )
        return

    if active_games.get(channel.id, False):
        await ctx.reply(
            "There is already a Mic game running in this channel.",
            mention_author=False,
        )
        return

    total_levels = rounds or DEFAULT_ROUNDS
    try:
        total_levels = int(total_levels)
    except (TypeError, ValueError):
        total_levels = DEFAULT_ROUNDS

    if total_levels < 1:
        total_levels = 1
    if total_levels > 50:
        total_levels = 50

    await ctx.reply(
        f"Starting Mic game with **{total_levels}** levels...",
        mention_author=False,
    )

    bot.loop.create_task(run_mic_game(channel, total_levels))


# ------------- RUN -------------

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
