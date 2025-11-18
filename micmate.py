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

# Same model/key setup as MugOff
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
client_oa = OpenAI(api_key=OPENAI_API_KEY)

ROUND_TIME = 60   # seconds to guess
BREAK_TIME = 15   # seconds between rounds
DEFAULT_ROUNDS = 10  # levels per /mic session

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix=commands.when_mentioned_or("!", "m."),
    intents=intents,
)


# ------------- STATE -------------

@dataclass
class SongRound:
    song_title: str
    artist: str
    lyric_lines: List[str]
    acceptable_titles: List[str]
    acceptable_artists: List[str]


# One active game per channel
active_games: Dict[int, bool] = {}


# ------------- HELPERS -------------

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


# ------------- OPENAI (NO FALLBACK) -------------

async def generate_song_round(last_song: Optional[SongRound] = None) -> SongRound:
    """
    Ask OpenAI for a song with 1–3 short lyric lines and acceptable answers.
    NO local fallback – if this fails, caller should stop the game.
    """
    avoid_text = """
You MUST NOT choose "Shape of You" by Ed Sheeran for this game.
"""
    if last_song is not None:
        avoid_text += f"""
The previous round used: "{last_song.song_title}" by {last_song.artist}.
You MUST NOT choose that same song again this round.
"""

    prompt = f"""
You are powering a Discord “guess the song” game using lyrics.

Pick a well-known, globally recognisable song that many people are likely to know.
{avoid_text}

Return ONLY a compact JSON object with this exact structure:

{{
  "song_title": "...",
  "artist": "...",
  "lyric_lines": ["...", "...", "..."],
  "acceptable_title_answers": ["...", "..."],
  "acceptable_artist_answers": ["...", "..."]
}}

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

    # Use the same style as MugOff (chat.completions)
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

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print("[MicMate] OpenAI returned invalid JSON:", repr(e), "raw:", text[:200])
        raise

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

    acc_title = [_norm(s) for s in norm_list(acc_title)]
    acc_artist = [_norm(s) for s in norm_list(acc_artist)]

    if not song_title or not safe_lines:
        print("[MicMate] OpenAI response missing title/lyrics. Raw:", text[:200])
        raise RuntimeError("OpenAI response missing song_title or lyric_lines")

    return SongRound(
        song_title=song_title,
        artist=artist or "Unknown",
        lyric_lines=safe_lines,
        acceptable_titles=acc_title,
        acceptable_artists=acc_artist,
    )


# ------------- GAME LOGIC -------------

async def play_single_level(
    channel: discord.TextChannel,
    level: int,
    total_levels: int,
    scores: Dict[int, int],
    last_song: Optional[SongRound] = None,
) -> Tuple[Optional[int], Optional[SongRound]]:
    """
    Runs one level:
    - gets a song (asks OpenAI, tries to avoid repeating last round)
    - sends lyrics embed
    - waits up to ROUND_TIME for a correct guess
    - reacts with 🎤 on the winning message
    - sends Answer / Time's up embed

    Returns (winner_id or None, song or None).
    If song is None, it means OpenAI failed and caller should stop.
    """

    # Try up to 5 times to get a different song than last round.
    # All attempts are still via OpenAI; no local fallback.
    song: Optional[SongRound] = None
    try:
        attempts = 0
        while attempts < 5:
            candidate = await generate_song_round(last_song)
            attempts += 1

            if last_song is None:
                song = candidate
                break

            same_title = _norm(candidate.song_title) == _norm(last_song.song_title)
            same_artist = _norm(candidate.artist) == _norm(last_song.artist)

            if not (same_title and same_artist):
                song = candidate
                break

        if song is None:
            # even after 5 attempts we only got the same thing
            song = candidate

    except Exception as e:
        print("[MicMate] Fatal error getting song from OpenAI:", repr(e))
        await channel.send(
            "⚠️ I couldn't get a new song from OpenAI. "
            "Mic game has been stopped. Try `/mic` again in a bit."
        )
        return None, None

    lyrics_block = "\n".join(f"• “{line}”" for line in song.lyric_lines)
    desc = (
        f"**Lyrics:**\n{lyrics_block}\n\n"
        "Mode: Guess the **TITLE** or **ARTIST** in chat.\n"
        f"You have **{ROUND_TIME} seconds**."
    )

    embed = discord.Embed(
        title=f"🎶 Mic – Level {level}",
        description=desc,
        color=discord.Color.blurple(),
    )
    embed.set_footer(text=f"Time left: {ROUND_TIME} seconds")

    await channel.send(embed=embed)

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

    # Winner vs no-winner embeds
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
    """Show Team Ranking embed (no level x/total, only level x)."""
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
            value="That’s the last level for this Mic session.\n"
                  "Use `/mic` or `m.mic` to start a new game.",
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
    """Main game loop for one Mic session in a channel."""
    active_games[channel.id] = True
    scores: Dict[int, int] = {}
    last_song: Optional[SongRound] = None

    await channel.send("🎮 **Karaoke game starting!**")


    try:
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

            # If OpenAI failed completely, this_song will be None – stop here
            if this_song is None:
                break

            last_song = this_song

            # If no winner → stop the whole game here
            if winner_id is None:
                await show_ranking(
                    channel,
                    scores,
                    next_level=level,
                    total_levels=total_levels,
                    final=True,
                )
                break

            # Winner path → continue
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

    except Exception as e:
        print("[MicMate] Unexpected error in run_mic_game:", repr(e))
        await channel.send("⚠️ Mic ran into an error and had to stop this game.")
    finally:
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


# ------------- SLASH COMMAND -------------

@bot.tree.command(
    name="mic",
    description="Start a Mic lyrics guessing game (multiple levels)."
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
        f"Starting Mic game with **{total_levels}** levels...",
        ephemeral=True,
    )

    bot.loop.create_task(run_mic_game(channel, total_levels))


# ------------- PREFIX COMMAND -------------

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
