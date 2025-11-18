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

# Per-channel round info for hints / passes
current_song: Dict[int, SongRound] = {}       # channel_id -> current SongRound
hints_used: Dict[int, int] = {}              # channel_id -> 0..3 (per game)
passes_used: Dict[int, int] = {}             # channel_id -> 0..3 (per game)


# ------------- HELPERS -------------

def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def is_correct_guess(song: SongRound, guess: str) -> bool:
    g = _norm(guess)
    if not g:
        return False

    title_norm = _norm(song.song_title)
    artist_norm = _norm(song.artist)

    for ans in song.acceptable_titles:
        if g == _norm(ans):
            return True
    if title_norm and title_norm in g:
        return True

    for ans in song.acceptable_artists:
        if g == _norm(ans):
            return True
    if artist_norm and artist_norm in g:
        return True

    return False


# ------------- OPENAI -------------

async def generate_song_round(last_song: Optional[SongRound] = None) -> SongRound:
    """
    Ask the model for a song with 1–3 short lyric lines and acceptable answers.
    No local fallback. If this fails, caller will stop the game.
    """
    avoid_text = """
You must not choose "Shape of You" by Ed Sheeran.
"""
    if last_song is not None:
        avoid_text += f"""
The previous round used: "{last_song.song_title}" by {last_song.artist}.
You must not choose that same song again this round.
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
  - Each line must be 8 words or fewer.
  - The total characters across all lines must stay safely under 90 characters.
  - Do not output full verses or long passages.
- It is OK if lines resemble real lyrics, as long as they stay under the above limits.
- In "acceptable_title_answers":
  - include sensible variations of the song title (title alone, title + artist, common short forms).
- In "acceptable_artist_answers":
  - include reasonable variations of the artist name (full name, common short name).
- Do not include any explanation or text outside the JSON object.
"""

    resp = client_oa.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=400,
    )

    text = (resp.choices[0].message.content or "").strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print("[MicMate] Invalid JSON from model:", repr(e), "raw:", text[:200])
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
        print("[MicMate] Missing title/lyrics. Raw:", text[:200])
        raise RuntimeError("Model response missing song_title or lyric_lines")

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
    used_titles: Optional[set] = None,
) -> Tuple[Optional[int], Optional[SongRound], bool]:
    """
    Runs one level and returns (winner_id, song, passed_flag).

    winner_id:
      - user id if someone guessed correctly
      - None if no one guessed / pass / error

    song:
      - SongRound object, or None if model failed

    passed_flag:
      - True if the round was skipped via pass
      - False otherwise
    """

    # Get a song, trying a few times to avoid any already-used title
    song: Optional[SongRound] = None
    passed_flag = False

    if used_titles is None:
        used_titles = set()

    try:
        attempts = 0
        last_title_norm = _norm(last_song.song_title) if last_song else None

        while attempts < 10:  # a few extra tries
            candidate = await generate_song_round(last_song)
            attempts += 1

            title_key = _norm(candidate.song_title)
            same_as_last = last_title_norm and title_key == last_title_norm
            seen_before = title_key in used_titles

            if not same_as_last and not seen_before:
                song = candidate
                break

        if song is None:
            # even after retries, just take the last candidate
            song = candidate



    except Exception as e:
        print("[MicMate] Fatal error getting song:", repr(e))
        await channel.send(
            "⚠️ I couldn't load a new song just now, "
            "so this Mic game has been stopped. Try `/mic` again in a bit."
        )
        return None, None, False

    # Register current song for hints/passes
    current_song[channel.id] = song

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

        content = msg.content.lower().strip()

        # Handle pass via plain message "m.pass"
        if content.startswith("m.pass"):
            used = passes_used.get(channel.id, 0)
            if used >= 3:
                await channel.send(
                    "🚫 No passes left for this Mic game (3/3 used)."
                )
                continue
            passes_used[channel.id] = used + 1
            passed_flag = True
            await channel.send(
                f"⏭️ Song skipped with a pass ({passes_used[channel.id]}/3 used). "
                "Next level will start shortly."
            )
            break

        # Normal guess
        if is_correct_guess(song, msg.content):
            winner_id = msg.author.id
            winner_msg = msg
            break

    # Outcome messages
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

    elif passed_flag:
        # Song was skipped by pass
        answer_embed = discord.Embed(
            title="⏭️ Song skipped!",
            description=(
                f"The round was passed ({passes_used.get(channel.id, 0)}/3 used).\n"
                "Next song will load in a moment."
            ),
            color=discord.Color.blurple(),
        )
        answer_embed.set_footer(text="Round skipped. Moving to the next level.")
        await channel.send(embed=answer_embed)

    else:
        # Time's up, no winner and no pass → game over
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

    return winner_id, song, passed_flag


async def show_ranking(
    channel: discord.TextChannel,
    scores: Dict[int, int],
    next_level: int,
    total_levels: int,
    final: bool = False,
):
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

    # 👉 add this line
    used_titles: set[str] = set()

    # reset per-game counters
    hints_used[channel.id] = 0
    passes_used[channel.id] = 0
    current_song.pop(channel.id, None)

    await channel.send("🎮 **Mic game starting!**")

    try:
        for level in range(1, total_levels + 1):
            if not active_games.get(channel.id, False):
                break

            # 👇 pass used_titles into play_single_level
            winner_id, this_song, passed_flag = await play_single_level(
                channel,
                level,
                total_levels,
                scores,
                last_song=last_song,
                used_titles=used_titles,
            )

            if this_song is None:
                # error already messaged in channel
                break

            last_song = this_song
            used_titles.add(_norm(this_song.song_title))

            # Time up with no winner and no pass -> final ranking + stop
            if winner_id is None and not passed_flag:
                await show_ranking(
                    channel,
                    scores,
                    next_level=level,
                    total_levels=total_levels,
                    final=True,
                )
                break

            # Winner or pass -> continue
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
        await channel.send("⚠️ Something went wrong and this Mic game had to stop.")
    finally:
        active_games.pop(channel.id, None)
        current_song.pop(channel.id, None)
        hints_used.pop(channel.id, None)
        passes_used.pop(channel.id, None)


# ------------- HINT & PASS COMMANDS -------------

async def use_hint(channel: discord.TextChannel, user: discord.abc.User):
    if not active_games.get(channel.id, False):
        await channel.send("There is no active Mic game in this channel.")
        return

    song = current_song.get(channel.id)
    if song is None:
        await channel.send("There is no active round to use a hint on.")
        return

    used = hints_used.get(channel.id, 0)
    if used >= 3:
        await channel.send("🚫 No hints left for this Mic game (3/3 used).")
        return

    hints_used[channel.id] = used + 1
    idx = used  # 0-based

    # Hint order: lyric 1, lyric 2 (if exists), then title-start hint
    if idx < len(song.lyric_lines):
        line = song.lyric_lines[idx]
        text = f"Hint {hints_used[channel.id]}/3: another lyric line – “{line}”."
    else:
        first_word = song.song_title.split()[0]
        text = (
            f"Hint {hints_used[channel.id]}/3: "
            f"The title starts with **{first_word}**."
        )

    await channel.send(text)


@bot.command(name="hint")
async def hint_prefix(ctx: commands.Context):
    await use_hint(ctx.channel, ctx.author)


@bot.tree.command(
    name="mic_hint",
    description="Use one of the shared hints in the current Mic game.",
)
async def mic_hint_slash(interaction: discord.Interaction):
    channel = interaction.channel
    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message(
            "Use this in a normal text channel.", ephemeral=True
        )
        return
    await interaction.response.send_message(
        "Hint used.", ephemeral=True
    )
    await use_hint(channel, interaction.user)


@bot.tree.command(
    name="mic_pass",
    description="Use one of the shared passes to skip the current song.",
)
async def mic_pass_slash(interaction: discord.Interaction):
    channel = interaction.channel
    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message(
            "Use this in a normal text channel.", ephemeral=True
        )
        return

    if not active_games.get(channel.id, False):
        await interaction.response.send_message(
            "There is no active Mic game in this channel.",
            ephemeral=True,
        )
        return

    used = passes_used.get(channel.id, 0)
    if used >= 3:
        await interaction.response.send_message(
            "No passes left for this Mic game (3/3 used).",
            ephemeral=True,
        )
        return

    # We don't skip here directly; we send a synthetic "m.pass" message
    # so play_single_level sees it and handles the pass logic.
    await interaction.response.send_message(
        "Pass used. Skipping this song…", ephemeral=True
    )
    await channel.send("m.pass")  # triggers pass handling inside play_single_level


# ------------- EVENTS -------------

@bot.event
async def on_ready():
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s).")
    except Exception as e:
        print(f"Failed to sync commands: {e}")
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")


# ------------- SLASH / PREFIX START -------------

@bot.tree.command(
    name="mic",
    description="Start a Mic lyrics guessing game."
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


@bot.command(name="mic")
async def mic_prefix(ctx: commands.Context, rounds: Optional[int] = None):
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
