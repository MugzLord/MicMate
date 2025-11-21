import os
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Set
import difflib
import io
import base64



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

ROUND_TIME = 60  # seconds to guess
BREAK_TIME = 5   # seconds between rounds
RANK_DELAY = 2   # seconds before posting Team Ranking

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
    hints: List[str] = field(default_factory=list)  # extra non-lyric hints

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
    """Stricter matching: must match title/artist, not just contain a word."""
    g = _norm(guess)
    if not g:
        return False

    title_norm = _norm(song.song_title)
    artist_norm = _norm(song.artist)

    # 1) Exact match against acceptable variations returned by the model
    if g in song.acceptable_titles or g in song.acceptable_artists:
        return True

    # 2) Exact match to full title or full artist
    if g == title_norm or g == artist_norm:
        return True

    # 3) Allow messages that contain BOTH full title and full artist
    #    e.g. "all i want for christmas is you mariah carey"
    if title_norm and artist_norm and title_norm in g and artist_norm in g:
        return True

    # No more substring-only matches like just "christmas" or "mariah"
    return False

# ------------- OPENAI -------------
async def generate_song_round(
    last_song: Optional[SongRound] = None,
    used_titles: Optional[Set[str]] = None,
    genre: Optional[str] = None,
    year: Optional[str] = None,
) -> SongRound:
    """
    Ask the model for a song with 1–3 short lyric lines,
    some non-lyric hints, and acceptable answers.
    No local hard-coded fallback – if this fails repeatedly,
    caller will stop the game.
    """
    if used_titles is None:
        used_titles = set()

    # Genre / year text for the prompt
    genre_text = ""
    if genre:
        g_lower = genre.lower().strip()

        # Special handling for Christmas / holiday
        if any(key in g_lower for key in ["christmas", "xmas", "holiday"]):
            genre_text = (
                "\nYou MUST choose a well-known Christmas / holiday song."
                "\nThink of classic or popular festive tracks that people are likely to know."
                "\nDo NOT choose non-Christmas songs."
            )
        else:
            genre_text = (
                f"\nYou MUST choose a song that clearly fits this genre or scene: {genre}."
                "\nDo NOT choose songs from other genres."
            )

    year_text = ""
    if year:
        year_text = f"\nPrefer a song released around this year/era: {year}."


    # Songs we NEVER want again (global)
    avoid_lines = [
        'You must not choose "Shape of You" by Ed Sheeran.',
        'You must not choose "Billie Jean" by Michael Jackson.',
    ]

    # Extra: if this is a Christmas / holiday genre, avoid the super overused ones
    if genre and any(k in genre.lower() for k in ["christmas", "xmas", "holiday"]):
        avoid_christmas_titles = [
            "All I Want for Christmas Is You",
            "Last Christmas",
            "Jingle Bells",
            "Jingle Bell Rock",
            "Rockin' Around the Christmas Tree",
            "Feliz Navidad",
            "Santa Tell Me",
            "It's Beginning to Look a Lot Like Christmas",
        ]
        avoid_lines.append(
            "You must not choose any of these very common Christmas songs; "
            "instead, pick other well-known festive songs that people still recognise: "
            + ", ".join(f'"{t}"' for t in avoid_christmas_titles)
            + ". This is a hard rule."
        )

    # Tell the model which titles already appeared in this Mic game
    if used_titles:
        used_list = ", ".join(f'"{t}"' for t in used_titles)
        avoid_lines.append(
            f"You must not choose any of these titles already used in this game: {used_list}."
        )

    # And also don’t repeat the immediate previous round
    if last_song is not None:
        avoid_lines.append(
            f'The previous round used: "{last_song.song_title}" by {last_song.artist}. '
            "You must not choose that same song again this round."
        )

    avoid_text = "\n".join(avoid_lines)

    prompt = f"""
You are powering a Discord “guess the song” game using lyrics.

Pick a well-known, globally recognisable song that many people are likely to know.
{avoid_text}
{genre_text}{year_text}

Return ONLY a compact JSON object with this exact structure:

{{
  "song_title": "...",
  "artist": "...",
  "lyric_lines": ["...", "...", "..."],
  "hints": ["...", "...", "..."],
  "acceptable_title_answers": ["...", "..."],
  "acceptable_artist_answers": ["...", "..."]
}}

Rules:
- "lyric_lines" must contain 1 to 3 very short lyric-style lines:
  - Each line must be 8 words or fewer.
  - The total characters across all lines must stay safely under 90 characters.
  - Do not output full verses or long passages.
- It is OK if lines resemble real lyrics, as long as they stay under the above limits.
- "hints" must contain 1 to 3 short hint lines that DO NOT quote the lyrics:
  - focus on era, mood, theme, interesting facts about the artist,
    word count of the title, first letter, country, etc.
  - do not copy any of the lyric lines or quote them directly.
- In "acceptable_title_answers":
  - include sensible variations of the song title (title alone, title + artist, common short forms).
- In "acceptable_artist_answers":
  - include reasonable variations of the artist name (full name, common short name).
- Do not include any explanation or text outside the JSON object.
"""

    # First call
    resp = client_oa.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=400,
    )

    data = None
    attempts_json = 0

    # --- SAFE JSON PARSE / VALIDATION LOOP ---
    while attempts_json < 5:
        attempts_json += 1

        text = (resp.choices[0].message.content or "").strip()

        # Strip ```json fences if the model adds them
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            candidate = json.loads(text)
        except Exception as e:
            print("[MicMate] Invalid JSON from model, retrying:", repr(e))
            resp = client_oa.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=400,
            )
            continue

        # Validate required fields: title + some lyric content
        if not candidate.get("song_title") or not candidate.get("lyric_lines"):
            print("[MicMate] Missing song_title or lyric_lines, retrying…")
            resp = client_oa.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=400,
            )
            continue

        data = candidate
        break

    if data is None:
        raise RuntimeError(
            "Model failed to return valid song JSON after multiple attempts."
        )

    # ---- Normalisation & safety ----

    song_title = str(data.get("song_title", "")).strip()
    artist = str(data.get("artist", "")).strip()
    lyric_lines = data.get("lyric_lines") or []
    hints_raw = data.get("hints") or []
    acc_title = data.get("acceptable_title_answers") or []
    acc_artist = data.get("acceptable_artist_answers") or []

    if not isinstance(lyric_lines, list):
        lyric_lines = [str(lyric_lines)]
    if not isinstance(hints_raw, list):
        hints_raw = [str(hints_raw)]
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
    hints_list = norm_list(hints_raw)[:3]

    # Enforce the 8-words / 90-char limit on lyrics only
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
        print("[MicMate] Missing title/lyrics after normalisation.")
        raise RuntimeError("Model response missing song_title or lyric_lines")

    return SongRound(
        song_title=song_title,
        artist=artist or "Unknown",
        lyric_lines=safe_lines,
        acceptable_titles=acc_title,
        acceptable_artists=acc_artist,
        hints=hints_list,
    )

# ------------- GAME LOGIC -------------
async def play_single_level(
    channel: discord.TextChannel,
    level: int,
    total_levels: int,
    scores: Dict[int, int],
    last_song: Optional[SongRound] = None,
    used_titles: Optional[set] = None,
    genre: Optional[str] = None,
    year: Optional[str] = None,
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

        while attempts < 25:  # a few extra tries
            candidate = await generate_song_round(
                last_song=last_song,
                used_titles=used_titles,
                genre=genre,
                year=year,
            )
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
        "Mode: Guess the **TITLE** or **ARTIST**.\n"
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
                    and (
                        not m.author.bot
                        or m.content.startswith("m.pass")  # allow our synthetic pass message
                    )
                ),
            )
        except asyncio.TimeoutError:
            break


        content = msg.content.lower().strip()

        # Handle pass via plain message "m.pass"
        if content.startswith("m.pass"):
            used = passes_used.get(channel.id, 0)
            if used >= 3:
                await channel.send("🚫 No passes left.")
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
                 "\n"  # <-- added blank line
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
    genre: Optional[str] = None,
    year: Optional[str] = None,
):
    """Main game loop for one Mic session in a channel."""
    active_games[channel.id] = True
    scores: Dict[int, int] = {}
    last_song: Optional[SongRound] = None

    # track used titles in this session
    used_titles: set[str] = set()

    # reset per-game counters
    hints_used[channel.id] = 0
    passes_used[channel.id] = 0
    current_song.pop(channel.id, None)

    await channel.send("🎮 **Mic game starting!**")

    level = 1

    try:
        while True:
            if not active_games.get(channel.id, False):
                break

            winner_id, this_song, passed_flag = await play_single_level(
                channel,
                level,
                total_levels,
                scores,
                last_song=last_song,
                used_titles=used_titles,
                genre=genre,
                year=year,
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
            # First wait a short time, then show Team Ranking,
            # then wait the remainder before the next song.
            await asyncio.sleep(RANK_DELAY)

            final_round = (total_levels > 0 and level >= total_levels)

            await show_ranking(
                channel,
                scores,
                next_level=level + 1,
                total_levels=total_levels,
                final=final_round,
            )

            if final_round:
                break

            # Wait the remaining time before next lyrics embed
            remaining_break = max(0, BREAK_TIME - RANK_DELAY)
            if remaining_break > 0:
                await asyncio.sleep(remaining_break)

            level += 1


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
        await channel.send("🚫 No hints left.")
        return

    hints_used[channel.id] = used + 1
    idx = used  # 0-based: 0,1,2

    # ---------- NEW HINT LOGIC ----------
    # Use non-lyric hints that were generated by OpenAI
    hint_list = getattr(song, "hints", None) or []

    if idx < len(hint_list):
        # Use the pre-generated hint (these should NOT quote lyrics)
        text = f"Hint {hints_used[channel.id]}/3: {hint_list[idx]}"
    else:
        # Fallback: structural hint from the TITLE, not from the lyrics
        title = song.song_title.strip()
        title_words = title.split()
        word_count = len(title_words)
        first_letter = title[0] if title else "?"
        last_letter = title[-1] if title else "?"

        text = (
            f"Hint {hints_used[channel.id]}/3: "
            f"The title has **{word_count}** word(s), starts with **{first_letter}** "
            f"and ends with **{last_letter}**."
        )
    # -----------------------------------

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

    # ✅ Check that a game and a round are actually running
    if not active_games.get(channel.id, False):
        await interaction.response.send_message(
            "There is no active Mic game in this channel.",
            ephemeral=True,
        )
        return

    if current_song.get(channel.id) is None:
        await interaction.response.send_message(
            "There is no active round to use a hint on.",
            ephemeral=True,
        )
        return

    # If we got here, it’s safe to use a hint
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
    genre="Optional genre (pop, rock, kpop, rnb, etc.)",
    year="Optional year or decade (e.g. 1980s, 90s, 2016)",
)
async def mic_slash(
    interaction: discord.Interaction,
    genre: Optional[str] = None,
    year: Optional[str] = None,
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

    # 0 = infinite levels (until someone fails / time runs out)
    total_levels = 0

    await interaction.response.send_message(
        "Starting Mic game ...",
        ephemeral=True,
    )

    bot.loop.create_task(
        run_mic_game(channel, total_levels, genre=genre, year=year)
    )

@bot.command(name="mic")
async def mic_prefix(ctx: commands.Context, rounds: Optional[int] = None):
    channel = ctx.channel
    if not isinstance(channel, discord.TextChannel):
        await ctx.reply("Please use this in a normal text channel.", mention_author=False)
        return

    if active_games.get(channel.id, False):
        await ctx.reply("There is already a Mic game running in this channel.", mention_author=False)
        return

    # None = infinite, number = cap
    if rounds is None:
        # no number given → infinite
        total_levels = 0
    else:
        try:
            # number given → cap at that number
            total_levels = int(rounds)
        except (TypeError, ValueError):
            # if someone types nonsense, just fall back to infinite
            total_levels = 0


    await ctx.reply("Starting Mic game ...", mention_author=False)
    bot.loop.create_task(run_mic_game(channel, total_levels))


# ------------- DOODLE GAME (GUESS THAT DOODLE) -------------

# 1) Chat → chooses the doodle concept + acceptable answers
# 2) Images → draws the doodle

DOODLE_ROUND_TIME = 30  # seconds to guess the doodle


@dataclass
class DoodleRound:
    word: str
    acceptable_answers: List[str]
    channel_id: int
    message_id: Optional[int] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    time_limit: int = DOODLE_ROUND_TIME
    winner_id: Optional[int] = None


# One active doodle round per channel
active_doodles: Dict[int, DoodleRound] = {}


def _norm_word(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _is_correct_doodle_guess(guess: str, round_obj: DoodleRound) -> bool:
    """
    Check guess against the main word AND acceptable answer variations.
    """
    g = _norm_word(guess)
    if not g:
        return False

    # Normalised acceptable answers
    acc = { _norm_word(a) for a in round_obj.acceptable_answers if a }

    # Direct match with any acceptable variation
    if g in acc:
        return True

    # Also compare to the base word
    base = _norm_word(round_obj.word)
    if g == base:
        return True

    # Allow word appearing in a longer sentence, e.g. "someone is running fast"
    if base in g.split():
        return True

    # Fuzzy similarity vs base word
    ratio = difflib.SequenceMatcher(None, g, base).ratio()
    return ratio >= 0.8


async def generate_doodle_concept() -> Tuple[str, List[str]]:
    """
    Ask OpenAI (chat) for a doodle concept and acceptable answers.
    Returns (word, acceptable_answers).
    """
    prompt = """
You are helping with a Discord "Guess That Doodle" game.

Pick ONE simple, visual, family-friendly concept that is easy to draw
as a doodle and easy to guess. It can be:
- a verb phrase like "playing guitar", "sleeping", "riding a bike"
- or a simple object like "umbrella", "pizza", "headphones"

Avoid abstract ideas. It must be something clearly drawable and guessable.

Return ONLY a compact JSON object, with this exact structure:

{
  "word": "...",
  "acceptable_answers": ["...", "...", "..."]
}

Rules:
- "word" should be the main concept used for the doodle prompt.
- "acceptable_answers" should include:
  - the word itself
  - small variations or synonyms
  - very short phrases players might type to guess it
- Do not include any explanation or text outside the JSON object.
"""

    attempts = 0
    data = None

    while attempts < 5 and data is None:
        attempts += 1
        resp = client_oa.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=200,
        )

        text = (resp.choices[0].message.content or "").strip()

        # Strip ```json fences if present
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            candidate = json.loads(text)
        except Exception as e:
            print("[MicMate-Doodle] Invalid JSON from model, retrying:", repr(e))
            continue

        if not candidate.get("word"):
            print("[MicMate-Doodle] Missing 'word' in doodle JSON, retrying…")
            continue

        data = candidate

    if data is None:
        raise RuntimeError("Model failed to return valid doodle JSON after multiple attempts.")

    word = str(data.get("word", "")).strip()
    acceptable = data.get("acceptable_answers") or []

    if not isinstance(acceptable, list):
        acceptable = [str(acceptable)]

    cleaned: List[str] = []
    for v in acceptable:
        s = str(v).strip()
        if s:
            cleaned.append(s)

    # Ensure the main word is in the acceptable answers
    if word and word not in cleaned:
        cleaned.append(word)

    if not word:
        raise RuntimeError("Doodle concept missing 'word' after normalisation.")

    return word, cleaned


def _build_doodle_prompt(word: str) -> str:
    """
    Prompt for the image model – keep it simple, doodle-style, no text.
    """
    return (
        f"Simple black-and-white doodle of the concept '{word}'. "
        "Cartoon style, minimal background, no text, no letters, no numbers. "
        "Make the drawing clear and easy to guess."
    )


def generate_doodle_image_sync(word: str) -> bytes:
    """
    Blocking call to OpenAI image API.
    Runs in a thread via asyncio.to_thread().
    Returns raw PNG bytes.
    Uses the gpt-image-1 model (which your account supports).
    """
    prompt = _build_doodle_prompt(word)
    print("[MicMate-Doodle] Generating image with gpt-image-1")

    img = client_oa.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="512x512",
    )

    # If this fails for any reason, the exception will bubble up
    # to start_doodle_round where we already catch and show it.
    b64_data = img.data[0].b64_json
    image_bytes = base64.b64decode(b64_data)
    print("[MicMate-Doodle] Image generation OK.")
    return image_bytes


    raise RuntimeError("Unknown error in generate_doodle_image_sync")


async def start_doodle_round(channel: discord.TextChannel):
    """
    Runs a single doodle round in the given channel.
    """
    # Don’t allow a doodle if one is already running there
    if channel.id in active_doodles:
        await channel.send("There is already a Doodle round running in this channel.")
        return

    # Optional: block doodle while Mic game is running in this channel
    if active_games.get(channel.id, False):
        await channel.send(
            "There is an active Mic game in this channel. "
            "Finish that first before starting Doodle."
        )
        return

    # 1) Get the doodle concept from OpenAI (chat)
    try:
        word, acceptable = await generate_doodle_concept()
    except Exception as e:
        print("[MicMate-Doodle] Error generating doodle concept:", repr(e))
        await channel.send("⚠️ I couldn't think of a doodle idea this time. Try again in a bit.")
        return

  
    # 2) Generate the doodle image from OpenAI (images)
    try:
        image_bytes = await asyncio.to_thread(generate_doodle_image_sync, word)
    except Exception as e:
        print("[MicMate-Doodle] Error generating doodle image:", repr(e))
        await channel.send(f"⚠️ Doodle failed: `{e}`")
        return


    file = discord.File(io.BytesIO(image_bytes), filename="doodle.png")

    embed = discord.Embed(
        title="🎨 Guess That Doodle!",
        description=(
            f"Type your guess in chat.\n"
            f"You have **{DOODLE_ROUND_TIME} seconds** ⏱️"
        ),
        color=discord.Color.blurple(),
    )
    embed.set_footer(text="First correct answer wins.")

    msg = await channel.send(embed=embed, file=file)

    round_obj = DoodleRound(
        word=word,
        acceptable_answers=acceptable,
        channel_id=channel.id,
        message_id=msg.id,
    )
    active_doodles[channel.id] = round_obj

    ends_at = datetime.now(timezone.utc) + timedelta(seconds=DOODLE_ROUND_TIME)
    winner_msg: Optional[discord.Message] = None

    while True:
        remaining = (ends_at - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            break

        try:
            guess_msg: discord.Message = await bot.wait_for(
                "message",
                timeout=remaining,
                check=lambda m: (
                    m.channel.id == channel.id
                    and not m.author.bot
                    and m.content.strip()
                ),
            )
        except asyncio.TimeoutError:
            break

        if _is_correct_doodle_guess(guess_msg.content, round_obj):
            round_obj.winner_id = guess_msg.author.id
            winner_msg = guess_msg
            break

    # Clean up the active round
    active_doodles.pop(channel.id, None)

    if round_obj.winner_id is not None and winner_msg is not None:
        try:
            await winner_msg.add_reaction("🖼️")
        except discord.HTTPException:
            pass

        answer_embed = discord.Embed(
            title="✅ Doodle guessed!",
            description=(
                f"**Answer:** `{round_obj.word}`\n\n"
                f"**Winner:** <@{round_obj.winner_id}>"
            ),
            color=discord.Color.green(),
        )
        await channel.send(embed=answer_embed)
    else:
        answer_embed = discord.Embed(
            title="⏰ Time's up!",
            description=(
                "No one guessed the doodle in time.\n\n"
                f"The answer was: `{round_obj.word}`"
            ),
            color=discord.Color.red(),
        )
        await channel.send(embed=answer_embed)


# --------- SLASH & PREFIX COMMANDS FOR DOODLE ---------


@bot.tree.command(
    name="doodle",
    description="Start a single-round 'Guess That Doodle' game (OpenAI-powered)."
)
async def doodle_slash(interaction: discord.Interaction):
    channel = interaction.channel
    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message(
            "Please use this in a normal text channel.",
            ephemeral=True,
        )
        return

    if channel.id in active_doodles:
        await interaction.response.send_message(
            "There is already a Doodle round running in this channel.",
            ephemeral=True,
        )
        return

    if active_games.get(channel.id, False):
        await interaction.response.send_message(
            "There is an active Mic game in this channel. Finish that first before starting Doodle.",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        "Starting a Doodle round ...",
        ephemeral=True,
    )

    bot.loop.create_task(start_doodle_round(channel))


@bot.command(name="doodle")
async def doodle_prefix(ctx: commands.Context):
    channel = ctx.channel
    if not isinstance(channel, discord.TextChannel):
        await ctx.reply("Please use this in a normal text channel.", mention_author=False)
        return

    if channel.id in active_doodles:
        await ctx.reply("There is already a Doodle round running in this channel.", mention_author=False)
        return

    if active_games.get(channel.id, False):
        await ctx.reply("There is an active Mic game in this channel. Finish that first before starting Doodle.", mention_author=False)
        return

    await ctx.reply("Starting a Doodle round ...", mention_author=False)
    bot.loop.create_task(start_doodle_round(channel))

# ------------- RUN -------------

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
