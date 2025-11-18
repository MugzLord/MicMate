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
    raise RuntimeError("OPENAI_API_KEY env var not set")

client_oa = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

ANSWER_TIME_LIMIT = 60  # seconds

intents = discord.Intents.default()
intents.message_content = True  # needed to read answers

# Allow both "!" and "m." prefixes, plus mention
bot = commands.Bot(
    command_prefix=commands.when_mentioned_or("!", "m."),
    intents=intents
)


# ------------- STATE -------------

@dataclass
class KaraokeRound:
    channel_id: int
    message_id: int
    song_title: str
    artist: str
    lyric_hints: List[str] = field(default_factory=list)
    acceptable_title_answers: List[str] = field(default_factory=list)
    acceptable_artist_answers: List[str] = field(default_factory=list)
    clue: str = ""
    mode: str = "either"  # "title", "artist", "either"
    is_active: bool = True
    ends_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
        + timedelta(seconds=ANSWER_TIME_LIMIT)
    )
    winner_id: Optional[int] = None
    genre: Optional[str] = None
    hints_used: int = 0        # shared per game
    passes_used: int = 0       # shared per game


# one active round per channel
active_rounds: Dict[int, KaraokeRound] = {}

# per-channel scoreboard: {channel_id: {user_id: score}}
scores_by_channel: Dict[int, Dict[int, int]] = {}

# per-channel solved streak: how many songs in a row were guessed correctly
streak_by_channel: Dict[int, int] = {}


# ------------- OPENAI HELPERS -------------

async def generate_song_round(genre: Optional[str]) -> Dict:
    """
    Uses OpenAI to generate a song round:
    - song_title
    - artist
    - lyric_hints (up to 3 very short lines)
    - clue (descriptive)
    - acceptable_title_answers
    - acceptable_artist_answers
    """

    genre_text = f" in the {genre} genre" if genre else ""

    prompt = f"""
You are powering a Discord karaoke guessing game.

Pick a well-known, globally recognisable song{genre_text}.

Return ONLY a compact JSON object with this exact structure:

{{
  "song_title": "...",
  "artist": "...",
  "lyric_hints": ["...", "...", "..."],
  "clue": "...",
  "acceptable_title_answers": ["...", "..."],
  "acceptable_artist_answers": ["...", "..."]
}}

Rules:
- "lyric_hints" must contain 1 to 3 very short lyric-style hints:
  - Each hint MUST be 8 words or fewer.
  - The TOTAL characters across all hints MUST stay safely under 90 characters.
  - Do NOT output full verses or long passages.
- It is OK if hints resemble real lyrics, as long as they stay under the above limits.
- "clue" should describe the song (theme, mood, era, scenario, style) but must NOT contain long lyric quotes.
- In "acceptable_title_answers":
  - include sensible variations of the song title (title alone, title + artist, common short forms).
- In "acceptable_artist_answers":
  - include reasonable variations of the artist name (full name, common short name).
- Do not include any explanation or text outside the JSON object.
"""

    resp = client_oa.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_output_tokens=300,
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
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenAI returned invalid JSON: {text[:200]}") from e

    song_title = str(data.get("song_title", "")).strip()
    artist = str(data.get("artist", "")).strip()
    lyric_hints = data.get("lyric_hints") or []
    clue = str(data.get("clue", "")).strip()
    acc_title = data.get("acceptable_title_answers") or []
    acc_artist = data.get("acceptable_artist_answers") or []

    if not isinstance(lyric_hints, list):
        lyric_hints = [str(lyric_hints)]

    if not isinstance(acc_title, list):
        acc_title = [str(acc_title)]
    if not isinstance(acc_artist, list):
        acc_artist = [str(acc_artist)]

    def normalise_list(str_list: List) -> List[str]:
        out = []
        for x in str_list:
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    # keep max 3 hints, each ≤8 words, total chars ≤90
    lyric_hints = normalise_list(lyric_hints)[:3]
    safe_hints: List[str] = []
    total_chars = 0
    for hint in lyric_hints:
        words = hint.split()
        hint_short = " ".join(words[:8])
        if len(hint_short) > 90:
            hint_short = hint_short[:90]
        total_chars += len(hint_short)
        if total_chars > 90:
            break
        safe_hints.append(hint_short)

    acc_title = [s.lower().strip() for s in normalise_list(acc_title)]
    acc_artist = [s.lower().strip() for s in normalise_list(acc_artist)]

    if not song_title or not clue:
        raise RuntimeError("OpenAI response missing song_title or clue")

    return {
        "song_title": song_title,
        "artist": artist or "Unknown",
        "lyric_hints": safe_hints,
        "clue": clue,
        "acceptable_title_answers": acc_title,
        "acceptable_artist_answers": acc_artist,
    }


# ------------- HELPERS -------------

def _normalise_answer(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _is_correct_guess(round_obj: KaraokeRound, guess: str) -> bool:
    """
    Checks correctness based on round_obj.mode:
    - "title": only title-based answers
    - "artist": only artist-based answers
    - "either": either title or artist counts
    """
    g = _normalise_answer(guess)
    if not g:
        return False

    title_norm = _normalise_answer(round_obj.song_title)
    artist_norm = _normalise_answer(round_obj.artist)

    def match_any(candidates: List[str]) -> bool:
        for ans in candidates:
            if g == _normalise_answer(ans):
                return True
        return False

    def contains_term(term: str) -> bool:
        return bool(term and term in g)

    title_ok = match_any(round_obj.acceptable_title_answers) or contains_term(title_norm)
    artist_ok = match_any(round_obj.acceptable_artist_answers) or contains_term(artist_norm)

    if round_obj.mode == "title":
        return title_ok
    elif round_obj.mode == "artist":
        return artist_ok
    else:  # "either"
        return title_ok or artist_ok


def build_round_description(round_obj: KaraokeRound) -> str:
    """Builds the embed description including hints used, clue and mode."""
    mode_text = {
        "title": "Guess the TITLE",
        "artist": "Guess the ARTIST",
        "either": "Guess the TITLE or ARTIST",
    }.get(round_obj.mode, "Guess the TITLE or ARTIST")

    parts: List[str] = []

    if round_obj.hints_used > 0 and round_obj.lyric_hints:
        shown_hints = round_obj.lyric_hints[: round_obj.hints_used]
        hint_lines = [f"• “{h}”" for h in shown_hints]
        parts.append("**Lyric hints:**\n" + "\n".join(hint_lines))

    parts.append(f"**Clue:** {round_obj.clue}")
    parts.append(
        f"\n**Mode:** {mode_text}\n"
        f"Guess in chat! You have **{ANSWER_TIME_LIMIT} seconds**.\n"
        "_(Full song + artist revealed at the end.)_\n"
        f"Hints used: `{round_obj.hints_used}/3`, Passes used: `{round_obj.passes_used}/3`"
    )

    return "\n\n".join(parts)


async def post_scoreboard(channel: discord.TextChannel):
    """
    Posts a scoreboard embed for the current channel.

    - Scores are cumulative across games until reset.
    - Streak (how many songs in a row were guessed correctly) is shown in the footer.
    """
    scores = scores_by_channel.get(channel.id, {})
    streak = streak_by_channel.get(channel.id, 0)

    emb = discord.Embed(
        title="🎶 Mic Scoreboard",
        color=discord.Color.gold()
    )

    if not scores:
        emb.description = "No scores yet. Win a round to get on the board!"
    else:
        sorted_scores = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        lines = []

        top_user_id, top_score = sorted_scores[0]
        emb.add_field(
            name="👑 Current Top Scorer",
            value=f"<@{top_user_id}> — `{top_score}` point(s)",
            inline=False,
        )

        for idx, (user_id, score) in enumerate(sorted_scores, start=1):
            lines.append(f"**{idx}.** <@{user_id}> — `{score}` point(s)")

        emb.add_field(
            name="Full Ranking",
            value="\n".join(lines),
            inline=False
        )

    emb.set_footer(text=f"Solved streak in this channel: {streak}")
    await channel.send(embed=emb)


async def start_new_song_after_pass(old_round: KaraokeRound, channel: discord.TextChannel):
    """
    Called when /mic_pass or m.pass is used:
    - Keeps scores & streak
    - Keeps hints_used & passes_used counters
    - Loads a NEW song into the same message
    - Starts a fresh run_karaoke_round with the new round object
    """
    try:
        song_data = await generate_song_round(old_round.genre)
    except Exception as e:
        print(f"Error generating new song after pass: {e}")
        await channel.send("⚠️ I couldn't load a new song right now. Try `/mic` again later.")
        return

    song_title = song_data["song_title"]
    artist = song_data["artist"]
    lyric_hints = song_data.get("lyric_hints") or []
    clue = song_data["clue"]
    acc_title = song_data.get("acceptable_title_answers") or []
    acc_artist = song_data.get("acceptable_artist_answers") or []

    ends_at = datetime.now(timezone.utc) + timedelta(seconds=ANSWER_TIME_LIMIT)

    try:
        msg = await channel.fetch_message(old_round.message_id)
    except discord.NotFound:
        await channel.send("⚠️ Original mic message was lost, starting a fresh game.")
        return

    new_round = KaraokeRound(
        channel_id=channel.id,
        message_id=msg.id,
        song_title=song_title,
        artist=artist,
        lyric_hints=lyric_hints,
        acceptable_title_answers=acc_title,
        acceptable_artist_answers=acc_artist,
        clue=clue,
        mode=old_round.mode,
        is_active=True,
        ends_at=ends_at,
        winner_id=None,
        genre=old_round.genre,
        hints_used=old_round.hints_used,   # carry over counters
        passes_used=old_round.passes_used, # carry over counters
    )

    emb = discord.Embed(
        title="🎤 Mic Game – New Song!",
        description=build_round_description(new_round),
        color=discord.Color.blurple()
    )
    if new_round.genre:
        emb.add_field(name="Genre", value=new_round.genre, inline=True)
    emb.set_footer(text=f"⏱ Time left: {ANSWER_TIME_LIMIT}s")

    await msg.edit(embed=emb)

    active_rounds[channel.id] = new_round
    bot.loop.create_task(run_karaoke_round(new_round, channel, msg))


# ------------- GAME LOGIC -------------

async def run_karaoke_round(round_obj: KaraokeRound, channel: discord.TextChannel, msg: discord.Message):
    """
    Main game loop for one song.
    - Ends on correct guess or timeout.
    - If all 3 hints + 3 passes used and still no winner on timeout → full reset.
    """
    round_obj.message_id = msg.id
    active_rounds[channel.id] = round_obj

    streak_by_channel.setdefault(channel.id, 0)

    bot.loop.create_task(update_embed_timer(round_obj, channel))
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
                break

            if not round_obj.is_active:
                break

            if _is_correct_guess(round_obj, guess_msg.content):
                round_obj.is_active = False
                round_obj.winner_id = guess_msg.author.id

                scores = scores_by_channel.setdefault(channel.id, {})
                scores[guess_msg.author.id] = scores.get(guess_msg.author.id, 0) + 1

                streak_by_channel[channel.id] = streak_by_channel.get(channel.id, 0) + 1

                try:
                    current_msg = await channel.fetch_message(round_obj.message_id)
                except discord.NotFound:
                    current_msg = None

                if current_msg and current_msg.embeds:
                    emb = current_msg.embeds[0]
                    mode_text = {
                        "title": "Title",
                        "artist": "Artist",
                        "either": "Title or Artist",
                    }.get(round_obj.mode, "Title or Artist")
                    emb.color = discord.Color.green()
                    emb.title = "🎤 Mic Round – We have a winner!"
                    emb.description = (
                        f"**Mode:** Guess the {mode_text}\n"
                        f"**Song:** {round_obj.song_title} – {round_obj.artist}\n"
                        f"**Winner:** {guess_msg.author.mention}\n\n"
                        "Start another `/mic` round anytime."
                    )
                    emb.set_footer(text="Round complete.")
                    await current_msg.edit(embed=emb)

                await channel.send(
                    f"✅ {guess_msg.author.mention} got it right! "
                    f"The song was **{round_obj.song_title}** by **{round_obj.artist}**.\n"
                    f"(Solved streak in this channel: `{streak_by_channel[channel.id]}`)"
                )

                await post_scoreboard(channel)
                break

        # timeout path
        if round_obj.is_active:
            round_obj.is_active = False
            try:
                current_msg = await channel.fetch_message(round_obj.message_id)
            except discord.NotFound:
                current_msg = None

            if current_msg and current_msg.embeds:
                emb = current_msg.embeds[0]
                mode_text = {
                    "title": "Title",
                    "artist": "Artist",
                    "either": "Title or Artist",
                }.get(round_obj.mode, "Title or Artist")
                emb.color = discord.Color.red()
                emb.title = "⏰ Time's up!"
                emb.description = (
                    f"**Mode:** Guess the {mode_text}\n"
                    f"No one guessed it in time.\n\n"
                    f"**Song:** {round_obj.song_title} – {round_obj.artist}\n\n"
                    f"Hints used: `{round_obj.hints_used}/3`, Passes used: `{round_obj.passes_used}/3`"
                )
                emb.set_footer(text="Round complete.")
                await current_msg.edit(embed=emb)

            await channel.send(
                f"⏰ Time's up! The song was **{round_obj.song_title}** by **{round_obj.artist}**."
            )

            streak_by_channel[channel.id] = 0

            # if 3 hints AND 3 passes used, nuke the game
            if round_obj.hints_used >= 3 and round_obj.passes_used >= 3:
                scores_by_channel[channel.id] = {}
                await channel.send(
                    "🧨 All **3 hints** and **3 passes** were used and still no one got it.\n"
                    "__Mic game has been reset in this channel.__ Scoreboard wiped."
                )

            await post_scoreboard(channel)

    finally:
        if active_rounds.get(channel.id) is round_obj:
            active_rounds.pop(channel.id, None)


async def update_embed_timer(round_obj: KaraokeRound, channel: discord.TextChannel):
    """Updates the footer with remaining time while the round is active."""
    await asyncio.sleep(1)

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

        await asyncio.sleep(5)


async def bump_reminder(round_obj: KaraokeRound, channel: discord.TextChannel):
    """Halfway reminder with jump link so embed doesn't get lost."""
    await asyncio.sleep(ANSWER_TIME_LIMIT / 2)
    if not round_obj.is_active:
        return

    try:
        msg = await channel.fetch_message(round_obj.message_id)
    except discord.NotFound:
        return

    await channel.send(
        f"🎵 Mic round still live! You’ve got about "
        f"**{int((round_obj.ends_at - datetime.now(timezone.utc)).total_seconds())}s** left.\n"
        f"Jump to the question: {msg.jump_url}"
    )


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

@bot.tree.command(name="mic", description="Start a 60-second karaoke guessing game in this channel.")
@app_commands.describe(
    mode="What are players guessing?",
    genre="Optional: pick a genre like pop, rock, kpop, rnb, etc."
)
@app_commands.choices(
    mode=[
        app_commands.Choice(name="Guess the title", value="title"),
        app_commands.Choice(name="Guess the artist", value="artist"),
        app_commands.Choice(name="Guess title or artist", value="either"),
    ]
)
async def mic_command(
    interaction: discord.Interaction,
    mode: app_commands.Choice[str],
    genre: Optional[str] = None
):
    channel = interaction.channel
    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message(
            "Please use this command in a normal text channel.", ephemeral=True
        )
        return

    if channel.id in active_rounds and active_rounds[channel.id].is_active:
        await interaction.response.send_message(
            "There is already a mic round running in this channel. Wait for it to finish first.",
            ephemeral=True
        )
        return

    mode_value = mode.value  # "title", "artist", "either"

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
    lyric_hints = song_data.get("lyric_hints") or []
    clue = song_data["clue"]
    acc_title = song_data.get("acceptable_title_answers") or []
    acc_artist = song_data.get("acceptable_artist_answers") or []

    ends_at = datetime.now(timezone.utc) + timedelta(seconds=ANSWER_TIME_LIMIT)

    round_obj = KaraokeRound(
        channel_id=channel.id,
        message_id=0,  # set after send
        song_title=song_title,
        artist=artist,
        lyric_hints=lyric_hints,
        acceptable_title_answers=acc_title,
        acceptable_artist_answers=acc_artist,
        clue=clue,
        mode=mode_value,
        is_active=True,
        ends_at=ends_at,
        winner_id=None,
        genre=genre,
        hints_used=0,
        passes_used=0,
    )

    emb = discord.Embed(
        title="🎤 Mic – Karaoke Guessing Game",
        description=build_round_description(round_obj),
        color=discord.Color.blurple()
    )
    if genre:
        emb.add_field(name="Genre", value=genre, inline=True)
    emb.set_footer(text=f"⏱ Time left: {ANSWER_TIME_LIMIT}s")

    msg = await interaction.followup.send(embed=emb, wait=True)
    round_obj.message_id = msg.id

    bot.loop.create_task(run_karaoke_round(round_obj, channel, msg))


@bot.tree.command(
    name="mic_hint",
    description="Use one of the shared hints for the current mic game (max 3 per game)."
)
async def mic_hint_command(interaction: discord.Interaction):
    channel = interaction.channel
    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message(
            "Use this in the same text channel as the game.", ephemeral=True
        )
        return

    round_obj = active_rounds.get(channel.id)
    if not round_obj or not round_obj.is_active:
        await interaction.response.send_message(
            "There is no active mic round in this channel.", ephemeral=True
        )
        return

    if round_obj.hints_used >= 3:
        await interaction.response.send_message(
            "All **3 hints** for this game have already been used.", ephemeral=True
        )
        return

    if round_obj.hints_used >= len(round_obj.lyric_hints):
        await interaction.response.send_message(
            "No more hints are available for this song.", ephemeral=True
        )
        return

    round_obj.hints_used += 1

    try:
        msg = await channel.fetch_message(round_obj.message_id)
    except discord.NotFound:
        await interaction.response.send_message(
            "I couldn't find the mic message to update.", ephemeral=True
        )
        return

    if msg.embeds:
        emb = msg.embeds[0]
        emb.description = build_round_description(round_obj)
        try:
            await msg.edit(embed=emb)
        except discord.HTTPException:
            pass

    await interaction.response.send_message(
        f"Hint `{round_obj.hints_used}/3` revealed for this game.", ephemeral=True
    )
    await channel.send(
        f"💡 A hint has been used! (`{round_obj.hints_used}/3`) Check the mic embed for updated hints."
    )


@bot.tree.command(
    name="mic_pass",
    description="Skip the current song (shared 3 passes per game)."
)
async def mic_pass_command(interaction: discord.Interaction):
    channel = interaction.channel
    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message(
            "Use this in the same text channel as the game.", ephemeral=True
        )
        return

    round_obj = active_rounds.get(channel.id)
    if not round_obj or not round_obj.is_active:
        await interaction.response.send_message(
            "There is no active mic round in this channel.", ephemeral=True
        )
        return

    if round_obj.passes_used >= 3:
        await interaction.response.send_message(
            "All **3 passes** for this game have already been used.", ephemeral=True
        )
        return

    round_obj.passes_used += 1
    used = round_obj.passes_used

    round_obj.is_active = False

    await interaction.response.send_message(
        f"You used a pass. (`{used}/3` for this game)", ephemeral=True
    )

    await channel.send(
        f"⏭️ Song passed! (`{used}/3` passes used this game)\n"
        "Loading a new song in the same game..."
    )

    await start_new_song_after_pass(round_obj, channel)


# ------------- PREFIX COMMANDS (m.hint / m.pass) -------------

@bot.command(name="hint")
async def m_hint(ctx: commands.Context):
    """Prefix version: m.hint"""
    channel = ctx.channel
    if not isinstance(channel, discord.TextChannel):
        await ctx.reply("Use this in the same text channel as the game.", mention_author=False)
        return

    round_obj = active_rounds.get(channel.id)
    if not round_obj or not round_obj.is_active:
        await ctx.reply("There is no active mic round in this channel.", mention_author=False)
        return

    if round_obj.hints_used >= 3:
        await ctx.reply("All **3 hints** for this game have already been used.", mention_author=False)
        return

    if round_obj.hints_used >= len(round_obj.lyric_hints):
        await ctx.reply("No more hints are available for this song.", mention_author=False)
        return

    round_obj.hints_used += 1

    try:
        msg = await channel.fetch_message(round_obj.message_id)
    except discord.NotFound:
        await ctx.reply("I couldn't find the mic message to update.", mention_author=False)
        return

    if msg.embeds:
        emb = msg.embeds[0]
        emb.description = build_round_description(round_obj)
        try:
            await msg.edit(embed=emb)
        except discord.HTTPException:
            pass

    await ctx.reply(f"Hint `{round_obj.hints_used}/3` revealed for this game.", mention_author=False)
    await channel.send(
        f"💡 A hint has been used! (`{round_obj.hints_used}/3`) Check the mic embed for updated hints."
    )


@bot.command(name="pass")
async def m_pass(ctx: commands.Context):
    """Prefix version: m.pass"""
    channel = ctx.channel
    if not isinstance(channel, discord.TextChannel):
        await ctx.reply("Use this in the same text channel as the game.", mention_author=False)
        return

    round_obj = active_rounds.get(channel.id)
    if not round_obj or not round_obj.is_active:
        await ctx.reply("There is no active mic round in this channel.", mention_author=False)
        return

    if round_obj.passes_used >= 3:
        await ctx.reply("All **3 passes** for this game have already been used.", mention_author=False)
        return

    round_obj.passes_used += 1
    used = round_obj.passes_used

    round_obj.is_active = False

    await ctx.reply(f"You used a pass. (`{used}/3` for this game)", mention_author=False)
    await channel.send(
        f"⏭️ Song passed! (`{used}/3` passes used this game)\n"
        "Loading a new song in the same game..."
    )

    await start_new_song_after_pass(round_obj, channel)


# ------------- RUN -------------

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
