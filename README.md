import asyncio
import textwrap
from typing import List
import os
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
import random
import aiohttp
import json
import time
from datetime import datetime, timedelta
import requests
from hide_game import run_grand_escape_round, get_status_message, HIDING_SPOTS

import logging

# Discord webhook URL for notifications
WEBHOOK_URL = "https://discord.com/api/webhooks/1444058241205014581/s9NxY2a2jn3GnKshVThgnNaQgpLV1WFcEWpCPGIgT9pQgaJxetdyrlz0hDrtPdp4MmMV"

# Load environment variables from .env
load_dotenv(dotenv_path='../.env')
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Validate runtime secrets early
if not DISCORD_BOT_TOKEN:
    raise RuntimeError("Missing DISCORD_BOT_TOKEN in environment. Set it in .env")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in environment. Set it in .env")

# Lazy import to avoid dependency import errors if not installed yet
try:
    from openai import OpenAI, OpenAIError
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    OpenAIError = None  # type: ignore

# Discord intents
intents = discord.Intents.default()
intents.message_content = True  # Required if you want to read message content for prefix commands
intents.members = True
intents.invites = True

# Create bot with prefix and application command tree for slash commands
bot = commands.Bot(command_prefix=commands.when_mentioned_or("j!"), intents=intents, help_command=None)

tree = bot.tree  # app_commands.CommandTree

# Helper: chunk long strings to respect Discord 2000 char message limit
MAX_DISCORD_MSG = 2000

def chunk_text(s: str, size: int = MAX_DISCORD_MSG) -> List[str]:
    chunks: List[str] = []
    s = s.replace("\r\n", "\n")
    while s:
        if len(s) <= size:
            chunks.append(s)
            break
        # Try to break on newline or space near boundary for nicer formatting
        cut = s.rfind("\n", 0, size)
        if cut == -1:
            cut = s.rfind(" ", 0, size)
        if cut == -1:
            cut = size
        chunks.append(s[:cut])
        s = s[cut:].lstrip(" \n")
    return chunks

# AI client factory
_client = None

def get_ai_client():
    global _client
    if _client is None:
        if OpenAI is None:
            raise RuntimeError("openai package is not available. Install dependencies with 'pip install -r requirements.txt'")
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
    return _client

ASYNC_TIMEOUT = 60

async def ask_openai(prompt: str, user: str) -> str:
    try:
        return await asyncio.wait_for(asyncio.to_thread(_ask_openai_sync, prompt, user), timeout=ASYNC_TIMEOUT)
    except asyncio.TimeoutError:
        return f"Sorry {user}, the AI took too long to respond. Try again."

def _ask_openai_sync(prompt: str, user: str) -> str:
    client = get_ai_client()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant in a Discord server. Keep answers concise unless asked for detail."},
                {"role": "user", "content": f"User: {user}\nQuestion: {prompt}"},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        content = resp.choices[0].message.content or "(No content returned)"
        return content.strip()
    except OpenAIError as e:
        if "quota" in str(e).lower() or "exceeded" in str(e).lower() or "insufficient_quota" in str(e):
            return f"Sorry {user}, the AI service is currently unavailable due to quota limits. Please try again later."
        return f"Error from AI service: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# AFK storage
afk_users = {}

# Quiet mode to stop bot from sending messages
quiet_mode = False

# Game storage
game_state = {}

# Invite tracking storage (in-memory)
invite_tracking = {}  # guild_id -> {'invites': {code: invite_data}, 'joins': [join_data]}

# Announcement channel storage (in-memory)
announce_channels = {}  # guild_id -> channel_id

# Tester responses storage (in-memory)
tester_responses = {}  # user_id -> {'accepted': bool, 'timestamp': str}



# UI for tester notification
class TesterResponseView(discord.ui.View):
    def __init__(self, user_id: int):
        super().__init__(timeout=None)  # No timeout for persistent buttons
        self.user_id = user_id

    @discord.ui.button(label="Accept", style=discord.ButtonStyle.green, emoji="✅")
    async def accept_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("This is not for you!", ephemeral=True)
            return

        tester_responses[self.user_id] = {'accepted': True, 'timestamp': datetime.utcnow().isoformat()}
        embed = discord.Embed(
            title="✅ Accepted!",
            description="Thank you for accepting the app test request! Our developers will be in touch soon.",
            color=discord.Color.green()
        )
        await interaction.response.edit_message(embed=embed, view=None)

    @discord.ui.button(label="Decline", style=discord.ButtonStyle.red, emoji="❌")
    async def decline_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("This is not for you!", ephemeral=True)
            return

        tester_responses[self.user_id] = {'accepted': False, 'timestamp': datetime.utcnow().isoformat()}
        embed = discord.Embed(
            title="❌ Declined",
            description="You have declined the app test request. Thank you for your consideration!",
            color=discord.Color.red()
        )
        await interaction.response.edit_message(embed=embed, view=None)

# Owner-only check using OWNER_ID env var (if set). Falls back to commands.is_owner().
def owner_only():
    owner_id = os.getenv('OWNER_ID')
    if owner_id:
        def predicate(ctx):
            try:
                return ctx.author.id == int(owner_id)
            except Exception:
                return False
        return commands.check(predicate)
    # Fallback to default owner check (application owner)
    return commands.is_owner()

# Event: Bot is ready
@bot.event
async def on_ready():
    try:
        synced = await tree.sync()
        print(f"Synced {len(synced)} commands")
    except Exception as e:
        print(f"Slash command sync failed: {e}")
    print(f"Logged in as: {bot.user} (ID: {bot.user.id})")
    custom_text = "👑 Clash Royale"
    custom_activity = discord.CustomActivity(name=custom_text)
    await bot.change_presence(status=discord.Status.online, activity=custom_activity)

    # Fetch invites for tracking
    for guild in bot.guilds:
        try:
            invites = await guild.invites()
            invite_tracking[guild.id] = {'invites': {inv.code: {'inviter': inv.inviter.id if inv.inviter else None, 'uses': inv.uses} for inv in invites}, 'joins': []}
        except Exception as e:
            print(f"Failed to fetch invites for {guild.name}: {e}")

    # Send online announcement to configured channels
    for guild in bot.guilds:
        channel_id = announce_channels.get(guild.id)
        if channel_id:
            channel = guild.get_channel(channel_id)
            if channel:
                try:
                    embed = discord.Embed(
                        title="🤖 Bot Online",
                        description=f"{bot.user.mention} is now online and ready to serve!",
                        color=discord.Color.green()
                    )
                    embed.set_footer(text=f"Connected at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    await channel.send(embed=embed)
                except Exception as e:
                    print(f"Failed to send online announcement in {guild.name}: {e}")

# Event: Member joins the server
@bot.event
async def on_invite_create(invite):
    guild_id = invite.guild.id
    if guild_id not in invite_tracking:
        invite_tracking[guild_id] = {'invites': {}, 'joins': []}
    invite_tracking[guild_id]['invites'][invite.code] = {'inviter': invite.inviter.id if invite.inviter else None, 'uses': invite.uses}

@bot.event
async def on_invite_delete(invite):
    guild_id = invite.guild.id
    if guild_id in invite_tracking and invite.code in invite_tracking[guild_id]['invites']:
        del invite_tracking[guild_id]['invites'][invite.code]

@bot.event
async def on_member_join(member):
    guild_id = member.guild.id
    if guild_id in invite_tracking:
        # Fetch current invites
        try:
            current_invites = await member.guild.invites()
            current_invites_dict = {inv.code: inv.uses for inv in current_invites}
            # Find which invite was used
            for code, data in invite_tracking[guild_id]['invites'].items():
                if code in current_invites_dict and current_invites_dict[code] > data['uses']:
                    # This invite was used
                    inviter_id = data['inviter']
                    invite_tracking[guild_id]['joins'].append({
                        'user_id': member.id,
                        'inviter_id': inviter_id,
                        'timestamp': datetime.utcnow().isoformat(),
                        'invite_code': code
                    })
                    # Update uses
                    invite_tracking[guild_id]['invites'][code]['uses'] = current_invites_dict[code]
                    break
        except Exception as e:
            print(f"Failed to track invite for {member}: {e}")

    # Check if quiet mode is enabled
    if quiet_mode:
        return  # Quiet mode, skip welcome message

    # Send DM to new member about app testing
    try:
        dm_message = (
            f"Hello {member.name},\n\n"
            "Welcome to the server! It looks like you joined via an invite, which means you might be interested in our **App Test Program**!\n"
            "Our developers will be in touch shortly with an invitation to the private testing server or additional instructions.\n"
            "Thank you for helping us improve our app!"
        )
        await member.send(dm_message)
    except discord.Forbidden:
        # User has DMs disabled, skip
        pass
    except Exception as e:
        print(f"Failed to send DM to {member}: {e}")

    channel = member.guild.system_channel
    if channel is not None:
        try:
            await channel.send(f'Welcome {member.mention} to the server!')
        except discord.HTTPException as e:
            if e.status == 429:
                # Rate limited, skip
                pass
            else:
                raise

@bot.event
async def on_message(message):
    # Ignore messages from bots (including itself)
    if message.author.bot:
        return

    # Check if user is AFK and remove status
    if message.author.id in afk_users:
        del afk_users[message.author.id]
        embed = discord.Embed(
            title="Welcome Back",
            description="Your AFK status has been removed.",
            color=discord.Color.green()
        )
        await message.channel.send(embed=embed)

    # Ensure other commands still process
    await bot.process_commands(message)

# Slash command: /ask
@tree.command(name="ask", description="Ask the AI a question")
@app_commands.describe(prompt="Your question or message for the AI")
async def ask(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True, ephemeral=False)
    # Safety: bound prompt length to avoid abuse
    prompt = prompt.strip()
    if len(prompt) > 4000:
        prompt = prompt[:4000]
    answer = await ask_openai(prompt, user=interaction.user.name)
    for part in chunk_text(answer):
        await interaction.followup.send(part)

# Slash command: /talk
@tree.command(name="talk", description="Repeat your message")
@app_commands.describe(message="Your message to repeat")
async def talk(interaction: discord.Interaction, message: str):
    # Safety: bound message length to avoid abuse
    message = message.strip()
    if len(message) > 4000:
        message = message[:4000]
    await interaction.response.send_message(message)

# Slash command: /help
@tree.command(name="help", description="Get help with bot commands")
async def help_command(interaction: discord.Interaction):
    help_text = """
**Available Commands:**
- `/ask <question>`: Ask the AI a question
- `/talk <message>`: Send the message in the prompt
- `/sync`: Sync all slash commands globally (admin only)
- `j!ask <question>`: Prefix command to ask the AI
- `j!talk <message>`: Prefix command to talk to the AI
- `/image <prompt>`: Generate an image
- `/game`: Play a game
- And many more...
    """
    await interaction.response.send_message(help_text, ephemeral=True)

# Slash command: /sync
@tree.command(name="sync", description="Sync all slash commands globally")
async def sync_commands(interaction: discord.Interaction):
    # Check if user has administrator permissions
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
        return

    await interaction.response.defer(thinking=True, ephemeral=True)
    try:
        synced = await tree.sync()
        await interaction.followup.send(f"Successfully synced {len(synced)} commands globally.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"Failed to sync commands: {e}", ephemeral=True)

# Slash Command: Ping
@bot.tree.command(name='ping', description='Responds with the bot\'s latency')
async def ping(interaction: discord.Interaction):
    # Defer the response to prevent timeout errors
    await interaction.response.defer()
    latency = round(bot.latency * 1000)
    await interaction.followup.send(f'Pong! Latency: {latency}ms')

# Slash Command: AI Chat
@bot.tree.command(name='ai', description='Chat with AI (supports text and optional image)')
async def ai(interaction: discord.Interaction, prompt: str, image_url: str = None):
    # Defer the response to prevent timeout errors
    await interaction.response.defer()

    try:
        client = get_ai_client()
        content = []
        if image_url:
            content.append({"type": "text", "text": prompt})
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        else:
            content = prompt

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
            temperature=0.7
        )

        ai_response = completion.choices[0].message.content.strip()
        await interaction.followup.send(ai_response)

    except Exception as e:
        embed = discord.Embed(
            title="AI Error",
            description=f"Failed to get AI response: {str(e)}",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed)

# Slash Command: Generate Image
@bot.tree.command(name='image', description='Generate an image using AI (free)')
@discord.app_commands.describe(
    prompt='The prompt for image generation',
    aspect_ratio='Aspect ratio for the image (default: 1:1)'
)
@discord.app_commands.choices(
    aspect_ratio=[
        discord.app_commands.Choice(name='1:1 (Square)', value='1:1'),
        discord.app_commands.Choice(name='16:9 (Landscape)', value='16:9'),
        discord.app_commands.Choice(name='9:16 (Portrait)', value='9:16'),
        discord.app_commands.Choice(name='21:9 (Ultra-wide)', value='21:9'),
        discord.app_commands.Choice(name='9:21 (Ultra-tall)', value='9:21')
    ]
)
async def image(interaction: discord.Interaction, prompt: str, aspect_ratio: str = "1:1"):
    # Defer the response to prevent timeout errors
    await interaction.response.defer()

    # Validate prompt length
    if len(prompt) > 1000:
        embed = discord.Embed(
            title="Invalid Prompt",
            description="Prompt must be 1000 characters or less.",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed)
        return

    try:
        # Map aspect ratios to Pollinations.ai dimensions
        aspect_map = {
            "1:1": (1024, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "21:9": (1024, 438),
            "9:21": (438, 1024)
        }
        width, height = aspect_map.get(aspect_ratio, (1024, 1024))

        # Use Pollinations.ai (free, no API key needed)
        image_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?width={width}&height={height}&model=flux"

        # Download the image
        response = requests.get(image_url, timeout=30)

        if response.status_code == 200:
            # The response content is the image binary
            image_data = response.content

            # Create a file-like object for Discord
            from io import BytesIO
            image_file = discord.File(BytesIO(image_data), filename="generated_image.jpeg")

            await interaction.followup.send(file=image_file)
        else:
            embed = discord.Embed(
                title="Image Generation Error",
                description=f"Failed to generate image: HTTP {response.status_code}",
                color=discord.Color.red()
            )
            await interaction.followup.send(embed=embed)

    except requests.exceptions.Timeout:
        embed = discord.Embed(
            title="Timeout Error",
            description="Image generation took too long. Please try again.",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed)
    except requests.exceptions.RequestException as e:
        embed = discord.Embed(
            title="Network Error",
            description=f"Failed to connect to Pollinations.ai: {str(e)}",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed)
    except Exception as e:
        embed = discord.Embed(
            title="Unexpected Error",
            description=f"An unexpected error occurred: {str(e)}",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed)

# Slash Command: Ship
@bot.tree.command(name='ship', description='Ship two users and rate their love compatibility')
async def ship(interaction: discord.Interaction, user1: discord.Member, user2: discord.Member):
    # Generate random love percentage with 1% chance of over 100%
    if random.random() < 0.01:  # 1% chance
        love_percentage = random.randint(101, 150)  # Over 100% for "over ship"
    else:
        love_percentage = random.randint(0, 100)

    # Determine compatibility message based on percentage
    if love_percentage > 100:
        compatibility_msg = "💫 OVER SHIP! Destiny has spoken! 💫"
    elif love_percentage >= 90:
        compatibility_msg = "💕 Perfect Match! Soulmates! 💕"
    elif love_percentage >= 70:
        compatibility_msg = "💖 Great Match! Love is in the air! 💖"
    elif love_percentage >= 50:
        compatibility_msg = "💗 Good Match! There's potential! 💗"
    elif love_percentage >= 30:
        compatibility_msg = "💓 It's Complicated... 💓"
    else:
        compatibility_msg = "💔 Not Meant to Be... 💔"

    # List of romantic images for variety
    romantic_images = [
        "https://i.imgur.com/8cNF0zQ.png",  # Hearts background
        "https://i.imgur.com/4Q3wX8L.png",  # Romantic couple
        "https://i.imgur.com/J8w7Z9K.png",  # Love birds
        "https://i.imgur.com/M5v8N2P.png",  # Sunset couple
        "https://i.imgur.com/R3tY7sL.png",  # Heart balloons
        "https://i.imgur.com/X9pW4mQ.png"   # Romantic dinner
    ]

    # Select random image
    selected_image = random.choice(romantic_images)

    # Create embed with random romantic image
    embed = discord.Embed(
        title=f"💘 Ship: {user1.name} x {user2.name} 💘",
        description=f"{user1.mention} ❤️ {user2.mention}\n\n**Love Compatibility: {love_percentage}%**\n{compatibility_msg}",
        color=discord.Color.from_rgb(255, 105, 180)  # Hot pink color
    )

    # Apply the random romantic image to the embed
    embed.set_image(url=selected_image)
    embed.set_footer(text=f"💕 Shipped by {interaction.user.name} 💕", icon_url=interaction.user.display_avatar.url)

    # Respond immediately with the embed
    await interaction.response.send_message(embed=embed)

# Slash Command: Pet
@bot.tree.command(name='pet', description='Show random images of cute pets')
async def pet(interaction: discord.Interaction, pet_type: str = None):
    # List of pet images with variety for each type
    pet_images = {
        'dog': [
            'https://i.imgur.com/CrWcxNh.png',
            'https://i.imgur.com/8Q3ZJfK.png',
            'https://i.imgur.com/4M5N7Pq.png',
            'https://i.imgur.com/9R2T6Sx.png'
        ],
        'cat': [
            'https://i.imgur.com/CrWcxNh.png',
            'https://i.imgur.com/2K8M4Lq.png',
            'https://i.imgur.com/7N9P3Rt.png',
            'https://i.imgur.com/5V6B8Uy.png'
        ],
        'bird': [
            'https://i.imgur.com/CrWcxNh.png',
            'https://i.imgur.com/1A3D5Fg.png',
            'https://i.imgur.com/6H7J9Km.png',
            'https://i.imgur.com/3E4F8Hn.png'
        ],
        'fish': [
            'https://i.imgur.com/CrWcxNh.png',
            'https://i.imgur.com/2B4C6Gj.png',
            'https://i.imgur.com/8K9L2Np.png',
            'https://i.imgur.com/5M7O3Qr.png'
        ],
        'rabbit': [
            'https://i.imgur.com/CrWcxNh.png',
            'https://i.imgur.com/4N6P8Rs.png',
            'https://i.imgur.com/9T2U7Vx.png',
            'https://i.imgur.com/1W3Y5Za.png'
        ],
        'hamster': [
            'https://i.imgur.com/CrWcxNh.png',
            'https://i.imgur.com/6B8D2Fk.png',
            'https://i.imgur.com/3G5H7Jl.png',
            'https://i.imgur.com/7K9M4Nq.png'
        ],
        'turtle': [
            'https://i.imgur.com/CrWcxNh.png',
            'https://i.imgur.com/2P4R6St.png',
            'https://i.imgur.com/8U1V3Wx.png',
            'https://i.imgur.com/5Y7Z9Ac.png'
        ],
        'snake': [
            'https://i.imgur.com/CrWcxNh.png',
            'https://i.imgur.com/4E6G8Hj.png',
            'https://i.imgur.com/9I2K5Lm.png',
            'https://i.imgur.com/1O3Q7Tn.png'
        ],
        'lizard': [
            'https://i.imgur.com/CrWcxNh.png',
            'https://i.imgur.com/6S8U2Va.png',
            'https://i.imgur.com/3D5F7Gb.png',
            'https://i.imgur.com/7H9J4Kc.png'
        ],
        'frog': [
            'https://i.imgur.com/CrWcxNh.png',
            'https://i.imgur.com/2X4Z6Cb.png',
            'https://i.imgur.com/8V1B3Nd.png',
            'https://i.imgur.com/5M7P9Fe.png'
        ]
    }

    # If no pet type specified, randomly choose from all types for variety
    if pet_type is None:
        available_types = list(pet_images.keys())
        selected_pet_type = random.choice(available_types)
        images = pet_images[selected_pet_type]
        pet_display_name = selected_pet_type.title()
    else:
        # Get images for the requested pet type (case insensitive)
        pet_lower = pet_type.lower()
        images = pet_images.get(pet_lower, pet_images['dog'])  # Default to dog if invalid type
        pet_display_name = pet_type.title()

    # Select a random image
    selected_image = random.choice(images)

    # Create embed with the pet image
    embed = discord.Embed(
        title=f"🐾 Cute {pet_display_name}! 🐾",
        description=f"Here's a random image of a {pet_display_name.lower()}!",
        color=discord.Color.blue()
    )
    embed.set_image(url=selected_image)
    embed.set_footer(text=f"Requested by {interaction.user.name}", icon_url=interaction.user.display_avatar.url)

    # Respond with the embed
    await interaction.response.send_message(embed=embed)

# Slash Command: AFK
@bot.tree.command(name='afk', description='Set yourself as AFK with an optional reason')
async def afk(interaction: discord.Interaction, reason: str = "AFK"):
    # Defer the response to prevent timeout errors
    await interaction.response.defer()
    afk_users[interaction.user.id] = reason
    embed = discord.Embed(
        title="AFK Set",
        description=f"You are now AFK: {reason}",
        color=discord.Color.blue()
    )
    await interaction.followup.send(embed=embed)

# Slash Command: Enable Quiet Mode
@bot.tree.command(name='quiet', description='Enable quiet mode (owner only)')
@commands.is_owner()
async def slash_quiet(interaction: discord.Interaction):
    global quiet_mode
    quiet_mode = True
    embed = discord.Embed(
        title="Quiet Mode Enabled",
        description="The bot will not send welcome messages or other automatic messages.",
        color=discord.Color.blue()
    )
    embed.set_footer(text=f"Requested by {interaction.user}")
    await interaction.response.send_message(embed=embed)

# Slash Command: Disable Quiet Mode
@bot.tree.command(name='unquiet', description='Disable quiet mode (owner only)')
@commands.is_owner()
async def slash_unquiet(interaction: discord.Interaction):
    global quiet_mode
    quiet_mode = False
    embed = discord.Embed(
        title="Quiet Mode Disabled",
        description="The bot will resume sending welcome messages and other automatic messages.",
        color=discord.Color.green()
    )
    embed.set_footer(text=f"Requested by {interaction.user}")
    await interaction.response.send_message(embed=embed)

# Slash Command: Set Announcement Channel
@bot.tree.command(name='setannounce', description='Set the announcement channel for bot online messages (manage guild permission required)')
@commands.has_permissions(manage_guild=True)
async def setannounce(interaction: discord.Interaction, channel: discord.TextChannel):
    # Defer the response to prevent timeout errors
    await interaction.response.defer()

    announce_channels[interaction.guild.id] = channel.id
    embed = discord.Embed(
        title="Announcement Channel Set",
        description=f"Online announcements will be sent to {channel.mention}",
        color=discord.Color.green()
    )
    embed.set_footer(text=f"Set by {interaction.user}")
    await interaction.followup.send(embed=embed)

# Slash Command: Invites
@bot.tree.command(name='invites', description='View invite statistics for the server')
async def invites(interaction: discord.Interaction):
    # Defer the response to prevent timeout errors
    await interaction.response.defer()

    guild_id = interaction.guild.id
    if guild_id not in invite_tracking:
        await interaction.followup.send("No invite data available for this server.")
        return

    data = invite_tracking[guild_id]
    invites = data['invites']
    joins = data['joins']

    embed = discord.Embed(
        title="📊 Invite Statistics",
        description=f"Invite tracking for {interaction.guild.name}",
        color=discord.Color.blue()
    )

    # List invites
    invite_list = ""
    for code, info in invites.items():
        inviter_id = info['inviter']
        uses = info['uses']
        inviter = interaction.guild.get_member(inviter_id)
        inviter_name = inviter.mention if inviter else f"User {inviter_id}"
        invite_list += f"**{code}** - {inviter_name} ({uses} uses)\n"

    if invite_list:
        embed.add_field(name="Active Invites", value=invite_list, inline=False)
    else:
        embed.add_field(name="Active Invites", value="None", inline=False)

    # Recent joins
    join_list = ""
    for join in joins[-10:]:  # Show last 10 joins
        user_id = join['user_id']
        inviter_id = join['inviter_id']
        timestamp = join['timestamp']
        user = interaction.guild.get_member(user_id)
        inviter = interaction.guild.get_member(inviter_id)
        user_name = user.mention if user else f"User {user_id}"
        inviter_name = inviter.mention if inviter else f"User {inviter_id}"
        join_list += f"{user_name} invited by {inviter_name} at {timestamp}\n"

    if join_list:
        embed.add_field(name="Recent Joins", value=join_list, inline=False)
    else:
        embed.add_field(name="Recent Joins", value="None", inline=False)

    embed.set_footer(text=f"Requested by {interaction.user}")
    await interaction.followup.send(embed=embed)

# Slash Command: Info (user profile + badges)
@bot.tree.command(name='info', description='Shows profile information and badges for a user')
async def info_slash(interaction: discord.Interaction, member: discord.Member = None):
    # Respond immediately with the embed (no defer needed for this command)
    user = member or interaction.user

    # Basic embed with avatar
    embed = discord.Embed(title=f"Profile: {user}", color=discord.Color.blurple())
    embed.set_thumbnail(url=user.display_avatar.url if getattr(user, 'display_avatar', None) else None)

    # Account creation
    created = getattr(user, 'created_at', None)
    if created:
        embed.add_field(name='Account Created', value=created.strftime('%Y-%m-%d %H:%M:%S UTC'), inline=False)

    # If Member, show roles and activity
    if isinstance(user, discord.Member):
        # Roles (exclude @everyone and None roles)
        roles = [r.mention for r in user.roles if r is not None and r.name != '@everyone']
        embed.add_field(name='Roles', value=', '.join(roles) if roles else 'None', inline=False)

        # Current Activity
        activity = user.activity
        if activity:
            embed.add_field(name='Current Activity', value=str(activity), inline=False)
        else:
            embed.add_field(name='Current Activity', value='None', inline=False)

    # Badges / Public flags
    flags = getattr(user, 'public_flags', None)
    # Mapping for badge names with emojis
    badge_map = {
        'staff': '👨‍💼 Discord Staff',
        'partner': '🤝 Partner',
        'hypesquad': '🏠 HypeSquad',
        'bug_hunter': '🐛 Bug Hunter',
        'bug_hunter_level_2': '🐞 Bug Hunter (Level 2)',
        'hypesquad_bravery': '💪 HypeSquad Bravery',
        'hypesquad_brilliance': '💡 HypeSquad Brilliance',
        'hypesquad_balance': '⚖️ HypeSquad Balance',
        'early_supporter': '👑 Early Supporter',
        'team_user': '👥 Team User',
        'verified_bot_developer': '🤖 Verified Bot Developer',
        'system': '⚙️ System',
        'premium_subscriber': '✨ Nitro Subscriber',
        'active_developer': '💻 Active Developer',
        'discord_employee': '🏢 Discord Employee',
        'certified_moderator': '🛡️ Certified Moderator'
    }

    badges = []
    if flags is not None:
        # Inspect all attributes on the flags object and collect truthy boolean flags
        for attr in dir(flags):
            if attr.startswith('_'):
                continue
            try:
                val = getattr(flags, attr)
            except Exception:
                continue
            if isinstance(val, bool) and val:
                label = badge_map.get(attr, attr.replace('_', ' ').title())
                badges.append(label)
        # Deduplicate while preserving order
        seen = set()
        unique_badges = []
        for b in badges:
            if b not in seen:
                unique_badges.append(b)
                seen.add(b)
        badges = unique_badges

    embed.add_field(name='Badges', value=', '.join(badges) if badges else 'None', inline=False)

    # If Member, add joined date at the bottom
    if isinstance(user, discord.Member):
        joined = getattr(user, 'joined_at', None)
        if joined:
            embed.add_field(name='Joined Server', value=joined.strftime('%Y-%m-%d %H:%M'), inline=False)

    # ID and mention
    embed.set_footer(text=f'User ID: {user.id} | Requested by {interaction.user}', icon_url=interaction.user.display_avatar.url)
    await interaction.response.send_message(embed=embed)

# Slash Command: Notify Tester
@bot.tree.command(name="notify_tester", description="DMs a user about their app test request.")
@commands.has_permissions(administrator=True)
async def notify_tester(interaction: discord.Interaction, user_id: str):
    """
    Sends a custom DM notification to a specific user ID.

    :param interaction: The interaction object from Discord.
    :param user_id: The ID of the user to receive the notification.
    """
    await interaction.response.defer(ephemeral=True)  # Acknowledge the command quickly

    try:
        # 1. Convert the input string ID to an integer
        target_id = int(user_id)

        # 2. Fetch the User object from Discord (this is necessary to send a DM)
        # Using bot.fetch_user is reliable for any user ID, even if they aren't in the same guild.
        target_user = await bot.fetch_user(target_id)

        if not target_user:
            await interaction.followup.send(f"Error: Could not find user with ID `{user_id}`.", ephemeral=True)
            return

        # 3. Create the notification embed with buttons
        embed = discord.Embed(
            title="🧪 App Test Program Invitation",
            description=(
                f"Hello {target_user.name}!\n\n"
                "We have received and are processing your application request for the **App Test Program**!\n"
                "Our developers will be in touch shortly with an invitation to the private testing server or additional instructions.\n\n"
                "Please let us know if you're interested in participating:"
            ),
            color=discord.Color.blue()
        )
        embed.set_footer(text="Click Accept or Decline to respond")

        # Create the view with buttons
        view = TesterResponseView(target_id)

        # 4. Send the Direct Message with embed and buttons
        await target_user.send(embed=embed, view=view)

        # 5. Send confirmation back to the command user (the developer/admin)
        await interaction.followup.send(
            f"✅ Success! Sent test notification DM to user: **{target_user.name}** (`{target_id}`).",
            ephemeral=True
        )

    except ValueError:
        # Handles cases where the user input is not a valid number
        await interaction.followup.send("Error: The user ID must be a valid number.", ephemeral=True)

    except discord.Forbidden:
        # Handles cases where the bot cannot DM the user (e.g., user blocked bot)
        await interaction.followup.send(
            f"Error: I could not send a DM to that user (ID: `{user_id}`). They may have DMs disabled for server members.",
            ephemeral=True
        )

    except Exception as e:
        # General error handling
        logging.error(f"An unexpected error occurred: {e}")
        await interaction.followup.send(f"An unexpected error occurred while sending the DM. Check the bot's logs. Error: `{e}`", ephemeral=True)

# Command: Ping (prefix version)
@bot.command(name='ping', help='Shows bot latency')
async def ping_prefix(ctx):
    latency = round(bot.latency * 1000)
    await ctx.send(f'Pong! Latency: {latency}ms')

# Command: Generate Image (prefix version)
@bot.command(name='image', help='Generate an image using AI (free): j!image <prompt> [aspect_ratio]')
async def image_prefix(ctx, prompt: str, aspect_ratio: str = "1:1"):
    # Validate prompt length
    if len(prompt) > 1000:
        embed = discord.Embed(
            title="Invalid Prompt",
            description="Prompt must be 1000 characters or less.",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
        return

    # Validate aspect_ratio
    valid_ratios = ["1:1", "16:9", "9:16", "21:9", "9:21"]
    if aspect_ratio not in valid_ratios:
        embed = discord.Embed(
            title="Invalid Aspect Ratio",
            description=f"Valid aspect ratios: {', '.join(valid_ratios)}",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
        return

    try:
        # Map aspect ratios to Pollinations.ai dimensions
        aspect_map = {
            "1:1": (1024, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "21:9": (1024, 438),
            "9:21": (438, 1024)
        }
        width, height = aspect_map.get(aspect_ratio, (1024, 1024))

        # Use Pollinations.ai (free, no API key needed)
        image_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?width={width}&height={height}&model=flux"

        # Download the image
        response = requests.get(image_url, timeout=30)

        if response.status_code == 200:
            # The response content is the image binary
            image_data = response.content

            # Create a file-like object for Discord
            from io import BytesIO
            image_file = discord.File(BytesIO(image_data), filename="generated_image.jpeg")

            # Create embed with generated image
            embed = discord.Embed(
                title="AI Image Generation (Free)",
                description=f"**Prompt:** {prompt}",
                color=discord.Color.blue()
            )
            embed.add_field(name="Aspect Ratio", value=aspect_ratio, inline=True)
            embed.add_field(name="Model", value="Flux (via Pollinations.ai)", inline=True)
            embed.set_image(url="attachment://generated_image.jpeg")
            embed.set_footer(text=f"Generated by {ctx.author}")

            await ctx.send(embed=embed, file=image_file)
        else:
            embed = discord.Embed(
                title="Image Generation Error",
                description=f"Failed to generate image: HTTP {response.status_code}",
                color=discord.Color.red()
            )
            await ctx.send(embed=embed)

    except requests.exceptions.Timeout:
        embed = discord.Embed(
            title="Timeout Error",
            description="Image generation took too long. Please try again.",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
    except requests.exceptions.RequestException as e:
        embed = discord.Embed(
            title="Network Error",
            description=f"Failed to connect to Pollinations.ai: {str(e)}",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
    except Exception as e:
        embed = discord.Embed(
            title="Unexpected Error",
            description=f"An unexpected error occurred: {str(e)}",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)

# Command: Clear messages
@bot.command(name='clear', help='Clears specified number of messages')
@commands.has_permissions(manage_messages=True)
async def clear(ctx, amount: int):
    await ctx.channel.purge(limit=amount + 1)
    await ctx.send(f'{amount} messages have been cleared!', delete_after=5)

# Command: Server info
@bot.command(name='serverinfo', help='Displays information about the server')
async def server_info(ctx):
    guild = ctx.guild
    embed = discord.Embed(title=f'{guild.name} Server Information', color=discord.Color.blue())
    embed.add_field(name='Server Owner', value=guild.owner.mention, inline=False)
    embed.add_field(name='Member Count', value=guild.member_count, inline=True)
    embed.add_field(name='Channel Count', value=len(guild.channels), inline=True)
    embed.add_field(name='Server Created', value=guild.created_at.strftime('%B %d, %Y'), inline=False)
    embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
    await ctx.send(embed=embed)

# Command: Set status (owner only)
@bot.command(name='setstatus', help='Owner-only: set bot presence. Usage: j!setstatus <type> [text]')
@owner_only()
async def setstatus(ctx, type: str, *, text: str = None):
    """
    type: playing | watching | listening | streaming | clear | online | idle | dnd | invisible
    text: For playing/watching/listening: the activity text.
          For streaming: use `Name|URL` (pipe-separated) so a url is provided for Streaming activity.
    """
    type = type.lower()

    # Status-only changes
    status_map = {
        'online': discord.Status.online,
        'idle': discord.Status.idle,
        'dnd': discord.Status.dnd,
        'invisible': discord.Status.invisible,
    }
    if type in status_map:
        await bot.change_presence(status=status_map[type])
        await ctx.send(f'Status set to {type}.')
        return

    # Clear activity
    if type == 'clear':
        await bot.change_presence(activity=None)
        await ctx.send('Cleared activity.')
        return

    # Activity changes
    activity = None
    if type == 'playing':
        activity = discord.Game(text or "")
    elif type == 'watching':
        activity = discord.Activity(type=discord.ActivityType.watching, name=text or "")
    elif type == 'listening':
        activity = discord.Activity(type=discord.ActivityType.listening, name=text or "")
    elif type == 'streaming':
        # Expect text in form "Name|URL"
        if not text or '|' not in text:
            await ctx.send('For streaming, use: j!setstatus streaming Name|URL')
            return
        name, url = [s.strip() for s in text.split('|', 1)]
        activity = discord.Streaming(name=name or "", url=url)
    else:
        await ctx.send('Invalid type. Use playing|watching|listening|streaming|clear|online|idle|dnd|invisible')
        return

    try:
        await bot.change_presence(activity=activity)
        await ctx.send(f'Set activity to {type} {f"\"{text}\"" if text else ""}')
    except Exception as e:
        await ctx.send(f'Failed to set activity: {e}')

# Command: List all commands
@bot.command(name='commands', help='Shows all available commands')
async def commands_list(ctx):
    # Create the select menu for command categories
    class CommandSelect(discord.ui.Select):
        def __init__(self):
            options = [
                discord.SelectOption(label="General Commands", description="Fun and utility commands", emoji="📋", value="general"),
                discord.SelectOption(label="Moderation Commands", description="Server management commands", emoji="🛡️", value="moderation"),
                discord.SelectOption(label="Owner Commands", description="Bot owner only commands", emoji="👑", value="owner"),
                discord.SelectOption(label="All Commands", description="View all commands at once", emoji="📚", value="all")
            ]
            super().__init__(placeholder="Choose a command category...", min_values=1, max_values=1, options=options)

        async def callback(self, interaction: discord.Interaction):
            # Check if the user who clicked is the same as the command user
            if interaction.user != ctx.author:
                await interaction.response.send_message("Only the person who ran the command can use this menu!", ephemeral=True)
                return

            selected_value = self.values[0]

            embed = discord.Embed(
                title="🤖 Joshua's Bot Commands",
                color=discord.Color.blue()
            )
            embed.set_thumbnail(url="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExb2p6cXNicnEzc3E4eHE1MDM1MXp3Z3lvOG82eW9mbHNqdnkxcnFzMiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WiIuC6fAOoXD2/giphy.gif")

            if selected_value == "general":
                general_commands = ""
                general_commands += "📋 `j!commands` - Shows this list of commands\n"
                general_commands += "🏓 `j!ping` or `/ping` - Shows bot latency\n"
                general_commands += "ℹ️ `j!serverinfo` - Shows server information\n"
                general_commands += "🗣️ `j!talk <message>` - Makes the bot say your message\n"
                general_commands += "🎙️ `/talk <message>` - Repeats whatever message you send\n"
                general_commands += "🤖 `/ai <prompt> [image_url]` - Chat with AI (supports text and optional image)\n"
                general_commands += "🎨 `/image <prompt>` - Generate an image using AI\n"
                general_commands += "💕 `/ship <user1> <user2>` - Ship two users and rate their love compatibility\n"
                general_commands += "🐾 `/pet <pet_type>` - Show random images of cute pets\n"
                general_commands += "😴 `/afk [reason]` - Set yourself as AFK\n"
                general_commands += "👤 `/info [user]` - Shows profile information and badges\n"
                general_commands += "👤 `j!info [user]` - Shows profile information and badges (prefix version)\n"
                general_commands += "🎮 `j!game` or `/game` - Start a word-based minesweeper game\n"
                general_commands += "🔍 `j!reveal <row> <col>` or `/reveal <row> <col>` - Reveal a cell in the minesweeper game\n"
                general_commands += "🕵️ `j!hide [choice]` or `/hide [choice]` - Play hide-and-seek with P Diddy\n"
                embed.add_field(name="📋 General Commands", value=general_commands, inline=False)

            elif selected_value == "moderation":
                mod_commands = ""
                mod_commands += "👢 `j!kick @user [reason]` - Kicks a user from the server\n"
                mod_commands += "🔨 `j!ban @user [reason]` - Bans a user from the server\n"
                mod_commands += "🔓 `j!unban Username#1234` - Unbans a user\n"
                mod_commands += "⏰ `j!timeout @user <minutes> [reason]` - Timeout (mute) a user\n"
                mod_commands += "🔄 `j!untimeout @user [reason]` - Removes timeout from a user\n"
                mod_commands += "🧹 `j!clear <number>` - Clears specified number of messages\n"
                embed.add_field(name="🛡️ Moderation Commands", value=mod_commands, inline=False)

            elif selected_value == "owner":
                owner_commands = ""
                owner_commands += "⚙️ `j!setstatus <type> [text]` - Set bot presence (owner only)\n"
                owner_commands += "⏹️ `j!close` - Shutdown the bot (bot owner only)\n"
                owner_commands += "🤫 `/quiet` - Enable quiet mode (owner only)\n"
                owner_commands += "🔊 `/unquiet` - Disable quiet mode (owner only)\n"
                owner_commands += "📢 `/setannounce <channel>` - Set announcement channel for bot online messages\n"
                embed.add_field(name="👑 Owner Commands", value=owner_commands, inline=False)

            elif selected_value == "all":
                # General Commands
                general_commands = ""
                general_commands += "📋 `j!commands` - Shows this list of commands\n"
                general_commands += "🏓 `j!ping` or `/ping` - Shows bot latency\n"
                general_commands += "ℹ️ `j!serverinfo` - Shows server information\n"
                general_commands += "🗣️ `j!talk <message>` - Makes the bot say your message\n"
                general_commands += "🎙️ `/talk <message>` - Repeats whatever message you send\n"
                general_commands += "🤖 `/ai <prompt> [image_url]` - Chat with AI (supports text and optional image)\n"
                general_commands += "🎨 `/image <prompt>` - Generate an image using AI\n"
                general_commands += "💕 `/ship <user1> <user2>` - Ship two users and rate their love compatibility\n"
                general_commands += "🐾 `/pet <pet_type>` - Show random images of cute pets\n"
                general_commands += "😴 `/afk [reason]` - Set yourself as AFK\n"
                general_commands += "👤 `/info [user]` - Shows profile information and badges\n"
                general_commands += "👤 `j!info [user]` - Shows profile information and badges (prefix version)\n"
                general_commands += "🎮 `j!game` or `/game` - Start a word-based minesweeper game\n"
                general_commands += "🔍 `j!reveal <row> <col>` or `/reveal <row> <col>` - Reveal a cell in the minesweeper game\n"
                general_commands += "🕵️ `j!hide [choice]` or `/hide [choice]` - Play hide-and-seek with P Diddy\n"
                embed.add_field(name="📋 General Commands", value=general_commands, inline=False)

                # Moderation Commands
                mod_commands = ""
                mod_commands += "👢 `j!kick @user [reason]` - Kicks a user from the server\n"
                mod_commands += "🔨 `j!ban @user [reason]` - Bans a user from the server\n"
                mod_commands += "🔓 `j!unban Username#1234` - Unbans a user\n"
                mod_commands += "⏰ `j!timeout @user <minutes> [reason]` - Timeout (mute) a user\n"
                mod_commands += "🔄 `j!untimeout @user [reason]` - Removes timeout from a user\n"
                mod_commands += "🧹 `j!clear <number>` - Clears specified number of messages\n"
                embed.add_field(name="🛡️ Moderation Commands", value=mod_commands, inline=False)

                # Owner Commands
                owner_commands = ""
                owner_commands += "⚙️ `j!setstatus <type> [text]` - Set bot presence (owner only)\n"
                owner_commands += "⏹️ `j!close` - Shutdown the bot (bot owner only)\n"
                owner_commands += "🤫 `/quiet` - Enable quiet mode (owner only)\n"
                owner_commands += "🔊 `/unquiet` - Disable quiet mode (owner only)\n"
                owner_commands += "📢 `/setannounce <channel>` - Set announcement channel for bot online messages\n"
                embed.add_field(name="👑 Owner Commands", value=owner_commands, inline=False)

            # Add usage notes
            notes = (
                "```\n"
                "[] = Optional parameter\n"
                "<> = Required parameter\n"
                "@user = Mention a user\n"
                "All moderation commands require appropriate permissions```"
            )
            embed.add_field(name="📝 Usage Notes", value=notes, inline=False)

            # Set footer with timestamp
            embed.set_footer(text=f"Requested by {ctx.author}", icon_url=ctx.author.display_avatar.url)
            embed.timestamp = ctx.message.created_at

            await interaction.response.edit_message(embed=embed, view=self.view)

    class CommandView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=300)  # 5 minutes timeout
            self.add_item(CommandSelect())

    # Send only the interactive dropdown view
    view = CommandView()
    await ctx.send(view=view)

# Command: Kick member
@bot.command(name='kick', help='Kicks a member from the server')
@commands.has_permissions(kick_members=True)
async def kick(ctx, member: discord.Member, *, reason=None):
    try:
        await member.kick(reason=reason)
        embed = discord.Embed(
            title="👢 Member Kicked",
            description=f"{member.mention} has been kicked from the server.",
            color=discord.Color.red()
        )
        embed.add_field(name="Reason", value=reason or "No reason provided")
        embed.set_footer(text=f"Kicked by {ctx.author}")
        await ctx.send(embed=embed)
    except discord.Forbidden:
        await ctx.send("I don't have permission to kick members!")

# Command: Ban member
@bot.command(name='ban', help='Bans a member from the server')
@commands.has_permissions(ban_members=True)
async def ban(ctx, member: discord.Member, *, reason=None):
    try:
        await member.ban(reason=reason)
        embed = discord.Embed(
            title="🔨 Member Banned",
            description=f"{member.mention} has been banned from the server.",
            color=discord.Color.dark_red()
        )
        embed.add_field(name="Reason", value=reason or "No reason provided")
        embed.set_footer(text=f"Banned by {ctx.author}")
        await ctx.send(embed=embed)
    except discord.Forbidden:
        await ctx.send("I don't have permission to ban members!")

# Command: Timeout (mute) member
@bot.command(name='timeout', help='Timeout (mute) a member for a specified duration (in minutes)')
@commands.has_permissions(moderate_members=True)
async def timeout(ctx, member: discord.Member, duration: int, *, reason=None):
    try:
        # Convert duration from minutes to seconds and create timedelta
        duration_seconds = duration * 60
        await member.timeout_for(datetime.timedelta(seconds=duration_seconds), reason=reason)
        embed = discord.Embed(
            title="⏰ Member Timed Out",
            description=f"{member.mention} has been timed out for {duration} minutes.",
            color=discord.Color.orange()
        )
        embed.add_field(name="Reason", value=reason or "No reason provided")
        embed.set_footer(text=f"Timed out by {ctx.author}")
        await ctx.send(embed=embed)
    except discord.Forbidden:
        await ctx.send("I don't have permission to timeout members!")

# Command: Remove timeout (unmute) member
@bot.command(name='untimeout', help='Remove timeout from a member')
@commands.has_permissions(moderate_members=True)
async def untimeout(ctx, member: discord.Member, *, reason=None):
    try:
        await member.remove_timeout(reason=reason)
        embed = discord.Embed(
            title="🔄 Timeout Removed",
            description=f"Timeout has been removed from {member.mention}.",
            color=discord.Color.green()
        )
        embed.add_field(name="Reason", value=reason or "No reason provided")
        embed.set_footer(text=f"Timeout removed by {ctx.author}")
        await ctx.send(embed=embed)
    except discord.Forbidden:
        await ctx.send("I don't have permission to remove timeouts!")

# Command: Unban member
@bot.command(name='unban', help='Unbans a member from the server')
@commands.has_permissions(ban_members=True)
async def unban(ctx, *, member):
    try:
        # Find the banned user
        banned_users = [entry async for entry in ctx.guild.bans()]
        member_name, member_discriminator = member.split('#')

        for ban_entry in banned_users:
            user = ban_entry.user
            if (user.name, user.discriminator) == (member_name, member_discriminator):
                await ctx.guild.unban(user)
                embed = discord.Embed(
                    title="🔓 Member Unbanned",
                    description=f"{user.mention} has been unbanned from the server.",
                    color=discord.Color.green()
                )
                embed.set_footer(text=f"Unbanned by {ctx.author}")
                await ctx.send(embed=embed)
                return
        await ctx.send(f"Could not find banned user {member}")
    except discord.Forbidden:
        await ctx.send("I don't have permission to unban members!")

# Command: Game (Word Minesweeper)
@bot.command(name='game', help='Play a word-based minesweeper game')
async def game(ctx):
    # Check if user already has a game
    user_id = ctx.author.id
    if user_id in game_state:
        await ctx.send("You already have a game in progress! Finish it first.")
        return

    # Generate a 5x5 grid with words
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange", "peach", "pear", "quince", "raspberry", "strawberry", "tangerine", "watermelon"]
    grid = [[random.choice(words) for _ in range(5)] for _ in range(5)]

    # Place 5 bombs randomly
    bombs = set()
    while len(bombs) < 5:
        x, y = random.randint(0, 4), random.randint(0, 4)
        bombs.add((x, y))

    # Store game state
    game_state[user_id] = {
        'grid': grid,
        'bombs': bombs,
        'revealed': set(),
        'started': True
    }

    # Display initial grid
    display_grid = ""
    for i in range(5):
        for j in range(5):
            display_grid += "⬜"
        display_grid += "\n"

    embed = discord.Embed(
        title="Word Minesweeper",
        description="Find the 5 hidden bombs! Click on a word to reveal it.\n\n" + display_grid,
        color=discord.Color.blue()
    )
    embed.set_footer(text="Use j!reveal <row> <col> to reveal a cell (1-5)")
    await ctx.send(embed=embed)

# Command: Reveal cell
@bot.command(name='reveal', help='Reveal a cell in the game: j!reveal <row> <col>')
async def reveal(ctx, row: int, col: int):
    user_id = ctx.author.id
    if user_id not in game_state:
        await ctx.send("You don't have a game in progress! Start one with j!game")
        return

    game = game_state[user_id]
    row -= 1  # 0-indexed
    col -= 1

    if not (0 <= row < 5 and 0 <= col < 5):
        await ctx.send("Invalid coordinates! Use 1-5 for row and column.")
        return

    if (row, col) in game['revealed']:
        await ctx.send("That cell is already revealed!")
        return

    game['revealed'].add((row, col))

    if (row, col) in game['bombs']:
        # Game over - bomb hit
        del game_state[user_id]
        embed = discord.Embed(
            title="💥 Game Over!",
            description="Ooof! You lost, git gud",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
        return

    # Check win condition
    total_cells = 25
    revealed_cells = len(game['revealed'])
    if revealed_cells == total_cells - 5:  # 25 - 5 bombs
        del game_state[user_id]
        embed = discord.Embed(
            title="🎉 You Win!",
            description="Congratulations! You found all the safe words!",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)
        return

    # Display updated grid
    display_grid = ""
    for i in range(5):
        for j in range(5):
            if (i, j) in game['revealed']:
                display_grid += f"{game['grid'][i][j]} "
            else:
                display_grid += "⬜"
        display_grid += "\n"

    embed = discord.Embed(
        title="Word Minesweeper",
        description=f"Revealed {revealed_cells} cells. {5 - len(game['bombs'] & game['revealed'])} bombs remaining.\n\n" + display_grid,
        color=discord.Color.blue()
    )
    embed.set_footer(text="Use j!reveal <row> <col> to reveal a cell (1-5)")
    await ctx.send(embed=embed)

# Command: Hide (prefix version)
@bot.command(name='hide', help='Play hide-and-seek with P Diddy: j!hide [choice]')
async def hide_prefix(ctx, choice: int = None):
    user_id = ctx.author.id

    if choice is None:
        # Display the status and available spots
        status = get_status_message(user_id)
        embed = discord.Embed(
            title="P Diddy is counting!",
            description=status,
            color=discord.Color.blue()
        )
        await ctx.send(embed=embed)
    else:
        # Run the game logic
        message, _ = run_grand_escape_round(user_id, choice)
        await ctx.send(message)

# Error handling
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.errors.CheckFailure):
        await ctx.send('You do not have the required permissions to use this command.')
    elif isinstance(error, commands.errors.MissingRequiredArgument):
        await ctx.send('Please provide all required arguments.')
    else:
        await ctx.send(f'An error occurred: {str(error)}')

if __name__ == "__main__":
    # Start the Discord bot
    bot.run(DISCORD_BOT_TOKEN)
