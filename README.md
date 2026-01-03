from keep_alive import keep_alive # NEW
import discord
from discord import app_commands
from discord.ext import commands
from discord.ui import Button, View
import asyncio
import random
from openai import OpenAI
from collections import defaultdict
import os
import datetime
import requests
import json
import time
from dotenv import load_dotenv
from cooldowns import check_cooldowns
from hide_game import run_grand_escape_round, get_status_message
import io
from dateutil import parser
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import socket
import google.generativeai as genai
import aiohttp

# Load environment variables
load_dotenv()

# Initialize YouTube client
try:
    youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
except Exception:
    youtube = None

# Initialize Cerebras client (OpenAI-compatible)
ai_client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.getenv('CEREBRAS_API_KEY')
)

# Initialize Gemini AI for image recognition
gemini_api_key = os.getenv('GEMINI_API_KEY')
if gemini_api_key:
    genai.configure(api_key=gemini_api_key.strip().strip('"').strip("'"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

async def get_image_bytes(url):
    """Downloads the image from Discord's servers"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()
            return None

# Custom status function
async def set_custom_status(status_text: str, emoji: str = None):
    """Sets a true Custom Status (no 'Playing' prefix)."""
    activity = discord.CustomActivity(name=status_text, emoji=emoji)
    await bot.change_presence(activity=activity)
    return f"Custom status set to: {status_text}"

# Tool definition for AI
tools = [
    {
        "type": "function",
        "function": {
            "name": "set_custom_status",
            "description": "Sets a custom text status for the bot. Use this when the user asks you to change your status text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status_text": {
                        "type": "string", 
                        "description": "The text the status should show (e.g., 'Feeling sleepy' or 'Vibing')"
                    },
                    "emoji": {
                        "type": "string", 
                        "description": "A single emoji character to display with the status (optional)"
                    }
                },
                "required": ["status_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_interface",
            "description": "Creates an interactive button interface. Use this when the user asks for buttons, menus, or interactive elements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string", 
                        "description": "The message to display with the interface"
                    },
                    "button_label": {
                        "type": "string", 
                        "description": "The text on the button"
                    },
                    "button_emoji": {
                        "type": "string", 
                        "description": "Emoji for the button (optional)"
                    }
                },
                "required": ["message", "button_label"]
            }
        }
    }
]

class MyInterface(View):
    def __init__(self, label="Click Me!", emoji="🚀"):
        super().__init__()
        self.label = label
        self.emoji = emoji
        
    @discord.ui.button(label="Click Me!", style=discord.ButtonStyle.primary, emoji="🚀")
    async def button_callback(self, interaction: discord.Interaction, button: discord.Button):
        await interaction.response.send_message("Interface Updated!", ephemeral=True)

async def create_interface(message: str, button_label: str, button_emoji: str = "🚀"):
    """Creates an interactive button interface."""
    view = MyInterface(button_label, button_emoji)
    return message, view

# Memory storage: Stores last 5 messages per user to keep context
# Format: {user_id: [messages]}
user_memory = defaultdict(list)

# AI mention cooldown storage
ai_cooldowns = {}

# Global AI response cooldown (5 seconds)
global_ai_cooldown = 0

# Configuration
BOT_NAME = "Synapse"
OWNER_ID = 1231525871257649213  # Your specific ID
COOLDOWN_TIME = 10
ANNOUNCE_FILE = 'announce_channels.json'

# User cooldowns for bot mentions - increased cooldown time
user_cooldowns = {}
user_last_message = {}  # Track last message content to prevent duplicates
processing_users = set()  # Track users currently being processed to prevent spam
user_locks = {}  # Dictionary to store a unique lock for every user
processing_lock = asyncio.Lock()  # Global lock to prevent multiple overlapping responses

# Bot busy state
is_busy = False

# Banned users from using the bot
banned_users = {1219625604740026378}

# Track if banned users have been notified
notified_banned_users = set()

class MyBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.invites = True
        intents.presences = True
        intents.guilds = True
        intents.messages = True
        super().__init__(command_prefix="j!", intents=intents)
        # Dictionary to store the last deleted message per channel
        # Format: {channel_id: message_data_object}
        self.deleted_messages = {}

    async def setup_hook(self):
        # This syncs your slash commands with Discord
        await self.tree.sync()
        print(f"Synced slash commands for {self.user}")

bot = MyBot()

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

# Welcome channel storage (in-memory)
welcome_channels = {}  # guild_id -> channel_id

# Welcome message cooldown per guild (to prevent spam)
welcome_cooldowns = {}  # guild_id -> last_welcome_time

# New Year message tracking
new_year_sent = False  # Has the message been sent?

# Global check for banned users on slash commands
@bot.tree.interaction_check
async def interaction_check(interaction: discord.Interaction) -> bool:
    return interaction.user.id not in banned_users

# Global check for banned users on prefix commands
@bot.check
async def global_ban_check(ctx):
    return ctx.author.id not in banned_users

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
    # Only log connection info when ready. Disabled automatic 'hue' message.
    print(f'{bot.user} has connected to Discord!')
    print(f'Bot is in {len(bot.guilds)} guilds')
    # Fetch invites for tracking
    for guild in bot.guilds:
        try:
            invites = await guild.invites()
            invite_tracking[guild.id] = {'invites': {inv.code: {'inviter': inv.inviter.id if inv.inviter else None, 'uses': inv.uses} for inv in invites}, 'joins': []}
        except Exception as e:
            print(f"Failed to fetch invites for {guild.name}: {e}")

    # Load announce channels from file
    try:
        with open(ANNOUNCE_FILE, 'r') as f:
            announce_channels.update(json.load(f))
    except FileNotFoundError:
        pass
    # Ensure the bot shows as online with a helpful activity
    try:
        await bot.change_presence(status=discord.Status.online, activity=discord.Activity(type=discord.ActivityType.watching, name="/help", state="Ping me to start a conversation!"))
    except Exception:
        # If for some reason presence can't be changed right away, ignore and continue
        pass

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
                    embed.set_footer(text=f"Connected at {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    await channel.send(embed=embed)
                except Exception as e:
                    print(f"Failed to send online announcement in {guild.name}: {e}")

    # Send New Year message immediately if date matches
    # bot.loop.create_task(send_new_year_now())

    # Start New Year check task
    # bot.loop.create_task(check_new_year())

# Event: Message handling
@bot.event
async def on_message(message):
    global global_ai_cooldown
    # 1. Ignore if the message is from the bot itself
    if message.author == bot.user:
        return

    # Check if message starts with command prefix
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    # 2. Check if the bot was pinged/mentioned
    if bot.user.mentioned_in(message):
        user_id = message.author.id

        # 3. Error Handling: Check if we are already talking to this user
        if user_id in processing_users:
            return # Simply ignore the second ping

        try:
            # Add user to the "lock" set
            processing_users.add(user_id)

            async with message.channel.typing():
                # --- YOUR AI LOGIC GOES HERE ---
                # Clean the message (remove mentions in guilds, keep full text in DMs)
                if message.guild:
                    clean_text = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()
                else:
                    clean_text = message.content.strip()

                # Check for image attachments
                image_data = None
                attachment = None
                if message.attachments:
                    attachment = message.attachments[0]
                    if any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'webp']):
                        image_data = await get_image_bytes(attachment.url)

                if not clean_text and not image_data:
                    await message.reply("Hey! You pinged me! How can I help? 😊")
                    # Update global cooldown after response
                    global_ai_cooldown = time.time()
                    return

                # Memory clear command
                if clean_text.lower() == "forget me":
                    user_memory[user_id] = []
                    await message.reply("Memory cleared! What's on your mind now?")
                    return

                # Try AI response
                owner_status = "You are talking to your creator." if user_id == OWNER_ID else "The user is not your creator."

                # Handle image + text or just image
                if image_data and gemini_model:
                    try:
                        prompt = clean_text if clean_text else "Describe this image in detail."
                        contents = [
                            prompt,
                            {"mime_type": attachment.content_type, "data": image_data}
                        ]
                        response = await asyncio.wait_for(
                            asyncio.to_thread(gemini_model.generate_content, contents),
                            timeout=30.0
                        )
                        ai_reply = response.text
                    except Exception as e:
                        print(f"Gemini Error: {e}")
                        ai_reply = "I can see you sent an image, but I'm having trouble analyzing it right now! 📸😅"
                else:
                    # Text-only response using Cerebras
                    full_prompt = f"""You are Synapse, a friendly and casual AI assistant created by your owner with Discord ID {OWNER_ID}.

IMPORTANT: The user you are talking to right now has Discord ID {user_id}. {owner_status}

If this user's ID matches {OWNER_ID}, they are your creator and owner - be extra friendly and casual with them.
Personality: Be casual, friendly, and conversational! Use emojis naturally. Talk like you're chatting with a friend.

User: {clean_text}"""

                    # Build messages with memory for better context
                    if user_id not in user_memory:
                        user_memory[user_id] = []

                    messages = [{"role": msg["role"], "content": msg["content"]} for msg in user_memory[user_id]] + [
                        {"role": "system", "content": f"You are Synapse, a friendly AI assistant. {owner_status}"},
                        {"role": "user", "content": clean_text}
                    ]

                    response = await make_ai_request(messages=messages, max_tokens=200, temperature=0.7)

                    ai_reply = response.choices[0].message.content

                    # Store conversation in memory
                    user_memory[user_id].append({"role": "user", "content": clean_text, "timestamp": datetime.datetime.now(datetime.timezone.utc)})
                    user_memory[user_id].append({"role": "assistant", "content": ai_reply, "timestamp": datetime.datetime.now(datetime.timezone.utc)})
                    # Keep only last 20 messages (10 exchanges)
                    if len(user_memory[user_id]) > 20:
                        user_memory[user_id] = user_memory[user_id][-20:]

                await message.reply(ai_reply)
                # Update global cooldown after successful response
                global_ai_cooldown = time.time()
                # -------------------------------

        except Exception as e:
            print(f"AI Error: {e}")
            fallback_responses = [
                "Hey! Having some tech issues right now, but I'm still here! 😅",
                "Oops! My brain is taking a coffee break! ☕",
                "Sorry dude, connection hiccups right now!"
            ]
            await message.reply(random.choice(fallback_responses))
            return
        finally:
            # 4. Always remove the user from the set, even if the AI fails
            processing_users.remove(user_id)

    # Required to allow other @bot.commands to work
    # await bot.process_commands(message)  # Moved to conditional above

# Event: Message deletion witness
@bot.event
async def on_message_delete(message):
    if message.author.bot:
        return

    # Store message details in our bot's memory
    bot.deleted_messages[message.channel.id] = {
        "content": message.content or "*(No text content - possibly an image or embed)*",
        "author": message.author,
        "time": message.created_at,
        "channel": message.channel.name
    }

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
                        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
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

    # Create a mock message for cooldown check
    class MockMessage:
        def __init__(self, author, channel):
            self.author = author
            self.channel = channel

    # Get the specific welcome channel
    welcome_channel_id = welcome_channels.get(guild_id)
    if welcome_channel_id is None:
        return  # No welcome channel set, skip welcome message

    channel = member.guild.get_channel(welcome_channel_id)

    if channel is None:
        return  # Channel not found, skip welcome message

    mock_message = MockMessage(member, channel)

    # Check cooldowns before sending welcome message
    if check_cooldowns(mock_message):
        return  # Rate limited, skip welcome message

    if channel is not None:
        try:
            await channel.send(f'Welcome {member.mention} to the server!')
        except discord.HTTPException as e:
            if e.status == 429:
                # Rate limited, skip
                pass
            else:
                raise



# Slash Command: Ping
@bot.tree.command(name='ping', description='Responds with the bot\'s latency')
async def ping(interaction: discord.Interaction):
    # Defer the response to prevent timeout errors
    await interaction.response.defer()
    latency = round(bot.latency * 1000)
    await interaction.followup.send(f'Pong! Latency: {latency}ms')







# Slash Command: Recover Deleted Message
@bot.tree.command(name="recover", description="Recover the last deleted message in this channel")
async def recover(interaction: discord.Interaction):
    # Check permissions: owner or admins/mods
    owner_id = os.getenv('OWNER_ID')
    if not (interaction.user.guild_permissions.administrator or (owner_id and interaction.user.id == int(owner_id))):
        await interaction.response.send_message("You need administrator permissions or be the bot owner to use this command.", ephemeral=True)
        return

    channel_id = interaction.channel_id
    data = bot.deleted_messages.get(channel_id)

    if not data:
        await interaction.response.send_message("I haven't seen any messages deleted here since I woke up!", ephemeral=True)
        return

    # Create a nice look for the recovered info
    embed = discord.Embed(
        title="🗑️ Recovered Message",
        description=data["content"],
        color=discord.Color.blue(),
        timestamp=data["time"]
    )
    embed.set_author(name=f"{data['author']} ({data['author'].id})", icon_url=data['author'].display_avatar.url)
    embed.set_footer(text="Sent at")

    await interaction.response.send_message(embed=embed)

# Slash Command: Talk
@bot.tree.command(name='talk', description='Text-to-speech using ElevenLabs')
async def talk(interaction: discord.Interaction, message: str):
    await interaction.response.defer()
    
    try:
        url = "https://api.elevenlabs.io/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": "sk_c2c8bdbf7e18561f4fd83e4d16b3bf3ab13a621c7e9b2ed3"
        }
        data = {
            "text": message,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    audio_file = discord.File(io.BytesIO(audio_data), filename="tts.mp3")
                    await interaction.followup.send(file=audio_file)
                else:
                    error_text = await response.text()
                    await interaction.followup.send(f"TTS failed: {response.status} - {error_text[:50]}...")
        
    except Exception as e:
        await interaction.followup.send(f"TTS failed: {str(e)[:50]}...")

# YouTube search function
def search_youtube(query, max_retries=2):
    """Search YouTube and return video data with thumbnails and links."""
    if not youtube:
        return "YouTube API not available"

    import time

    for attempt in range(max_retries + 1):
        try:
            request = youtube.search().list(
                q=query,
                part="snippet",
                maxResults=5,
                type="video"
            )
            response = request.execute()

            if not response['items']:
                return None

            results = []
            for item in response['items']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                channel = item['snippet']['channelTitle']
                thumbnail = item['snippet']['thumbnails']['high']['url']
                url = f"https://youtube.com/watch?v={video_id}"

                results.append({
                    'title': title,
                    'channel': channel,
                    'url': url,
                    'thumbnail': thumbnail,
                    'video_id': video_id
                })

            return results

        except Exception as e:
            error_str = str(e)
            if "[WinError 10054]" in error_str or "ConnectionError" in error_str or "connection" in error_str.lower():
                if attempt < max_retries:
                    time.sleep(2)  # Wait 2 seconds before retry
                    continue
                return "YouTube search failed: Network connection error - please try again later"
            elif "quota" in error_str.lower() or "403" in error_str:
                return "YouTube search failed: API quota exceeded"
            elif "400" in error_str:
                return "YouTube search failed: Invalid API request"
            else:
                return f"YouTube search failed: {error_str}"

class JumpModal(discord.ui.Modal, title="Jump to Page"):
    page_num = discord.ui.TextInput(label="Page Number", placeholder="Enter a number...")

    def __init__(self, view):
        super().__init__()
        self.view = view

    async def on_submit(self, interaction: discord.Interaction):
        try:
            page = int(self.page_num.value) - 1
            if 0 <= page < len(self.view.results):
                self.view.current_page = page
                await self.view.update_message(interaction)
            else:
                await interaction.response.send_message(f"Invalid page! Please enter 1-{len(self.view.results)}", ephemeral=True)
        except ValueError:
            await interaction.response.send_message("Please enter a valid number!", ephemeral=True)

class YouTubeView(discord.ui.View):
    def __init__(self, results):
        super().__init__(timeout=300)
        self.results = results
        self.current_page = 0

    async def update_message(self, interaction):
        result = self.results[self.current_page]
        embed = discord.Embed(
            title=result['title'],
            description=f"Uploaded by **{result['channel']}**\n{result['url']}",
            color=0xff0000
        )
        embed.set_author(name=f"Page {self.current_page + 1} of {len(self.results)}")
        embed.set_image(url=result['thumbnail'])
        
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(label="Back", style=discord.ButtonStyle.secondary, emoji="◀️")
    async def back(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page > 0:
            self.current_page -= 1
            await self.update_message(interaction)
        else:
            await interaction.response.send_message("Already on first page!", ephemeral=True)

    @discord.ui.button(label="Forward", style=discord.ButtonStyle.secondary, emoji="▶️")
    async def forward(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page < len(self.results) - 1:
            self.current_page += 1
            await self.update_message(interaction)
        else:
            await interaction.response.send_message("Already on last page!", ephemeral=True)

    @discord.ui.button(label="Jump", style=discord.ButtonStyle.secondary, emoji="🔢")
    async def jump(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(JumpModal(self))

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await interaction.delete_original_response()

# Smart AI response with reasoning
def get_smart_response(user_query):
    system_prompt = """
You are an advanced reasoning engine. Before providing a final answer, 
you must follow these three steps:
1. BREAKDOWN: Identify the core components of the user's request.
2. REASONING: Solve the problem step-by-step in a hidden scratchpad.
3. CRITIQUE: Look for errors in your reasoning.

Format your response as follows:
<thinking> [Your step-by-step logic here] </thinking>
<final_answer> [Your actual response to the user] </final_answer>
"""

    response = ai_client.chat.completions.create(
        model="llama3.1-8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content

# Helper function for AI requests with timeout and retry
async def make_ai_request(messages, max_tokens=200, temperature=0.7, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: ai_client.chat.completions.create(
                        model="llama3.1-8b",
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                ),
                timeout=30.0
            )
            return response
        except asyncio.TimeoutError:
            if attempt < max_retries:
                print(f"AI request timed out, retrying ({attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(1)  # Wait 1 second before retry
            else:
                raise Exception("AI request timed out after retries")
        except Exception as e:
            if attempt < max_retries:
                print(f"AI request failed: {e}, retrying ({attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(1)
            else:
                raise e

# AI request using Cerebras (OpenAI-compatible) with smart reasoning
async def cerebras_ai_request(prompt, user_id=None):
    try:
        print(f"Making Cerebras AI request with prompt: {prompt[:50]}...")
        
        # Check for YouTube search requests
        youtube_keywords = ['youtube', 'video', 'watch', 'reviews', 'tutorial', 'how to']
        if any(keyword in prompt.lower() for keyword in youtube_keywords):
            # Generate search query
            search_response = await make_ai_request(
                messages=[{"role": "user", "content": f"Create a YouTube search query for: {prompt}"}],
                temperature=0
            )
            query = search_response.choices[0].message.content.strip('"')
            
            # Get YouTube results
            youtube_data = await asyncio.to_thread(search_youtube, query)
            
            # Generate final response with YouTube data
            final_response = await make_ai_request(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Use the YouTube data to answer the user."},
                    {"role": "assistant", "content": f"Here's what I found on YouTube:\n{youtube_data}"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            ai_response = final_response.choices[0].message.content
        
        # Use smart reasoning for complex queries
        elif any(word in prompt.lower() for word in ['calculate', 'solve', 'how many', 'what if', 'analyze', 'compare', 'reason']):
            try:
                ai_response = await asyncio.wait_for(
                    asyncio.to_thread(get_smart_response, prompt),
                    timeout=60.0  # Longer timeout for reasoning
                )
            except asyncio.TimeoutError:
                raise Exception("Smart reasoning timed out")
            
            # Extract final answer from thinking tags
            if '<final_answer>' in ai_response and '</final_answer>' in ai_response:
                start = ai_response.find('<final_answer>') + len('<final_answer>')
                end = ai_response.find('</final_answer>')
                ai_response = ai_response[start:end].strip()
        else:
            # Build messages with memory if user_id provided
            if user_id and user_id in user_memory:
                messages = [{"role": msg["role"], "content": msg["content"]} for msg in user_memory[user_id]] + [{"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]

            response = await make_ai_request(messages=messages, max_tokens=200, temperature=0.7)
            ai_response = response.choices[0].message.content
        
        # Store in memory if user_id provided
        if user_id:
            if user_id not in user_memory:
                user_memory[user_id] = []
            user_memory[user_id].append({"role": "user", "content": prompt, "timestamp": datetime.datetime.now(datetime.timezone.utc)})
            user_memory[user_id].append({"role": "assistant", "content": ai_response, "timestamp": datetime.datetime.now(datetime.timezone.utc)})
            # Keep only last 20 messages (10 exchanges)
            if len(user_memory[user_id]) > 20:
                user_memory[user_id] = user_memory[user_id][-20:]
        
        print(f"Cerebras AI response: {ai_response[:50]}...")
        return ai_response
    except Exception as e:
        print(f"Cerebras AI Error: {e}")
        return None

# Slash Command: AI Chat with cooldown
@bot.tree.command(name='ai', description='Chat with AI (supports text and optional image)')
@app_commands.describe(prompt="Your question or message to the AI")
async def ai(interaction: discord.Interaction, prompt: str, attachment: discord.Attachment = None):
    user_id = interaction.user.id
    
    # Create a lock for the user if it doesn't exist
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()
    
    # Try to acquire the lock
    if user_locks[user_id].locked():
        await interaction.response.send_message(f"⚠️ {interaction.user.mention}, I'm still processing your previous prompt! Please wait.", ephemeral=True)
        return
    
    # Check cooldown manually for slash commands (10 seconds)
    current_time = time.time()
    if user_id in ai_cooldowns and current_time - ai_cooldowns[user_id] < 10:
        retry_after = 10 - (current_time - ai_cooldowns[user_id])
        await interaction.response.send_message(f"Slow down! Try again in {retry_after:.2f}s.", ephemeral=True)
        return
    
    ai_cooldowns[user_id] = current_time
    await interaction.response.defer()
    
    async with user_locks[user_id]:
        try:
            # Check for image attachment
            image_data = None
            if attachment and any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'webp']):
                image_data = await get_image_bytes(attachment.url)
            
            # Handle image + text or just text
            if image_data and gemini_model:
                try:
                    contents = [
                        prompt,
                        {"mime_type": attachment.content_type, "data": image_data}
                    ]
                    response = await asyncio.to_thread(gemini_model.generate_content, contents)
                    ai_response = response.text
                except Exception as e:
                    print(f"Gemini Error: {e}")
                    ai_response = "I can see you sent an image, but I'm having trouble analyzing it right now! 📸😅"
            else:
                # Text-only response using Cerebras
                ai_response = await cerebras_ai_request(prompt, user_id)
            
            if ai_response:
                # Truncate if too long for Discord (2000 char limit)
                if len(ai_response) > 1900:
                    ai_response = ai_response[:1900] + "..."
                
                await interaction.followup.send(ai_response)
            else:
                await interaction.followup.send("AI service is currently unavailable. Please try again later.")

        except Exception as e:
            await interaction.followup.send(f"AI Error: {str(e)[:100]}...")



# Slash Command: Generate Image
@bot.tree.command(name='image', description='Generate an image using AI')
async def image(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    
    try:
        image_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=30) as response:
                if response.status == 200:
                    data = await response.read()
                    image_file = discord.File(io.BytesIO(data), filename="image.png")
                    await interaction.followup.send(file=image_file)
                else:
                    await interaction.followup.send("Image generation failed")
                
    except Exception as e:
        await interaction.followup.send(f"Image generation failed: {str(e)[:50]}...")



# Slash Command: Describe
@bot.tree.command(name='describe', description='Generate a description using AI')
@app_commands.describe(attachment="Attach an image to describe")
async def describe(interaction: discord.Interaction, attachment: discord.Attachment):
    """Command to describe an attached image"""

    # Verify it's an image file
    if any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'webp']):
        await interaction.response.defer()
        if not gemini_model:
            await interaction.followup.send("Image recognition is not available - Gemini API key not configured.")
            return

        # Download the image
        image_data = await get_image_bytes(attachment.url)

        if image_data:
            try:
                # Wrap the data for Gemini
                contents = [
                    "Describe this image in detail.",
                    {"mime_type": attachment.content_type, "data": image_data}
                ]

                # Generate the vision response
                response = await asyncio.to_thread(gemini_model.generate_content, contents)
                await interaction.followup.send(response.text)
            except Exception as e:
                error_msg = str(e)
                if "API key not valid" in error_msg or "400" in error_msg:
                    await interaction.followup.send("⚠️ **Configuration Error:** The Gemini API key is invalid. Please check your `.env` file.")
                else:
                    await interaction.followup.send(f"Failed to analyze image: {error_msg[:100]}...")
        else:
            await interaction.followup.send("Failed to download the image.")
    else:
        await interaction.response.send_message("That doesn't look like a supported image format.", ephemeral=True)

# Slash Command: Ship
@bot.tree.command(name='ship', description='Check the compatibility of two users')
async def ship(interaction: discord.Interaction, user1: discord.Member, user2: discord.Member):
    score = random.randint(0, 100)
    
    # Create a visual progress bar
    filled = int(score / 10)
    bar = "❤️" * filled + "💔" * (10 - filled)
    
    # Determine the status message
    if score >= 80:
        status = "It's a Match! 💍"
        color = discord.Color.red()
    elif score >= 50:
        status = "There's a spark! ✨"
        color = discord.Color.orange()
    else:
        status = "Just friends... for now. ☕"
        color = discord.Color.blue()

    embed = discord.Embed(
        title="💘 Love Calculator",
        color=color
    )
    
    embed.add_field(name="💑 Couple", value=f"{user1.mention} ❤️ {user2.mention}", inline=False)
    embed.add_field(name="📊 Compatibility Score", value=f"**{score}%**\n{bar}", inline=False)
    embed.add_field(name="🔮 Result", value=status, inline=False)
    embed.set_thumbnail(url="https://cdn.discordapp.com/emojis/1234567890123456789.png")  # Love/heart emoji
    embed.set_footer(text=f"Shipped by {interaction.user.display_name}", icon_url=interaction.user.display_avatar.url)
    embed.timestamp = datetime.datetime.now(datetime.timezone.utc)
    
    await interaction.response.send_message(embed=embed)

# Slash Command: Pet
@bot.tree.command(name='pet', description='Pet a specific animal!')
@app_commands.choices(animal=[
    app_commands.Choice(name="Dog", value="dog"),
    app_commands.Choice(name="Cat", value="cat"),
    app_commands.Choice(name="Bunny", value="bunny"),
    app_commands.Choice(name="Panda", value="panda")
])
async def pet(interaction: discord.Interaction, animal: app_commands.Choice[str]):
    # Data dictionary for each pet type
    pet_data = {
        "dog": {
            "title": "Good Boy! 🐶",
            "msg": "You gave the dog a belly rub!",
            "url": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOHYycW8xdm90c2Z3Z3R4eHh4eHh4eHh4eHh4eHh4eHh4eHh4JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/4T7eWG7jRmsTVEDKXO/giphy.gif",
            "color": discord.Color.gold()
        },
        "cat": {
            "title": "Purrfection 🐱",
            "msg": "The cat purrs and leans into your hand.",
            "url": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOHYycW8xdm90c2Z3Z3R4eHh4eHh4eHh4eHh4eHh4eHh4eHh4JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/3o72F7YT6s0EMFI0zk/giphy.gif",
            "color": discord.Color.purple()
        },
        "bunny": {
            "title": "Soft Hops 🐰",
            "msg": "You gently pat the bunny's soft ears.",
            "url": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOHYycW8xdm90c2Z3Z3R4eHh4eHh4eHh4eHh4eHh4eHh4eHh4JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/108GZES8iG0qcE/giphy.gif",
            "color": discord.Color.from_rgb(255, 192, 203)  # Pink
        },
        "panda": {
            "title": "Bamboo Friend 🐼",
            "msg": "The panda stops eating bamboo for a quick head pat.",
            "url": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOHYycW8xdm90c2Z3Z3R4eHh4eHh4eHh4eHh4eHh4eHh4eHh4JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/EatwJZRUIv41G/giphy.gif",
            "color": discord.Color.light_grey()
        }
    }

    # Get the data for the selected animal
    selected = pet_data[animal.value]

    # Build the Embed
    embed = discord.Embed(
        title=selected["title"],
        description=f"{interaction.user.mention}, {selected['msg']}",
        color=selected["color"]
    )
    embed.set_image(url=selected["url"])
    embed.set_footer(text=f"Petting a {animal.name}!")

    await interaction.response.send_message(embed=embed)

# Slash Command: Enable Quiet Mode
@bot.tree.command(name='quiet', description='Enable quiet mode (owner only)')
async def slash_quiet(interaction: discord.Interaction):
    owner_id = os.getenv('OWNER_ID')
    if not owner_id or interaction.user.id != int(owner_id):
        await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
        return
        
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
async def slash_unquiet(interaction: discord.Interaction):
    owner_id = os.getenv('OWNER_ID')
    if not owner_id or interaction.user.id != int(owner_id):
        await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
        return
        
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
@app_commands.guild_only()
async def setannounce(interaction: discord.Interaction, channel: discord.TextChannel):
    if not interaction.user.guild_permissions.manage_guild:
        await interaction.response.send_message("You need 'Manage Server' permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer()

    announce_channels[interaction.guild.id] = channel.id
    
    # Save to file
    try:
        with open(ANNOUNCE_FILE, 'w') as f:
            json.dump(announce_channels, f)
    except Exception as e:
        print(f"Error saving announce channels: {e}")

    embed = discord.Embed(
        title="Announcement Channel Set",
        description=f"Online announcements will be sent to {channel.mention}",
        color=discord.Color.green()
    )
    embed.set_footer(text=f"Set by {interaction.user}")
    await interaction.followup.send(embed=embed)

# Slash Command: Invite Track (Set Welcome Channel)
@bot.tree.command(name='invite-track', description='Set the welcome channel where join messages are sent (manage guild permission required)')
async def invite_track(interaction: discord.Interaction, channel: discord.TextChannel):
    if not interaction.user.guild_permissions.manage_guild:
        await interaction.response.send_message("You need 'Manage Server' permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer()

    welcome_channels[interaction.guild.id] = channel.id
    embed = discord.Embed(
        title="Welcome Channel Set",
        description=f"Welcome messages will be sent to {channel.mention}",
        color=discord.Color.blue()
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
        if inviter_id:
            inviter = interaction.guild.get_member(inviter_id)
            inviter_name = inviter.mention if inviter else f"User {inviter_id}"
        else:
            inviter_name = "Unknown"
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
        user_name = user.mention if user else f"User {user_id}"
        if inviter_id:
            inviter = interaction.guild.get_member(inviter_id)
            inviter_name = inviter.mention if inviter else f"User {inviter_id}"
        else:
            inviter_name = "Unknown"
        join_list += f"{user_name} invited by {inviter_name} at {timestamp}\n"

    if join_list:
        embed.add_field(name="Recent Joins", value=join_list, inline=False)
    else:
        embed.add_field(name="Recent Joins", value="None", inline=False)

    embed.set_footer(text=f"Requested by {interaction.user}")
    await interaction.followup.send(embed=embed)

# Slash Command: Joke
@bot.tree.command(name='joke', description='Tell a random joke')
async def joke(interaction: discord.Interaction):
    urls = [
        "https://v2.jokeapi.dev/joke/Any?safe-mode&type=single",
        "https://official-joke-api.appspot.com/random_joke"
    ]
    
    try:
        selected_url = random.choice(urls)
        async with aiohttp.ClientSession() as session:
            async with session.get(selected_url) as resp:
                response = await resp.json()
                if "joke" in response:
                    joke_text = response["joke"]
                else:
                    joke_text = f"{response['setup']}\n\n*... {response['punchline']}*"
        
        await interaction.response.send_message(f"😂 {joke_text}")
    except:
        fallback_jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? He was outstanding in his field!",
            "What do you call a fake noodle? An impasta!"
        ]
        await interaction.response.send_message(f"😂 {random.choice(fallback_jokes)}")



# Slash Command: Purge
@bot.tree.command(name='purge', description='Clears specified number of messages')
@app_commands.checks.has_permissions(manage_messages=True)
@app_commands.describe(amount="Number of messages to delete")
async def purge(interaction: discord.Interaction, amount: int):
    await interaction.response.defer(ephemeral=True)
    deleted = await interaction.channel.purge(limit=amount + 1)
    await interaction.followup.send(f'{len(deleted) - 1} messages have been cleared!', ephemeral=True)

# Slash Command: Server Info
@bot.tree.command(name='serverinfo', description='Displays information about the server')
async def server_info(interaction: discord.Interaction):
    guild = interaction.guild
    embed = discord.Embed(title=f'{guild.name} Server Information', color=discord.Color.blue())
    embed.add_field(name='Server Owner', value=guild.owner.mention, inline=False)
    embed.add_field(name='Member Count', value=guild.member_count, inline=True)
    embed.add_field(name='Channel Count', value=len(guild.channels), inline=True)
    embed.add_field(name='Server Created', value=guild.created_at.strftime('%B %d, %Y'), inline=False)
    embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
    await interaction.response.send_message(embed=embed)


# Slash Command: Info (user profile + badges)
@bot.tree.command(name='info', description='Shows profile information and badges for a user')
async def info(interaction: discord.Interaction, member: discord.Member = None):
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

        # Joined date
        joined = getattr(user, 'joined_at', None)
        if joined:
            embed.add_field(name='Joined Server', value=joined.strftime('%Y-%m-%d %H:%M'), inline=False)

    # Badges / Public flags - Display ALL badges
    flags = getattr(user, 'public_flags', None)
    # Comprehensive mapping for ALL badge names with emojis
    badge_map = {
        'staff': '👨‍💼 Discord Staff',
        'partner': '🤝 Partnered Server Owner',
        'hypesquad': '🏠 HypeSquad Events',
        'bug_hunter': '🐛 Bug Hunter',
        'bug_hunter_level_2': '🐞 Bug Hunter Level 2',
        'hypesquad_bravery': '💪 HypeSquad Bravery',
        'hypesquad_brilliance': '💡 HypeSquad Brilliance',
        'hypesquad_balance': '⚖️ HypeSquad Balance',
        'early_supporter': '👑 Early Supporter',
        'early_verified_bot_developer': '🤖 Early Verified Bot Developer',
        'team_user': '👥 Team User',
        'verified_bot_developer': '🤖 Verified Bot Developer',
        'system': '⚙️ System',
        'premium_subscriber': '✨ Nitro Subscriber',
        'active_developer': '💻 Active Developer',
        'discord_employee': '🏢 Discord Employee',
        'certified_moderator': '🛡️ Certified Moderator',
        'bot_http_interactions': '🤖 Bot HTTP Interactions',
        'spammer': '🚫 Spammer',
        # Quest badges
        'quest_apprentice': '🔮 Quest Apprentice',
        'quest_journeyman': '🧙 Quest Journeyman',
        'quest_master': '🧙‍♂️ Quest Master',
        'quest_hero': '🏆 Quest Hero',
        'quest_legend': '👑 Quest Legend',
        # Other special badges
        'legacy_username': '📜 Legacy Username',
        'username_changeable': '🔄 Username Changeable',
        'guild_booster_lvl_1': '🚀 Server Booster Level 1',
        'guild_booster_lvl_2': '🚀 Server Booster Level 2',
        'guild_booster_lvl_3': '🚀 Server Booster Level 3',
        'guild_booster_lvl_4': '🚀 Server Booster Level 4',
        'guild_booster_lvl_5': '🚀 Server Booster Level 5',
        'guild_booster_lvl_6': '🚀 Server Booster Level 6',
        'guild_booster_lvl_7': '🚀 Server Booster Level 7',
        'guild_booster_lvl_8': '🚀 Server Booster Level 8',
        'guild_booster_lvl_9': '🚀 Server Booster Level 9'
    }

    badges = []
    if flags is not None:
        # Inspect all attributes on the flags object and collect ALL truthy boolean flags
        for attr in dir(flags):
            if attr.startswith('_'):
                continue
            try:
                val = getattr(flags, attr)
            except Exception:
                continue
            if isinstance(val, bool) and val:
                # Use mapped name if available, otherwise format the attribute name
                label = badge_map.get(attr, attr.replace('_', ' ').title())
                badges.append(label)

        # Also check for any integer flags that might represent special badges
        try:
            if hasattr(flags, 'value') and flags.value:
                # Check for special flag combinations
                if flags.value & 1:  # Discord Employee
                    if '🏢 Discord Employee' not in badges:
                        badges.append('🏢 Discord Employee')
                if flags.value & 2:  # Discord Partner
                    if '🤝 Partnered Server Owner' not in badges:
                        badges.append('🤝 Partnered Server Owner')
                if flags.value & 4:  # HypeSquad Events
                    if '🏠 HypeSquad Events' not in badges:
                        badges.append('🏠 HypeSquad Events')
                if flags.value & 8:  # Bug Hunter Level 1
                    if '🐛 Bug Hunter' not in badges:
                        badges.append('🐛 Bug Hunter')
                if flags.value & 64:  # House Bravery
                    if '💪 HypeSquad Bravery' not in badges:
                        badges.append('💪 HypeSquad Bravery')
                if flags.value & 128:  # House Brilliance
                    if '💡 HypeSquad Brilliance' not in badges:
                        badges.append('💡 HypeSquad Brilliance')
                if flags.value & 256:  # House Balance
                    if '⚖️ HypeSquad Balance' not in badges:
                        badges.append('⚖️ HypeSquad Balance')
                if flags.value & 512:  # Early Supporter
                    if '👑 Early Supporter' not in badges:
                        badges.append('👑 Early Supporter')
                if flags.value & 1024:  # Team User
                    if '👥 Team User' not in badges:
                        badges.append('👥 Team User')
                if flags.value & 2048:  # System
                    if '⚙️ System' not in badges:
                        badges.append('⚙️ System')
                if flags.value & 4096:  # Bug Hunter Level 2
                    if '🐞 Bug Hunter Level 2' not in badges:
                        badges.append('🐞 Bug Hunter Level 2')
                if flags.value & 8192:  # Verified Bot
                    if '🤖 Verified Bot Developer' not in badges:
                        badges.append('🤖 Verified Bot Developer')
                if flags.value & 16384:  # Early Verified Bot Developer
                    if '🤖 Early Verified Bot Developer' not in badges:
                        badges.append('🤖 Early Verified Bot Developer')
                if flags.value & 262144:  # Certified Moderator
                    if '🛡️ Certified Moderator' not in badges:
                        badges.append('🛡️ Certified Moderator')
                if flags.value & 4194304:  # Active Developer
                    if '💻 Active Developer' not in badges:
                        badges.append('💻 Active Developer')
        except Exception:
            pass  # Ignore any errors in flag parsing

    # Deduplicate while preserving order
    seen = set()
    unique_badges = []
    for b in badges:
        if b not in seen:
            unique_badges.append(b)
            seen.add(b)
    badges = unique_badges

    embed.add_field(name='Badges', value=', '.join(badges) if badges else 'None', inline=False)

    # ID and mention
    embed.set_footer(text=f'User ID: {user.id} | Requested by {interaction.user}', icon_url=interaction.user.display_avatar.url)
    await interaction.response.send_message(embed=embed)



# Slash Command: Set Status (Owner Only)
@bot.tree.command(name='setstatus', description='Set bot status (Owner only)')
@app_commands.describe(status="The status text to set")
async def setstatus(interaction: discord.Interaction, status: str):
    owner_id = os.getenv('OWNER_ID')
    if owner_id and interaction.user.id != int(owner_id):
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return
    
    try:
        custom_activity = discord.CustomActivity(name=status)
        await bot.change_presence(activity=custom_activity)
        await interaction.response.send_message(f"Status set to: {status}")
    except Exception as e:
        await interaction.response.send_message(f"Failed to set status: {str(e)}", ephemeral=True)



# Slash Command: Sync (owner only)
@bot.tree.command(name='sync', description='Sync slash commands (owner only)')
async def sync(interaction: discord.Interaction):
    owner_id = os.getenv('OWNER_ID')
    if owner_id and interaction.user.id != int(owner_id):
        await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
        return
    
    try:
        await bot.tree.sync()
        await interaction.response.send_message("✅ Slash commands synced successfully!")
    except Exception as e:
        await interaction.response.send_message(f"❌ Failed to sync: {e}", ephemeral=True)

# Slash Command: Close (owner only)
@bot.tree.command(name='close', description='Shutdown the bot (owner only)')
async def close_bot(interaction: discord.Interaction):
    owner_id = os.getenv('OWNER_ID')
    if owner_id and interaction.user.id != int(owner_id):
        await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
        return
    
    await interaction.response.send_message("🛑 Shutting down bot...")
    await bot.close()



# Slash Command: Kick
@bot.tree.command(name='kick', description='Kicks a member from the server')
@app_commands.checks.has_permissions(kick_members=True)
@app_commands.describe(member="Member to kick", reason="Reason for kicking")
async def kick(interaction: discord.Interaction, member: discord.Member, reason: str = None):
    try:
        await member.kick(reason=reason)
        embed = discord.Embed(
            title="👢 Member Kicked",
            description=f"{member.mention} has been kicked from the server.",
            color=discord.Color.red()
        )
        embed.add_field(name="Reason", value=reason or "No reason provided")
        embed.set_footer(text=f"Kicked by {interaction.user}")
        await interaction.response.send_message(embed=embed)
    except discord.Forbidden:
        await interaction.response.send_message("I don't have permission to kick members!", ephemeral=True)

# Slash Command: Ban
@bot.tree.command(name='ban', description='Bans a member from the server')
@app_commands.checks.has_permissions(ban_members=True)
@app_commands.describe(member="Member to ban", reason="Reason for banning")
async def ban(interaction: discord.Interaction, member: discord.Member, reason: str = None):
    try:
        await member.ban(reason=reason)
        embed = discord.Embed(
            title="🔨 Member Banned",
            description=f"{member.mention} has been banned from the server.",
            color=discord.Color.dark_red()
        )
        embed.add_field(name="Reason", value=reason or "No reason provided")
        embed.set_footer(text=f"Banned by {interaction.user}")
        await interaction.response.send_message(embed=embed)
    except discord.Forbidden:
        await interaction.response.send_message("I don't have permission to ban members!", ephemeral=True)

# Slash Command: Timeout
@bot.tree.command(name='timeout', description='Timeout (mute) a member for a specified duration (in minutes)')
@app_commands.checks.has_permissions(moderate_members=True)
@app_commands.describe(member="Member to timeout", duration="Duration in minutes", reason="Reason for timeout")
async def timeout(interaction: discord.Interaction, member: discord.Member, duration: int, reason: str = None):
    try:
        # Convert duration from minutes to seconds and create timedelta
        duration_seconds = duration * 60
        await member.timeout(datetime.timedelta(seconds=duration_seconds), reason=reason)
        embed = discord.Embed(
            title="⏰ Member Timed Out",
            description=f"{member.mention} has been timed out for {duration} minutes.",
            color=discord.Color.orange()
        )
        embed.add_field(name="Reason", value=reason or "No reason provided")
        embed.set_footer(text=f"Timed out by {interaction.user}")
        await interaction.response.send_message(embed=embed)
    except discord.Forbidden:
        await interaction.response.send_message("I don't have permission to timeout members!", ephemeral=True)

# Slash Command: Untimeout
@bot.tree.command(name='untimeout', description='Remove timeout from a member')
@app_commands.checks.has_permissions(moderate_members=True)
@app_commands.describe(member="Member to untimeout", reason="Reason for removing timeout")
async def untimeout(interaction: discord.Interaction, member: discord.Member, reason: str = None):
    try:
        await member.remove_timeout(reason=reason)
        embed = discord.Embed(
            title="🔄 Timeout Removed",
            description=f"Timeout has been removed from {member.mention}.",
            color=discord.Color.green()
        )
        embed.add_field(name="Reason", value=reason or "No reason provided")
        embed.set_footer(text=f"Timeout removed by {interaction.user}")
        await interaction.response.send_message(embed=embed)
    except discord.Forbidden:
        await interaction.response.send_message("I don't have permission to remove timeouts!", ephemeral=True)

# Slash Command: Unban
@bot.tree.command(name='unban', description='Unbans a member from the server')
@app_commands.checks.has_permissions(ban_members=True)
@app_commands.describe(member="Username or Username#Discriminator of the banned user")
async def unban(interaction: discord.Interaction, member: str):
    try:
        # Find the banned user
        banned_users_list = [entry async for entry in interaction.guild.bans()]
        
        # Handle both old format (name#discriminator) and new format (username)
        if '#' in member:
            try:
                member_name, member_discriminator = member.split('#')
                for ban_entry in banned_users_list:
                    user = ban_entry.user
                    if (user.name, str(user.discriminator)) == (member_name, member_discriminator):
                        await interaction.guild.unban(user)
                        embed = discord.Embed(
                            title="🔓 Member Unbanned",
                            description=f"{user.mention} has been unbanned from the server.",
                            color=discord.Color.green()
                        )
                        embed.set_footer(text=f"Unbanned by {interaction.user}")
                        await interaction.response.send_message(embed=embed)
                        return
            except ValueError:
                pass
        
        # New username format or fallback
        for ban_entry in banned_users_list:
            user = ban_entry.user
            if user.name == member or str(user) == member:
                await interaction.guild.unban(user)
                embed = discord.Embed(
                    title="🔓 Member Unbanned",
                    description=f"{user.mention} has been unbanned from the server.",
                    color=discord.Color.green()
                )
                embed.set_footer(text=f"Unbanned by {interaction.user}")
                await interaction.response.send_message(embed=embed)
                return
        
        await interaction.response.send_message(f"Could not find banned user {member}", ephemeral=True)
    except discord.Forbidden:
        await interaction.response.send_message("I don't have permission to unban members!", ephemeral=True)

# Slash Command: Game (Word Minesweeper)
@bot.tree.command(name='game', description='Start a word-based minesweeper game')
async def slash_game(interaction: discord.Interaction):
    # Defer the response to prevent timeout errors
    await interaction.response.defer()

    # Check if user already has a game
    user_id = interaction.user.id
    if user_id in game_state:
        await interaction.followup.send("You already have a game in progress! Finish it first.")
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
    embed.set_footer(text="Use /reveal to reveal a cell (1-5)")
    await interaction.followup.send(embed=embed)



# Slash Command: Reveal cell
@bot.tree.command(name='reveal', description='Reveal a cell in the minesweeper game')
async def slash_reveal(interaction: discord.Interaction, row: int, col: int):
    # Defer the response to prevent timeout errors
    await interaction.response.defer()

    user_id = interaction.user.id
    if user_id not in game_state:
        await interaction.followup.send("You don't have a game in progress! Start one with /game")
        return

    game = game_state[user_id]
    row -= 1  # 0-indexed
    col -= 1

    if not (0 <= row < 5 and 0 <= col < 5):
        await interaction.followup.send("Invalid coordinates! Use 1-5 for row and column.")
        return

    if (row, col) in game['revealed']:
        await interaction.followup.send("That cell is already revealed!")
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
        await interaction.followup.send(embed=embed)
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
        await interaction.followup.send(embed=embed)
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
    embed.set_footer(text="Use /reveal to reveal a cell (1-5)")
    await interaction.followup.send(embed=embed)



# Slash Command: Hide (P Diddy Hide-and-Seek Game)
@bot.tree.command(name='hide', description='Play hide-and-seek with P Diddy')
async def slash_hide(interaction: discord.Interaction, choice: int = None):
    # Defer the response to prevent timeout errors
    await interaction.response.defer()

    user_id = interaction.user.id

    if choice is None:
        # Display the status and available spots
        status = get_status_message(user_id)
        embed = discord.Embed(
            title="P Diddy is counting!",
            description=status,
            color=discord.Color.blue()
        )
        await interaction.followup.send(embed=embed)
    else:
        # Run the game logic
        message, _ = run_grand_escape_round(user_id, choice)
        await interaction.followup.send(message)



# Note: j!search command removed per user request.

# Slash Command: Pulse (Channel Vibe Check)
@bot.tree.command(name="pulse", description="Analyzes the current 'vibe' of the channel")
async def pulse(interaction: discord.Interaction):
    await interaction.response.defer()
    
    try:
        # Gather history
        history = []
        async for message in interaction.channel.history(limit=50):
            if not message.author.bot:
                history.append(f"{message.author.display_name}: {message.content}")
        
        if not history:
            await interaction.followup.send("📊 **Channel Pulse:** No recent messages to analyze!")
            return
        
        chat_context = "\n".join(history)
        prompt = f"Analyze the following chat log and give a brief 'vibe check' (2 sentences max) and a sentiment percentage:\n\n{chat_context}"
        
        analysis = await cerebras_ai_request(prompt)
        if analysis:
            await interaction.followup.send(f"📊 **Channel Pulse:**\n{analysis}")
        else:
            await interaction.followup.send("📊 **Channel Pulse:** AI analysis is currently unavailable.")
        
    except Exception as e:
        await interaction.followup.send(f"😅 Oops, pulse check went a bit wonky: {str(e)[:50]}... no worries though! 🤷‍♂️")

# Slash Command: Recap
@bot.tree.command(name='recap', description='Get a recap of topics discussed with AI today')
async def recap(interaction: discord.Interaction):
    await interaction.response.defer()

    user_id = interaction.user.id
    today = datetime.datetime.now(datetime.timezone.utc).date()

    if user_id not in user_memory or not user_memory[user_id]:
        await interaction.followup.send("You haven't had any AI conversations today!")
        return

    # Filter messages from today
    todays_messages = [msg for msg in user_memory[user_id] if msg.get('timestamp', datetime.datetime.min).date() == today]

    if not todays_messages:
        await interaction.followup.send("You haven't had any AI conversations today!")
        return

    # Collect user prompts from today
    user_prompts = [msg['content'] for msg in todays_messages if msg['role'] == 'user']

    if not user_prompts:
        await interaction.followup.send("No user prompts found today.")
        return

    # Create a summary prompt
    conversation_text = "\n".join(user_prompts[:10])  # Limit to last 10 to avoid too long
    summary_prompt = f"Summarize the main topics discussed in these AI conversations today: {conversation_text}"

    # Use AI to summarize
    recap_text = await cerebras_ai_request(summary_prompt)

    if recap_text:
        embed = discord.Embed(title="📝 AI Conversation Recap", description=recap_text, color=discord.Color.blue())
        embed.set_footer(text=f"Recap for {interaction.user.display_name} - Today")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("Unable to generate recap right now.")

# Slash Command: Timezone
@bot.tree.command(name="timezone", description="Convert a time to a dynamic Discord tag for everyone")
@app_commands.describe(time_input="e.g., 'tomorrow at 5pm' or 'Friday 10:30am'")
async def timezone(interaction: discord.Interaction, time_input: str):
    try:
        dt = parser.parse(time_input, fuzzy=True, default=datetime.datetime.now())
        unix_time = int(dt.timestamp())
        discord_tag = f"<t:{unix_time}:F> (<t:{unix_time}:R>)"
        
        await interaction.response.send_message(
            f"🕐 **Time Converted - pretty neat, right?** ✨\nGlobal Tag: `{discord_tag}`\nYour result: {discord_tag} 😎"
        )
    except Exception:
        await interaction.response.send_message("🤔 Hmm, that time format's got me scratching my head... Try something like 'Oct 20 5pm' or 'tomorrow 3pm' and we'll be golden! 😊", ephemeral=True)

# Slash Command: Feedback
@bot.tree.command(name="feedback", description="Send feedback, bug reports, or ideas directly to the bot owner")
@app_commands.describe(
    category="Select the type of feedback",
    message="Your feedback message", 
    command_name="Which command failed? (Required for Command execution failure)",
    anonymous="Send feedback anonymously", 
    attachment="Optional file attachment"
)
@app_commands.choices(category=[
    app_commands.Choice(name="General Feedback", value="general"),
    app_commands.Choice(name="Bug Report", value="bug"),
    app_commands.Choice(name="Feature Request", value="feature"),
    app_commands.Choice(name="Command execution failure", value="command_error")
])
async def feedback(interaction: discord.Interaction, category: app_commands.Choice[str], message: str, command_name: str = None, anonymous: bool = False, attachment: discord.Attachment = None):
    await interaction.response.defer(ephemeral=True)
    
    # Check if command_name is required for command_error category
    if category.value == "command_error" and not command_name:
        await interaction.followup.send("🤷‍♂️ Hey, I'm gonna need that command name when reporting failures - helps me figure out what went sideways! 😅", ephemeral=True)
        return
    
    owner_id = int(os.getenv('OWNER_ID', '1231525871257649213'))
    owner = bot.get_user(owner_id)
    if not owner:
        await interaction.followup.send("😬 Can't find the boss anywhere... might wanna slide into their DMs the old school way! 📱😂", ephemeral=True)
        return

    embed = discord.Embed(
        title=f"✨ {category.name}",
        description=message,
        color=discord.Color.green(),
        timestamp=datetime.datetime.now(datetime.timezone.utc)
    )
    
    if category.value == "command_error":
        embed.add_field(name="Failed Command", value=f"`{command_name}`", inline=True)

    if not anonymous:
        embed.set_author(name=interaction.user, icon_url=interaction.user.avatar.url if interaction.user.avatar else None)
        user_text = f"From **{interaction.user}** ({interaction.user.id})"
    else:
        embed.set_author(name="Anonymous User")
        user_text = f"From User ID: {interaction.user.id}"

    server = interaction.guild.name if interaction.guild else "Direct Messages"
    embed.add_field(name="Server", value=server, inline=True)
    embed.add_field(name="User", value=user_text, inline=True)

    files = []
    if attachment:
        file = await attachment.to_file()
        files.append(file)
        embed.set_image(url=f"attachment://{file.filename}")

    await owner.send(embed=embed, files=files)
    await interaction.followup.send("🚀 Boom! Feedback delivered straight to the boss's inbox! Thanks for keeping it real ❤️✨", ephemeral=True)

# Slash Command: Topic
@bot.tree.command(name='topic', description='Get a random conversation starter')
async def topic(interaction: discord.Interaction):
    topics = [
        "What is the most underrated movie of all time?",
        "If you could have dinner with any historical figure, who would it be?",
        "What's the weirdest food combination you actually enjoy?",
        "If you won the lottery tomorrow, what's the first thing you'd buy?",
        "What is your 'hot take' that everyone else disagrees with?",
        "What's a hobby you've always wanted to pick up but haven't?"
    ]
    selected_topic = random.choice(topics)
    await interaction.response.send_message(f"💭 **Here's something to chat about:** {selected_topic} 🤔✨")

# Slash Command: Remind
@bot.tree.command(name='remind', description='Set a reminder')
@app_commands.describe(seconds="How many seconds until the reminder?", task="What should I remind you about?")
async def remind(interaction: discord.Interaction, seconds: int, task: str):
    await interaction.response.send_message(f"📝 Got it! I'll ping you about '{task}' in {seconds} seconds - don't worry, I won't forget! 😉⏰")
    
    # Create a background task for the reminder
    async def send_reminder():
        await asyncio.sleep(seconds)
        try:
            await interaction.followup.send(f"🔔 **Yo {interaction.user.mention}!** Time's up for: {task} ⏰✨")
        except:
            pass  # Ignore if followup fails
    
    # Start the reminder task in the background
    asyncio.create_task(send_reminder())

# Warnings storage
user_warnings = {}

# Slash Command: Warnings
@bot.tree.command(name='warnings', description='Display warnings for a user')
@app_commands.describe(user="User to check warnings for")
async def warnings(interaction: discord.Interaction, user: discord.Member):
    guild_id = interaction.guild.id
    user_id = user.id

    warnings_list = user_warnings.get(guild_id, {}).get(user_id, [])

    embed = discord.Embed(title=f"Warnings for {user.display_name}", color=discord.Color.orange())
    
    if not warnings_list:
        embed.description = "No warnings found for this user."
    else:
        for idx, warning in enumerate(warnings_list, 1):
            # Assuming warning structure is dict or string. 
            # Based on previous code it seemed to be just strings or objects.
            # Adapting to generic display:
            embed.add_field(name=f"Warning {idx}", value=str(warning), inline=False)
            
    embed.set_footer(text=f"Requested by {interaction.user}")
    await interaction.response.send_message(embed=embed)


# Slash Command: YouTube Search
@bot.tree.command(name='youtube', description='Search YouTube videos')
@app_commands.describe(query="What to search for on YouTube")
async def youtube_search(interaction: discord.Interaction, query: str):
    await interaction.response.defer()
    
    if not youtube:
        await interaction.followup.send("❌ YouTube API not configured")
        return
    
    try:
        results = await asyncio.to_thread(search_youtube, query)
        if isinstance(results, str):
            await interaction.followup.send(f"❌ {results}")
            return
        
        if not results:
            await interaction.followup.send("❌ No videos found")
            return
        
        # Create embed for first result
        result = results[0]
        embed = discord.Embed(
            title=result['title'],
            description=f"Uploaded by **{result['channel']}**\n{result['url']}",
            color=0xff0000
        )
        embed.set_author(name=f"Page 1 of {len(results)}")
        embed.set_image(url=result['thumbnail'])
        embed.set_footer(text=f"Searched by {interaction.user}")
        
        view = YouTubeView(results)
        await interaction.followup.send(embed=embed, view=view)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Search failed: {str(e)[:50]}...")

# Slash Command: Daily (for badge activity)
@bot.tree.command(name='daily', description='Get your daily motivation!')
async def daily(interaction: discord.Interaction):
    try:
        with open('daily_index.txt', 'r', encoding='utf-8') as f:
            quotes = [line.strip() for line in f if line.strip()]
        if not quotes:
            await interaction.response.send_message("No quotes available right now!")
            return
    except FileNotFoundError:
        await interaction.response.send_message("Quotes file not found!")
        return

    # Load current index
    try:
        with open('current_daily_index.txt', 'r') as f:
            current_index = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        current_index = 0

    # Get the quote
    quote = quotes[current_index]

    # Update index for next time
    next_index = (current_index + 1) % len(quotes)
    with open('current_daily_index.txt', 'w') as f:
        f.write(str(next_index))

    await interaction.response.send_message(quote)

# Slash Command: AFK
@bot.tree.command(name='afk', description='Add [AFK] to your nickname with optional reason')
@app_commands.describe(reason="Optional reason for being AFK")
async def afk(interaction: discord.Interaction, reason: str = None):

    try:
        user_id = interaction.user.id
        if user_id in afk_users:
            await interaction.response.send_message("😴 You're already marked as AFK! 💤", ephemeral=True)
            return

        current_nick = interaction.user.display_name
        afk_users[user_id] = {'original_nick': current_nick, 'reason': reason}

        if reason:
            new_nick = f"[AFK] {reason}"
            await interaction.response.send_message(f"😴 You're now marked as AFK: {reason}! Sweet dreams! 💤✨")
        else:
            new_nick = f"[AFK] {current_nick}"
            await interaction.response.send_message("😴 You're now marked as AFK! Sweet dreams! 💤✨")

        await interaction.user.edit(nick=new_nick)
    except discord.Forbidden:
        await interaction.response.send_message("😅 I don't have permission to change your nickname... ask an admin to hook me up! 🤷♂️", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f"😬 Something went wrong: {str(e)}", ephemeral=True)

# Slash Command: Countdown
@bot.tree.command(name='countdown', description='Start a live countdown timer')
@app_commands.describe(
    days="Number of days",
    hours="Number of hours",
    minutes="Number of minutes",
    seconds="Number of seconds",
    year="Target year (optional)",
    month="Target month 1-12 (optional)",
    day="Target day (optional)"
)
async def countdown(interaction: discord.Interaction, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0, year: int = None, month: int = None, day: int = None):
    if year and month and day:
        try:
            target_date = datetime.datetime(year, month, day)
            timestamp = int(target_date.timestamp())
            await interaction.response.send_message(f"🎯 Your event is happening <t:{timestamp}:R> - mark your calendar! 📅✨")
            return
        except ValueError:
            await interaction.response.send_message("📅 That date's looking a bit funky... double-check those numbers for me! 😅", ephemeral=True)
            return

    total_seconds = (days * 86400) + (hours * 3600) + (minutes * 60) + seconds

    if total_seconds <= 0:
        await interaction.response.send_message("⏱️ I'm gonna need some actual time to count down from... zero doesn't really work here! 😄", ephemeral=True)
        return

    # Calculate target timestamp
    now = datetime.datetime.now(datetime.timezone.utc)
    target_time = now + datetime.timedelta(seconds=total_seconds)
    timestamp = int(target_time.timestamp())

    await interaction.response.send_message(f"🚀 **Countdown started!**\nTime remaining: <t:{timestamp}:R>\nEnds at: <t:{timestamp}:T>")

    await asyncio.sleep(total_seconds)

    try:
        await interaction.edit_original_response(content=f"🎉 **Time's up!** Finished <t:{timestamp}:R>! ⏰✨")
        await interaction.followup.send(f"🔔 Ding ding! {interaction.user.mention}, your countdown just wrapped up! 🎯✨")
    except discord.NotFound:
        pass

# Slash Command: Debate
@bot.tree.command(name='debate', description='The bot argues with itself over a topic')
async def debate(interaction: discord.Interaction, topic: str):
    await interaction.response.defer()  # AI takes a few seconds

    if not gemini_model:
        await interaction.followup.send("Debate feature is not available - Gemini API key not configured.")
        return

    prompt = (f"Write a short, heated debate about '{topic}'. "
              "Provide 'Side A' and 'Side B' separately. Keep each side under 300 characters.")

    response = await asyncio.to_thread(gemini_model.generate_content, prompt)
    # Simple split logic - you can get fancier with regex
    text = response.text

    embed = discord.Embed(title=f"⚖️ The Great Debate: {topic}", color=0x2b2d31)
    embed.description = text
    embed.set_footer(text="Who won? React with 🔴 or 🔵")

    message = await interaction.followup.send(embed=embed)
    await message.add_reaction("🔴")
    await message.add_reaction("🔵")

# Slash Command: Impulse
@bot.tree.command(name='impulse', description='Drops a hot take and freezes the chat')
@app_commands.checks.has_permissions(manage_channels=True)
async def impulse(interaction: discord.Interaction):
    takes = [
        "Water is not wet. It makes things wet.",
        "The letter 'W' starts with a 'D'. Why?",
        "If we admit that tomatoes are fruits, we must admit ketchup is a smoothie.",
        "Pineapple on pizza is a culinary masterpiece."
    ]

    hot_take = random.choice(takes)
    channel = interaction.channel
    everyone = interaction.guild.default_role

    # 1. Lock the channel
    await channel.set_permissions(everyone, send_messages=False)

    await interaction.response.send_message(
        f"🚨 **IMPULSE TOPIC:** \n> {hot_take}\n\n*Chat frozen for 15s. Process your emotions.*"
    )

    # 2. The Timer
    await asyncio.sleep(15)

    # 3. Unlock the channel
    await channel.set_permissions(everyone, send_messages=None)  # Reset to default
    await interaction.followup.send("🔓 **UNLOCKED.** Let the chaos begin.")

# Error handling if user doesn't have permissions
@impulse.error
async def impulse_error(interaction: discord.Interaction, error):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("You don't have permission to freeze the chat!", ephemeral=True)



# Safe send function to handle Discord's 2000 character limit
async def safe_send(ctx, content):
    # Discord has a 2000 character limit per message
    if len(content) > 2000:
        await ctx.send("📝 Whoa, that result's way too chunky for one message! Maybe try something shorter? 😅")
    else:
        await ctx.send(content)

# Slash Command: Help
@bot.tree.command(name='help', description='Show all available commands')
async def show_commands(interaction: discord.Interaction):
    class CommandSelect(discord.ui.Select):
        def __init__(self):
            options = [
                discord.SelectOption(label="🎮 Fun & Games", description="Games, jokes, pets, debates", emoji="🎮"),
                discord.SelectOption(label="🤖 AI & Chat", description="AI chat, TTS, image analysis", emoji="🤖"),
                discord.SelectOption(label="🖼️ Media", description="Images, YouTube, daily quotes", emoji="🖼️"),
                discord.SelectOption(label="⚙️ Utility", description="Ping, reminders, timezone", emoji="⚙️"),
                discord.SelectOption(label="🔧 Administration", description="Bot settings, channels", emoji="🔧"),
                discord.SelectOption(label="🛡️ Moderation", description="Kick, ban, purge, warnings", emoji="🛡️"),
                discord.SelectOption(label="ℹ️ Information", description="Server info, user profiles", emoji="ℹ️")
            ]
            super().__init__(placeholder="Choose a command category...", options=options)

        async def callback(self, interaction: discord.Interaction):
            selected = self.values[0]
            if selected == "🎮 Fun & Games":
                embed = discord.Embed(title="🎮 Fun & Games Commands", color=discord.Color.gold())
                embed.description = (
                    "```/ship <user1> <user2> - Check compatibility between two users\n"
                    "/pet <animal> - Pet a specific animal (dog, cat, bunny, panda)\n"
                    "/joke - Tell a random joke\n"
                    "/topic - Get a random conversation starter\n"
                    "/game - Start a word-based minesweeper game\n"
                    "/reveal <row> <col> - Reveal a cell in the minesweeper game\n"
                    "/hide [choice] - Play hide-and-seek with P Diddy\n"
                    "/impulse - Drops a hot take and freezes the chat\n"
                    "/debate <topic> - The bot argues with itself over a topic```"
                )
            elif selected == "🤖 AI & Chat":
                embed = discord.Embed(title="🤖 AI & Chat Commands", color=discord.Color.blue())
                embed.description = (
                    "```/ai <prompt> [attachment] - Chat with AI (supports text and optional image)\n"
                    "/talk <message> - Text-to-speech using ElevenLabs\n"
                    "/describe <attachment> - Describe an attached image using AI\n"
                    "/pulse - Analyzes the current 'vibe' of the channel\n"
                    "/recap - Get a recap of topics discussed with AI today```"
                )
            elif selected == "🖼️ Media":
                embed = discord.Embed(title="🖼️ Media Commands", color=discord.Color.purple())
                embed.description = (
                    "```/image <prompt> - Generate an image using AI\n"
                    "/youtube <query> - Search YouTube videos\n"
                    "/daily - Get your daily motivation quote```"
                )
            elif selected == "⚙️ Utility":
                embed = discord.Embed(title="⚙️ Utility Commands", color=discord.Color.green())
                embed.description = (
                    "```/ping - Responds with the bot's latency\n"
                    "/timezone <time_input> - Convert a time to a dynamic Discord tag\n"
                    "/remind <seconds> <task> - Set a reminder\n"
                    "/countdown [days] [hours] [minutes] [seconds] - Start a live countdown timer\n"
                    "/feedback <category> <message> [command_name] [anonymous] [attachment] - Send feedback\n"
                    "/afk [reason] - Add [AFK] to your nickname```"
                )
            elif selected == "🔧 Administration":
                embed = discord.Embed(title="🔧 Administration Commands", color=discord.Color.orange())
                embed.description = (
                    "```/setannounce <channel> - Set the announcement channel\n"
                    "/invite-track <channel> - Set the welcome channel\n"
                    "/invites - View invite statistics for the server\n"
                    "/quiet - Enable quiet mode (owner only)\n"
                    "/unquiet - Disable quiet mode (owner only)\n"
                    "/setstatus <status> - Set bot status (owner only)\n"
                    "/sync - Sync slash commands (owner only)\n"
                    "/close - Shutdown the bot (owner only)```"
                )
            elif selected == "🛡️ Moderation":
                embed = discord.Embed(title="🛡️ Moderation Commands", color=discord.Color.red())
                embed.description = (
                    "```/purge <amount> - Clears specified number of messages\n"
                    "/kick <member> [reason] - Kicks a member from the server\n"
                    "/ban <member> [reason] - Bans a member from the server\n"
                    "/timeout <member> <duration> [reason] - Timeout (mute) a member\n"
                    "/untimeout <member> [reason] - Remove timeout from a member\n"
                    "/unban <member> - Unbans a member from the server\n"
                    "/warnings <user> - Display warnings for a user```"
                )
            elif selected == "ℹ️ Information":
                embed = discord.Embed(title="ℹ️ Information Commands", color=discord.Color.teal())
                embed.description = (
                    "```/serverinfo - Displays information about the server\n"
                    "/info [member] - Shows profile information and badges for a user\n"
                    "/recover - Recover the last deleted message in this channel (admin/owner only)\n"
                    "/help - Show this command list```"
                )

            embed.set_thumbnail(url="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExaTJ3aWVpa2o2dW5sM3htZzBlNmc2bG4wMHkycXp5NHVjZWE0cTF0NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WiIuC6fAOoXD2/giphy.gif")
            await interaction.response.edit_message(embed=embed, view=self.view)

    class CommandView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=60)
            self.add_item(CommandSelect())

    embed = discord.Embed(title="🤖 Bot Help", description="Select a category below to see all commands!", color=discord.Color.purple())
    embed.set_thumbnail(url="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExaTJ3aWVpa2o2dW5sM3htZzBlNmc2bG4wMHkycXp5NHVjZWE0cTF0NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WiIuC6fAOoXD2/giphy.gif")
    view = CommandView()
    await interaction.response.send_message(embed=embed, view=view)

# Event: AutoMod Action Logging
@bot.event
async def on_automod_action_execution(payload):
    """Logs when a Native AutoMod rule is triggered"""
    log_channel = discord.utils.get(payload.guild.channels, name="mod-logs")
    if log_channel:
        embed = discord.Embed(title="AutoMod Action", color=discord.Color.orange())
        embed.add_field(name="User", value=f"<@{payload.user_id}>")
        embed.add_field(name="Rule", value=payload.rule_name)
        embed.add_field(name="Action", value=payload.action.type.name)
        embed.set_footer(text=f"Triggered in #{payload.channel_id}")
        await log_channel.send(embed=embed)

# Error handling
@bot.event
async def on_command_error(ctx, error):
    """
    This event is triggered whenever a command raises an error.
    """
    # Prevent any commands with local handlers from being handled here
    if hasattr(ctx.command, 'on_error'):
        return

    # Allows us to check for original exceptions raised and sent to CommandInvokeError.
    # If nothing is found. We keep the exception passed to on_command_error.
    error = getattr(error, 'original', error)

    # Case: Command not found
    if isinstance(error, commands.CommandNotFound):
        return # Usually best to ignore this to avoid spamming the chat

    # Case: Missing required arguments (e.g., !ban without a user)
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Missing argument: Please specify `{error.param.name}`.")

    # Case: User doesn't have permission (e.g., trying to kick without Admin)
    elif isinstance(error, commands.MissingPermissions):
        perms = ", ".join(error.missing_permissions)
        await ctx.send(f"🚫 You don't have permission to do that! (Missing: {perms})")

    # Case: Bot doesn't have permission
    elif isinstance(error, commands.BotMissingPermissions):
        perms = ", ".join(error.missing_permissions)
        await ctx.send(f"⚠️ I can't do that. I need the `{perms}` permission.")

    # Case: Command is on cooldown
    elif isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"⏳ Slow down! Try again in {error.retry_after:.2f} seconds.")

    # Case: Bad Argument (e.g., passing a string where a number is expected)
    elif isinstance(error, commands.BadArgument):
        await ctx.send("📋 Invalid argument. Please check your input and try again.")

    # Case: Catch-all for everything else
    else:
        print(f'Ignoring exception in command {ctx.command}:', error)
        await ctx.send("🆘 An unexpected error occurred. My developers have been notified.")

# Run the bot
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    TOKEN = os.getenv('DISCORD_TOKEN')
    
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found in .env file")
        exit(1)
    
    try:
        bot.run(TOKEN)
    except discord.LoginFailure as e:
        print(f"Failed to log in: {e}")
        print("Please check your DISCORD_TOKEN in the .env file. It might be invalid or expired.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
