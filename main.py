import disnake
from disnake.ext import commands
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import string
nltk.download('punkt')
nltk.download('wordnet')

ignoreLetters = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

clientIntents = disnake.Intents.default()
clientIntents.message_content = True
clientIntents.members = True
client = commands.Bot(command_prefix="!" , intents=clientIntents, help_command=None)

TIPPS = [
    "Get help from a staff member with /staffhelp.",
    "This is a Beta Version!",
    "Make your questions simple & easy to understand."
    "Check the FAQ channel for more help."
]

memory = []
context = {}

@client.event
async def on_ready():
    await client.change_presence(
            activity=disnake.Activity(
                type=disnake.ActivityType.playing,
                name=f'BETA - Venady AI'),
            status=disnake.Status.online)

THRESHOLD = 0.95

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.channel.id == 1087444800816566353:
        msg = message.content.lower()
        wordsList = nltk.word_tokenize(msg)
        wordsList = [lemmatizer.lemmatize(word) for word in wordsList if word not in ignoreLetters]
        bag = [0] * len(words)
        for w in wordsList:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1

        res = model.predict(np.array([bag]))[0]
        resultsIndex = np.argmax(res)
        tag = classes[resultsIndex]

            # Check if model prediction is above threshold
        if res[resultsIndex] > THRESHOLD:
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    response = random.choice(intent['responses'])
                    break
        else:
            response = "I'm sorry, I didn't understand your request or I cant find a answer. Would you please try rephrasing your question with more specificity? Alternatively, you can create a ticket to speak with a staff member who may be able to assist you further."

        em = disnake.Embed(
                            title="Venady Support AI (BETA) <a:1134verifiedanimated:1056889498081964072>",
                            description=f"Answer: **{response}** \n Developer Info: Threshold - {res[resultsIndex]} \n \n *Tipp: If the Support AI cant help you, create a ticket.* \n",
                            color=0x2F3136
                        )
        await message.channel.send(embed=em)


        
    if message.channel.category_id == 981634879433355274:
        bypass_roles = ["Staff"]
        if any(role.name in bypass_roles for role in message.author.roles):
            return
        if message.author.bot:
            return
        channel_id = message.channel.id
        with open('forbidden_channels.json', 'r') as f:
            data = json.load(f)
        if 'channel_ids' in data and channel_id not in data['channel_ids']:
            msg = message.content.lower()
            wordsList = nltk.word_tokenize(msg)
            wordsList = [lemmatizer.lemmatize(word) for word in wordsList if word not in ignoreLetters]
            bag = [0] * len(words)
            for w in wordsList:
                for i, word in enumerate(words):
                    if word == w:
                        bag[i] = 1

            res = model.predict(np.array([bag]))[0]
            resultsIndex = np.argmax(res)
            tag = classes[resultsIndex]

                # Check if model prediction is above threshold
            if res[resultsIndex] > THRESHOLD:
                for intent in intents['intents']:
                    if intent['tag'] == tag:
                        response = random.choice(intent['responses'])
                        break
            else:
                response = "I'm sorry, I didn't understand your request or I can't find an answer. Could you please try rephrasing your question with more specificity? Alternatively, you can use the command /staffhelp to speak with a staff member who may be able to assist you further."
            em = disnake.Embed(
                                        title="Venady Support AI (BETA) <a:1134verifiedanimated:1056889498081964072>",
                                        description=f"Answer: **{response}** \n Developer Info: Threshold - {res[resultsIndex]} \n \n *Tipp: {random.choice(TIPPS)}* \n",
                                        color=0x2F3136
                                    )
            await message.channel.send(embed=em)

@client.slash_command()
async def staffhelp(inter):
    if inter.channel.category_id == 981634879433355274:
        channel_id = inter.channel.id
        with open('forbidden_channels.json', 'r') as f:
            data = json.load(f)
        if 'channel_ids' not in data:
            data['channel_ids'] = []  
        if channel_id not in data['channel_ids']:
            data['channel_ids'].append(channel_id) 
            with open('forbidden_channels.json', 'w') as f:
                json.dump(data, f)
        bypass_roles = ["Staff"]
        if not any(role.name in bypass_roles for role in inter.author.roles):
            role_id = 1083806940175536158
            role = inter.guild.get_role(role_id)
            role_mention = role.mention
            await inter.send(f"AI disabled! A staff will help you now.")
            notify = await inter.bot.fetch_channel(975737813314191381)
            await notify.send(f"{inter.author} need help in ticket <#{channel_id}>")
        else:
            await inter.send(f"AI disabled!")
    else:
        await inter.send("This command only works in support tickets.")

client.run("TOKEN")

