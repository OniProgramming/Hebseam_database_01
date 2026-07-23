print("=============================")
print("HANGMAN GAME".center(30))
print("=============================")
import random
word_list = ["apple", "banana", "orange", "grapes", "cherry", "mango", "peach", "tomato", "potato", "carrot", "onion", "garlic", "pepper", "lettuce", "cabbage", "broccoli", "spinach", "pumpkin", "cucumber", "avocado", "pineapple", "watermelon", "strawberry", "blueberry", "raspberry", "blackberry", "computer", "keyboard", "monitor", "printer", "speaker", "microphone", "headphones", "laptop", "desktop", "charger", "battery", "internet", "website", "browser", "python", "program", "coding", "function", "variable", "integer", "string", "boolean", "library", "package", "module", "network", "server", "client", "database", "science", "biology", "physics", "chemistry", "history", "geography", "language", "english", "germany", "france", "romania", "america", "canada", "mexico", "brazil", "argentina", "australia", "japan", "korea", "china", "india", "thailand", "vietnam", "singapore", "teacher", "student", "school", "college", "university", "lesson", "chapter", "notebook", "backpack", "library", "pencil", "eraser", "marker", "crayon", "window", "curtain", "blanket", "pillow", "mattress", "kitchen", "bedroom", "bathroom", "hallway", "garden", "garage", "ceiling", "flooring", "carpet", "hammer", "screwdriver", "wrench", "pliers", "drilling", "builder", "engineer", "doctor", "nurse", "lawyer", "farmer", "driver", "captain", "pilot", "artist", "musician", "dancer", "painter", "writer", "journal", "newspaper", "magazine", "television", "cinema", "theater", "concert", "festival", "holiday", "vacation", "journey", "travel", "airport", "station", "railway", "highway", "bridge", "tunnel", "mountain", "valley", "forest", "desert", "island", "ocean", "river", "stream", "waterfall", "volcano", "glacier", "penguin", "elephant", "giraffe", "tiger", "leopard", "panther", "monkey", "gorilla", "kangaroo", "dolphin", "whale", "octopus", "shark", "salmon", "rabbit", "hamster", "squirrel", "buffalo", "chicken", "turkey", "penguin", "ostrich", "alligator", "crocodile", "lizard", "turtle", "spider", "beetle", "butterfly", "dragonfly", "mosquito", "blanket", "freedom", "justice", "honesty", "kindness", "courage", "respect", "success", "failure", "victory", "defeat", "weather", "sunshine", "rainfall", "snowfall", "thunder", "lightning", "hurricane", "tornado", "earthquake", "adventure", "treasure", "mystery", "fantasy", "kingdom", "castle", "village", "country", "capital", "soldier", "general", "commander", "mission", "battle", "weapon", "shield", "helmet", "armor", "warrior", "knight", "dragon", "wizard", "monster", "zombie", "vampire", "pirates", "captain", "diamond", "emerald", "crystal", "golden", "silver", "bronze", "copper", "plastic", "wooden", "marble", "granite", "sandstone", "factory", "machine", "engine", "vehicle", "bicycle", "scooter", "airplane", "helicopter", "rocket", "satellite", "planet", "galaxy", "universe", "asteroid", "comet", "nebula", "oxygen", "hydrogen", "nitrogen", "carbon", "protein", "vitamin", "mineral", "energy", "gravity", "quantum", "formula", "equation", "decimal", "fraction", "holiday", "birthday", "wedding", "anniversary", "celebrate", "surprise", "present", "friend", "family", "brother", "sister", "mother", "father", "grandma", "grandpa", "cousin", "neighbor", "manager", "director", "company", "business", "meeting", "project", "account", "finance", "payment", "balance", "receipt", "message", "picture", "drawing", "painting", "reading", "writing", "running", "walking", "jumping", "swimming", "cycling", "football", "baseball", "cricket", "tennis", "volleyball", "basketball", "badminton", "hockey", "wrestling", "boxing", "karate", "taekwondo", "judo", "archery", "fishing", "camping", "climbing", "explorer", "adoption", "reaction", "solution", "problem", "decision", "question", "answer", "example", "practice", "exercise", "learning", "knowledge", "wisdom", "memory", "imagine", "creative", "careful", "powerful", "beautiful", "dangerous", "friendly", "helpful", "important", "different", "possible", "excellent", "brilliant"]
chosen_word = random.choice(word_list)
print("The skeleton of your word is:")

skeleton = []
if len(chosen_word) <= 8:
    first_hint = random.choice(chosen_word)
    second_hint = random.choice(chosen_word)
    while first_hint == second_hint:
        second_hint = random.choice(chosen_word)
    for letter in chosen_word:
        if first_hint == letter:
            skeleton.append(first_hint)
        elif second_hint == letter:
            skeleton.append(second_hint)
        else:
            skeleton.append("_")
if len(chosen_word) > 8:
    first_hint = random.choice(chosen_word)
    second_hint = random.choice(chosen_word)
    third_hint = random.choice(chosen_word)
    while first_hint == second_hint ==  third_hint:
        first_hint = random.choice(chosen_word)
        second_hint = random.choice(chosen_word)
        third_hint = random.choice(chosen_word)
    for letter in chosen_word:
        if letter == first_hint:
            skeleton.append(first_hint)
        elif letter == second_hint:
            skeleton.append(second_hint)
        elif letter == third_hint:
            skeleton.append(third_hint)
        else:
            skeleton.append("_")
print(" ".join(skeleton))
input("Enter to continue: ")
listed_word = list(chosen_word)
tries = 6
while skeleton != list(chosen_word):
    guess = input("Enter the next letter you think is in the word: ")
    if guess in chosen_word:
        for x in range (len(chosen_word)):
            if chosen_word[x] == guess:
                skeleton[x] = guess
        print ("Great! The new word is"," ".join(skeleton))
        print("")
    else:
        tries -= 1
        print("Wrong letter, you have",tries,"out of 6 tries remaining")
        print("")
    if tries == 0:
        print("Game over! You are out of tries :(")
        print("The correct word was",chosen_word)
        break
    elif skeleton == list(chosen_word):
        print("Great job! The word you have found is",chosen_word)
        break
    else:
        continue